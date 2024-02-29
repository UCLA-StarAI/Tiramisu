import multiprocessing as mp
import os
import numpy as np
import torch
import faiss
from multiprocessing.connection import Connection as Conn


class KmeansWorker(mp.Process):
    def __init__(self, pipe: Conn, device_id: int, args):
        super(KmeansWorker, self).__init__()

        self.pipe = pipe

        self.device_id = device_id
        self.args = args

    def run(self):

        while True:

            command, args = self._recv_message()

            if command == "kmeans_get_cids":
                task_id, data = args
                data = np.ascontiguousarray(data)

                print(f"> [task {task_id}] Running Kmeans...")
                kmeans = faiss.Clustering(data.shape[1], self.args.n_clusters)
                kmeans.verbose = False
                kmeans.niter = 100
                kmeans.nredo = 3
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = True
                cfg.device = self.device_id
                index = faiss.GpuIndexFlatL2(
                    faiss.StandardGpuResources(),
                    data.shape[1],
                    cfg
                )
                kmeans.train(data, index)
                centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(self.args.n_clusters, data.shape[1])

                index = faiss.IndexFlatL2(centroids.shape[1])
                index.add(np.ascontiguousarray(centroids))
                _, labels = index.search(data, 1)
                labels = labels.ravel()
                print(f"> [task {task_id}] Done!")

                self._send_message("kmeans_get_cids", (task_id, labels))

            elif command == "update_args":
                key, val = kwargs

                self.args.__dict__[key] = val

                self._send_message("update_args", None)

            elif command == "kill":
                return

            else:
                raise NotImplementedError()

    def _send_message(self, command, kwargs):
        self.pipe.send((command, kwargs))

    def _recv_message(self):
        self.pipe.poll(None) # wait until new message is received
        command, kwargs = self.pipe.recv()

        return command, kwargs


class KmeansManager(mp.Process):
    def __init__(self, device_ids, args):
        super(KmeansManager, self).__init__()

        self.device_ids = device_ids
        self.args = args

        self.num_workers = len(device_ids)

        self.workers = []
        self.pipes = []
        for worker_idx in range(self.num_workers):
            parent_pipe, child_pipe = mp.Pipe()
            worker = KmeansWorker(child_pipe, self.device_ids[worker_idx], self.args)

            self.workers.append(worker)
            self.pipes.append(parent_pipe)

        for worker in self.workers:
            worker.start()

    def run_tasks(self, tasks):

        workers_status = np.zeros([self.num_workers], dtype = bool)

        rets = [None for _ in range(len(tasks))]

        while np.any(workers_status) or len(tasks) > 0:

            # Check for completed tasks
            for worker_idx in range(self.num_workers):
                command, ret_kwargs = self._recv_message_nonblocking(worker_idx)
                if command is not None:
                    workers_status[worker_idx] = False
                else:
                    continue

                if command == "kmeans_get_cids":
                    task_id, labels = ret_kwargs

                    rets[task_id] = labels

                else:
                    raise NotImplementedError()

            # Assign tasks to empty workers
            while len(tasks) > 0 and not np.all(workers_status):
                worker_idx = np.where(~workers_status)[0][0]

                command, args = tasks.pop()
                task_id = len(tasks)

                self._send_message(worker_idx, command, (task_id, *args))

                workers_status[worker_idx] = True

        return rets

    def update_args(self, key, val):
        self.args.__dict__[key] = val
        for worker_idx in range(self.num_workers):
            self._send_message(worker_idx, "update_args", (key, val))

        for worker_idx in range(self.num_workers):
            command, _ = self._recv_message(worker_idx)
            assert command == "update_args"

    def _send_message(self, worker_idx, command, args = None):
        pipe = self.pipes[worker_idx]

        pipe.send((command, args))

    def _recv_message_nonblocking(self, worker_idx):
        pipe = self.pipes[worker_idx]

        if not pipe.poll():
            return None, None

        command, args = pipe.recv()

        return command, args

    def _recv_message(self, worker_idx):
        pipe = self.pipes[worker_idx]

        pipe.poll(None)

        command, args = pipe.recv()

        return command, args

    def __del__(self):
        for worker_idx in range(self.num_workers):
            self._send_message(worker_idx, "kill")