import multiprocessing as mp
import os
import numpy as np
import torch
import torch.optim as optim
import faiss
import time
from torch.utils.data import TensorDataset, DataLoader
from multiprocessing.connection import Connection as Conn
from typing import Sequence, Callable
from argparse import Namespace
from tqdm import tqdm
import sys
from sklearn.decomposition import PCA
import contextlib
import io
import math
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap

sys.path.append("./")
sys.path.append("../../")

from pyjuice.utils import BitSet


def unique(data):
    assert len(data.shape) == 2
    b = np.ascontiguousarray(data).view(
        np.dtype((np.void, data.dtype.itemsize * data.shape[1]))
    )
    u = np.unique(b).view(data.dtype).reshape(-1, data.shape[1])
    return u


def cluster_imbalanceness(cluster_ids):
    """
    Calculate the imbalance measure based on cluster IDs.
    
    Arguments:
    cluster_ids -- Cluster IDs for each sample (tensor of shape [n_samples])
    
    Returns:
    imbalance_measure -- Imbalance measure based on cluster sizes (float)
    """
    
    # Get the unique cluster IDs and their counts
    unique_clusters, counts = torch.unique(cluster_ids, return_counts=True)
    
    # Calculate the cluster sizes
    cluster_sizes = counts.float()
    
    # Calculate the imbalance measure (e.g., coefficient of variation)
    imbalance_measure = torch.std(cluster_sizes) / torch.mean(cluster_sizes)
    
    return imbalance_measure.item()


def get_rmse(data, centroids):
    dists = np.power(data, 2).sum(axis = 1)[:,None] + np.power(centroids, 2).sum(axis = 1)[None,:] - 2 * np.dot(data, centroids.transpose(1, 0))
    dists = dists.clip(min = 1e-6)
    return np.sqrt(dists.min(axis = 1)).mean() / data.shape[1], np.sqrt(dists).mean() / data.shape[1]


class MPWorker(mp.Process):
    def __init__(self, pipe: Conn, device_ids: Sequence[int], model_constructor: Callable, worker_args: Namespace):
        super(MPWorker, self).__init__()

        self.pipe = pipe

        self.model_constructor = model_constructor

        self.device_ids = device_ids
        self.worker_args = worker_args

        self.model = None

    def init_process(self):

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(lambda i: str(i), self.device_ids))

        self.model = self.model_constructor(self.worker_args)
        self.model.cuda()
        self.model.eval()

    def run(self):

        self.init_process()

        while True:

            command, args = self._recv_message()

            if command == "get_subset_cids":
                task_id, data, cond_data, scopes, n_clusters = args

                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                if isinstance(cond_data, np.ndarray):
                    cond_data = torch.from_numpy(cond_data)

                scopes = [scope.to_list() for scope in scopes]
                num_scopes = len(scopes)

                data_loader = DataLoader(
                    dataset = TensorDataset(data, cond_data),
                    batch_size = self.worker_args.batch_size,
                    shuffle = False
                )

                # Get subset embeddings
                print(f"> [task {task_id:04d}] Collecting embeddings...")
                latent_embds = None
                latent_embds_gpu = None

                start_time = time.time()
                total_ll = 0.0
                last_log = 0
                processed = 0
                all_latent_embds = []
                for scope in scopes:
                    target_vars = torch.tensor(scope).cuda()
                    sid = 0
                    for x, cond_x in data_loader:
                        x = x.cuda()
                        cond_x = cond_x.cuda()
                        eid = sid + x.size(0)

                        embds, ll = self.model.get_subset_embeddings(x, cond_x, target_vars)
                        total_ll += ll.item()

                        if latent_embds is None:
                            max_data_size = int(math.ceil(min((1024**3) // 4 // embds.size(1) // embds.size(2), data.size(0))))
                            
                            latent_embds = torch.zeros([data.size(0), embds.size(1), embds.size(2)])
                            latent_embds_gpu = torch.zeros([max_data_size, embds.size(1), embds.size(2)]).cuda()
                            gpu_sid = 0

                        if eid - gpu_sid <= max_data_size:
                            latent_embds_gpu[sid-gpu_sid:eid-gpu_sid,:,:] = embds
                        else:
                            latent_embds[gpu_sid:sid,:,:] = latent_embds_gpu[:sid-gpu_sid,:,:].cpu()
                            gpu_sid = sid
                            latent_embds_gpu[sid-gpu_sid:eid-gpu_sid,:,:] = embds

                        sid = eid

                        processed += x.size(0)
                        if processed - last_log > self.worker_args.gpt_log_every:
                            curr_time = time.time()
                            remaining_time = (curr_time - start_time) / processed * (data.size(0) * len(scopes) - processed)
                            print(f"> [task {task_id:04d}] Processed {processed:08d}/{(data.size(0) * len(scopes)):08d} - {remaining_time:.2f}s remaining")
                            last_log = processed

                    latent_embds[gpu_sid:sid,:,:] = latent_embds_gpu[:sid-gpu_sid,:,:].cpu()
                    all_latent_embds.append(latent_embds)

                    latent_embds = None
                    latent_embds_gpu = None

                all_latent_embds = torch.stack(all_latent_embds, dim = 1)

                print(f"> [task {task_id:04d}] Number of target variables: {len(scopes[0])}")
                per_dim_ll = total_ll / len(data_loader) / len(scopes)
                print(f"> [task {task_id:04d}] Average per-dim LL: {per_dim_ll}")
                latent_mean, latent_std = all_latent_embds.mean(), all_latent_embds.std()
                print(f"> [task {task_id:04d}] Extracted embeddings statistics: {latent_mean:.4f}±{latent_std:.4f}")

                # Run dimension reduction
                print(f"> [task {task_id:04d}] Do dimension reduction from {all_latent_embds.size(3)} dims to {self.worker_args.reduced_dim} dims...")
                latent_embds = all_latent_embds.numpy()

                reduced_dim = max(min(self.worker_args.reduced_dim, self.worker_args.max_kmeans_feature_dim // latent_embds.shape[2]), 16)
                ld_embds = np.zeros([data.size(0), num_scopes, latent_embds.shape[2], reduced_dim], dtype = np.float32)
                tot_num_samples = data.size(0) * num_scopes
                with contextlib.redirect_stdout(io.StringIO()):
                    pca_models = []
                    for vid in range(latent_embds.shape[2]):
                        x = latent_embds[:,:,vid,:].reshape(tot_num_samples, -1)
                        pca = PCA(n_components = reduced_dim).fit(x)
                        ld_embds[:,:,vid,:] = pca.transform(x).reshape(data.size(0), num_scopes, reduced_dim)

                        pca_models.append(pca)
                ld_embds = ld_embds.reshape(data.size(0), num_scopes, -1)

                # Run Kmeans
                print(f"> [task {task_id:04d}] Clustering data of size [{ld_embds.shape[0]},{ld_embds.shape[1]},{ld_embds.shape[2]}] into {n_clusters} clusters...")
                
                ld_embds = ld_embds.reshape(tot_num_samples, -1)
                if ld_embds.shape[0] > self.worker_args.max_kmeans_num_samples:
                    frac = self.worker_args.max_kmeans_num_samples / ld_embds.shape[0]
                    mask = np.random.random([ld_embds.shape[0]]) < frac
                    all_embds = ld_embds[mask,:]
                else:
                    all_embds = ld_embds
                all_embds = np.ascontiguousarray(all_embds)
                vecs = unique(all_embds)

                if vecs.shape[0] < n_clusters:
                    vecs = np.repeat(vecs, int(n_clusters // vecs.shape[0]) + 1, axis = 0)

                kmeans = faiss.Kmeans(
                    vecs.shape[1], n_clusters, niter = self.worker_args.kmeans_niter, nredo = self.worker_args.kmeans_nredo,
                    verbose = False, max_points_per_centroid = vecs.shape[0] // n_clusters, 
                    gpu = True
                )
                kmeans.train(vecs)

                _, idx2cluster = kmeans.index.search(ld_embds, 1)
                cids = idx2cluster.reshape(ld_embds.shape[0])
                centroids = kmeans.centroids.copy()

                imbalanceness = cluster_imbalanceness(torch.from_numpy(cids))
                kmeans_rmse, full_rmse = get_rmse(ld_embds, centroids)

                print(f"> [task {task_id:04d}] Kmeans rmse: {kmeans_rmse:.4f}/{full_rmse:.4f}, imbalanceness: {imbalanceness:.4f}")
                print(f"> [task {task_id:04d}] Task completed!")

                cids = cids.reshape(data.size(0), num_scopes)

                scopes = [BitSet.from_array(scope) for scope in scopes]

                self._send_message("get_subset_cids", (task_id, scopes, cids, centroids, pca_models))

            elif command == "get_xprod_assignments":
                # target_cids:      [B]
                # chs_cids:         [B, n_chs]
                # target_centroids: [n_clusters, n_embd]
                # chs_centroids:    [n_clusters, n_embd] * n_chs
                task_id, target_cids, chs_cids, target_centroids, chs_centroids, centroid_dim = args

                if isinstance(target_cids, torch.Tensor):
                    target_cids = target_cids.numpy()
                if isinstance(chs_cids, torch.Tensor):
                    chs_cids = chs_cids.numpy()
                if isinstance(target_centroids, torch.Tensor):
                    target_centroids = target_centroids.numpy()
                if isinstance(chs_centroids[0], torch.Tensor):
                    chs_centroids = [item.numpy() for item in chs_centroids]

                B = target_cids.shape[0]
                n_chs = chs_cids.shape[1]

                new_chs_centroids = np.zeros([chs_centroids[0].shape[0], centroid_dim, len(chs_centroids)], dtype = np.float32)
                for i in range(len(chs_centroids)):
                    if centroid_dim <= chs_centroids[i].shape[1]:
                        new_chs_centroids[:,:,i] = PCA(n_components = centroid_dim).fit_transform(chs_centroids[i])
                    else:
                        m = chs_centroids[i].shape[1]
                        new_chs_centroids[:,:m,i] = chs_centroids[i]
                chs_centroids = new_chs_centroids

                xprod_cids = set()

                num_target_cids = target_cids.max() + 1
                last_logging = 0
                t_start = time.time()
                for target_cid in range(num_target_cids):
                    mask = (target_cids == target_cid)
                    cids = chs_cids[mask,:]

                    if cids.shape[0] < 5:
                        continue

                    n_clusters = int(max(min(self.worker_args.n_cls_per_target, cids.shape[0] // (B // num_target_cids // \
                        self.worker_args.num_configs_per_target_cls)), 1))

                    data = np.zeros([cids.shape[0], n_chs, centroid_dim], dtype = np.float32)
                    for i in range(n_chs):
                        data[:,i,:] = chs_centroids[cids[:,i],:,i]
                    data = np.ascontiguousarray(data.reshape(cids.shape[0], n_chs * centroid_dim))

                    # Run Kmeans
                    with contextlib.redirect_stdout(io.StringIO()):
                        kmeans_method = "sklearn"
                        if kmeans_method == "faiss":
                            kmeans = faiss.Kmeans(
                                data.shape[1], n_clusters, niter = self.worker_args.kmeans_niter, nredo = self.worker_args.kmeans_nredo,
                                verbose = False, max_points_per_centroid = data.shape[0] // n_clusters, 
                                gpu = True
                            )
                            kmeans.train(data)

                            centroids = kmeans.centroids.copy()
                        elif kmeans_method == "sklearn":
                            vecs = unique(data)

                            if vecs.shape[0] < n_clusters:
                                centroids = vecs.copy()
                                n_clusters = centroids.shape[0]
                            else:
                                kmeans = KMeans(n_clusters = n_clusters, n_init = "auto", init = "k-means++").fit(vecs)

                                centroids = kmeans.cluster_centers_.copy()
                        else:
                            raise ValueError()

                    centroids = centroids.reshape(n_clusters, n_chs, centroid_dim)
                    group_cids = np.zeros([n_clusters, n_chs], dtype = np.int64)
                    for i in range(n_chs):
                        dists = np.power(centroids[:,i,:], 2).sum(axis = 1)[:,None] + np.power(chs_centroids[:,:,i], 2).sum(axis = 1)[None,:] - \
                            2 * np.dot(centroids[:,i,:], chs_centroids[:,:,i].transpose(1, 0))

                        ids = [item[i] for item in xprod_cids]
                        nsamples = int(dists.shape[0] // 6)
                        dists[:nsamples,ids] = 1e6

                        group_cids[:,i] = np.argmin(dists, axis = 1)

                    for i in range(n_clusters):
                        xprod_cids.add(tuple(group_cids[i,:]))

                    if target_cid - last_logging > 100:
                        cov = []
                        for i in range(n_chs):
                            cov.append(len(set([item[i] for item in xprod_cids])))

                        t_curr = time.time()
                        remaining_time = (t_curr - t_start) / target_cid * (num_target_cids - target_cid)

                        print(f"> [task {task_id:04d}] processed targets: {target_cid:04d}/{num_target_cids:04d} - {len(xprod_cids):05d} product configs - {remaining_time:.2f}s remaining")
                        print(f"> [task {task_id:04d}] child clusters coverage:", cov)

                        last_logging = target_cid

                self._send_message("get_xprod_assignments", (task_id, list(xprod_cids)))

            elif command == "pred_cids":
                task_id, data, cond_data, scope, centroids, pca_models = args

                print(f"> [task {task_id:04d}] Collecting embeddings...")
                latent_embds = None
                latent_embds_gpu = None

                num_samples = data.size(0)

                data_loader = DataLoader(
                    dataset = TensorDataset(data, cond_data),
                    batch_size = self.worker_args.batch_size,
                    shuffle = False
                )

                target_vars = torch.tensor(scope.to_list()).cuda()

                start_time = time.time()
                total_ll = 0.0
                last_log = 0
                sid = 0
                for x, cond_x in data_loader:
                    x = x.cuda()
                    cond_x = cond_x.cuda()
                    eid = sid + x.size(0)

                    embds, ll = self.model.get_subset_embeddings(x, cond_x, target_vars)
                    total_ll += ll.item()

                    if latent_embds is None:
                        max_data_size = min((1024**3) // 4 // embds.size(1) // embds.size(2), num_samples)
                        
                        latent_embds = torch.zeros([num_samples, embds.size(1), embds.size(2)])
                        latent_embds_gpu = torch.zeros([max_data_size, embds.size(1), embds.size(2)]).cuda()
                        gpu_sid = 0

                    if eid - gpu_sid <= max_data_size:
                        latent_embds_gpu[sid-gpu_sid:eid-gpu_sid,:,:] = embds
                    else:
                        latent_embds[gpu_sid:sid,:,:] = latent_embds_gpu[:sid-gpu_sid,:,:].cpu()
                        gpu_sid = sid
                        latent_embds_gpu[sid-gpu_sid:eid-gpu_sid,:,:] = embds

                    sid = eid

                    if sid - last_log > self.worker_args.gpt_log_every:
                        curr_time = time.time()
                        remaining_time = (curr_time - start_time) / sid * (num_samples - sid)
                        print(f"> [task {task_id:04d}] Processed {sid:08d}/{num_samples:08d} - {remaining_time:.2f}s remaining")
                        last_log = sid

                print(f"> [task {task_id:04d}] Embeddings collected...")

                latent_embds[gpu_sid:sid,:,:] = latent_embds_gpu[:sid-gpu_sid,:,:].cpu()
                latent_embds = np.ascontiguousarray(latent_embds.numpy())

                # Apply dimension reduction
                reduced_dim = pca_models[0].get_params()["n_components"]
                ld_embds = np.zeros([num_samples, latent_embds.shape[1], reduced_dim], dtype = np.float32)
                with contextlib.redirect_stdout(io.StringIO()):
                    for vid in range(latent_embds.shape[1]):
                        x = latent_embds[:,vid,:]
                        ld_embds[:,vid,:] = pca_models[vid].transform(x)
                ld_embds = ld_embds.reshape(num_samples, -1)

                index = faiss.IndexFlatL2(centroids.shape[1])
                index.add(np.ascontiguousarray(centroids))
                _, cids = index.search(ld_embds, 1)
                cids = cids.ravel()

                print(f"> [task {task_id:04d}] Task completed...")

                self._send_message("pred_cids", (task_id, scope, cids))

            elif command == "update_args":
                key, val = kwargs

                self.args.__dict__[key] = val

                self._send_message("update_args", None)

            elif command == "kill":
                print("killed")
                return

            else:
                raise NotImplementedError(f"Unknown {command}")

    def _send_message(self, command, kwargs):
        self.pipe.send((command, kwargs))

    def _recv_message(self):
        self.pipe.poll(None) # wait until new message is received
        command, kwargs = self.pipe.recv()

        return command, kwargs


class MPManager(mp.Process):
    def __init__(self, worker_devices, model_constructor, worker_args):
        super(MPManager, self).__init__()

        self.worker_devices = worker_devices
        self.model_constructor = model_constructor
        self.worker_args = worker_args

        self.num_workers = len(worker_devices)

        self.workers = []
        self.pipes = []
        for worker_idx in range(self.num_workers):
            parent_pipe, child_pipe = mp.Pipe()
            worker = MPWorker(
                child_pipe, 
                self.worker_devices[worker_idx], 
                self.model_constructor, 
                self.worker_args
            )

            self.workers.append(worker)
            self.pipes.append(parent_pipe)

        for worker in self.workers:
            worker.start()

    def run_tasks(self, tasks):

        workers_status = np.zeros([self.num_workers], dtype = bool)

        rets = [None for _ in range(len(tasks))]

        while np.any(workers_status) or len(tasks) > 0:

            time.sleep(1.0)

            # Check for completed tasks
            for worker_idx in range(self.num_workers):
                command, ret_kwargs = self._recv_message_nonblocking(worker_idx)
                if command is not None:
                    workers_status[worker_idx] = False
                else:
                    continue

                if command == "get_subset_cids":
                    task_id, scopes, cids, centroids, pca_models = ret_kwargs

                    rets[task_id] = (scopes, cids, centroids, pca_models)

                elif command == "pred_cids":
                    task_id, scope, cids = ret_kwargs

                    rets[task_id] = (scope, cids)

                elif command == "get_xprod_assignments":
                    task_id, xprod_cids = ret_kwargs

                    rets[task_id] = xprod_cids

                else:
                    raise NotImplementedError(f"Got unknown command {command}.")

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

    def kill(self):
        for worker_idx in range(self.num_workers):
            self._send_message(worker_idx, "kill")

    def __del__(self):
        self.kill()
