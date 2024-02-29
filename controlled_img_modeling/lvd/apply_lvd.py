import numpy as np
import torch
import pyjuice as juice
from pyjuice.nodes import CircuitNodes
from pyjuice.utils import BitSet
from pyjuice.nodes.methods.lvd_backend.counting import get_pairwise_count
from copy import deepcopy
import math
import functools
import uuid
import triton
import triton.language as tl
import os
import shutil
from tqdm import tqdm
import ray
import sys
import subprocess
import multiprocessing as mp
import time

sys.path.append("./")
sys.path.append("../../")
sys.path.append("../../external/taming-transformers/")


def update_duplicated(root_ns):
    for ns in root_ns:
        if ns.is_tied():
            source_ns = ns.get_source_ns()
            if ns.is_input():
                ns.dist = deepcopy(source_ns.dist)
            else:
                ns.edge_ids = deepcopy(source_ns.edge_ids)
                new_prod_ns = []
                for pns, source_pns in zip(ns.chs, source_ns.chs):
                    new_prod_ns.append(source_pns.duplicate(*pns.chs))

                ns.chs = new_prod_ns


def apply_lvd(root_ns, scope2group_id, scope_groups, sgroup_partition_scopes, 
              group_id2ids, sgroup_cids, xprod_cids, lv_centroids, 
              num_latents, num_cats, device = torch.device("cuda:0"), sparsity = 0.002):
    
    for ns in root_ns:
        if not ns.is_tied():
            if ns.is_input():
                group_id = scope2group_id[ns.scope]
                cids, nids = group_id2ids[group_id]

                counts = get_pairwise_count(
                    cids, nids, num_latents, num_cats, device = device
                ).cpu()

                ns.set_params(counts, normalize = True)

                print("> Completed an input node group")

            elif ns.is_sum():

                new_prod_ns = []
                new_params = []

                par_group_id = scope2group_id[ns.scope]
                for prod_ns in ns.chs:
                    n_chs = prod_ns.num_chs
                    chs_group_id = tuple(scope2group_id[cs.scope] for cs in prod_ns.chs)
                    group_partition = (par_group_id, chs_group_id)

                    prod_ids = xprod_cids[group_partition]
                    prod_ids = torch.tensor(prod_ids).long().reshape(-1, n_chs).to(device) # (num_prods, n_chs)
                    num_prods = prod_ids.size(0)

                    pns = juice.multiply(*prod_ns.chs, edge_ids = prod_ids)
                    new_prod_ns.append(pns)

                    # Assign parameters
                    target_group_cids, chs_group_cids = sgroup_cids[group_partition]
                    target_group_cids = target_group_cids.to(device)
                    chs_group_cids = chs_group_cids.to(device)

                    # (num_prods, n_chs, num_latents)
                    prod_id_scores = torch.zeros([num_prods, n_chs, num_latents], device = device)
                    for ch_id in range(n_chs):
                        scope = prod_ns.chs[ch_id].scope
                        group_id = scope2group_id[scope]
                        centroids = torch.from_numpy(lv_centroids[group_id]).to(device) # (num_latents, n_embd)

                        xs = centroids[prod_ids[:,ch_id],:]
                        ys = centroids
                        dists = torch.sqrt((xs.pow(2).sum(dim = 1)[:,None] + ys.pow(2).sum(dim = 1)[None,:] - 2 * torch.matmul(xs, ys.permute(1, 0))).clip(min = 0.0))
                        prod_id_scores[:,ch_id,:] = torch.softmax(-dists / dists.std(), dim = 1)

                    sum_params = torch.zeros([num_latents, num_prods], device = device)
                    for cid in range(num_latents):
                        mask = (target_group_cids == cid)
                        nsamples = mask.sum()
                        if nsamples > 0:
                            indices = chs_group_cids[mask,:][:,None,:,None].expand(-1, num_prods, -1, -1) # (nsamples, num_prods, n_chs, 1)
                            scores = prod_id_scores[None,:,:,:].expand(nsamples, -1, -1, -1)
                            sum_params[cid,:] = (scores.gather(3, indices)).squeeze(3).clip(min = 1e-6).log().sum(dim = 2).logsumexp(dim = 0) # (num_prods,)
                            sum_params[cid,:] -= sum_params[cid,:].max()
                            sum_params[cid,:] = sum_params[cid,:].exp()
                        else:
                            sum_params[cid,:] = 1.0 / num_prods

                    new_params.append(sum_params.cpu())

                # Assemble 
                sum_params = torch.cat(new_params, dim = 1)
                num_edges_per_node = int(math.ceil(sum_params.size(1) * sparsity))
                
                sum_edge_ids = torch.zeros([2, num_edges_per_node * num_latents], dtype = torch.long)
                params = torch.zeros([num_edges_per_node * num_latents], dtype = torch.float32)
                for i in range(num_latents):
                    sid, eid = i * num_edges_per_node, (i + 1) * num_edges_per_node
                    tk = torch.topk(sum_params[i,:], num_edges_per_node)

                    sum_edge_ids[0,sid:eid] = i
                    sum_edge_ids[1,sid:eid] = tk.indices
                    params[sid:eid] = tk.values

                ns.chs = new_prod_ns
                ns.edge_ids = sum_edge_ids
                ns.set_params(params)

                print("> Completed a sum node group")

    update_duplicated(root_ns)

    return root_ns


## LVD with group-wise pretraining ##


@triton.jit
def _weighted_counts_kernel(counts_ptr, xids_ptr, yids_ptr, weights_ptr, num_samples: tl.constexpr, n_ys: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_samples

    xid = tl.load(xids_ptr + offsets, mask = mask, other = 0)
    yid = tl.load(yids_ptr + offsets, mask = mask, other = 0)
    cid = xid * n_ys + yid
    weight = tl.load(weights_ptr + offsets, mask = mask, other = 0)

    tl.atomic_add(counts_ptr + cid, weight, mask = mask)


def weighted_counts(counts, xids, yids, weights):

    num_samples = xids.size(0)
    n_ys = counts.size(1)

    grid = lambda meta: (triton.cdiv(num_samples, meta['BLOCK_SIZE']),)

    _weighted_counts_kernel[grid](
        counts_ptr = counts,
        xids_ptr = xids,
        yids_ptr = yids,
        weights_ptr = weights,
        num_samples = num_samples,
        n_ys = n_ys,
        BLOCK_SIZE = 2048
    )


def get_prod_lls(chs_lls, prod_ids, device, batch_size = 2048):
    num_nodes = prod_ids.size(0)
    num_chs = prod_ids.size(1)
    num_samples = chs_lls[0].size(1)

    prod_ids = prod_ids.to(device)

    prod_lls = torch.zeros([num_nodes, num_samples], dtype = torch.float32)
    prod_lls_gpu = torch.zeros([num_nodes, batch_size], dtype = torch.float32, device = device)
    for sid in range(0, num_samples, batch_size):
        eid = min(sid + batch_size, num_samples)
        nb_samples = eid - sid
        prod_lls_gpu[:,:nb_samples] = 0.0
        for idx in range(num_chs):
            ch_lls = chs_lls[idx][:,sid:eid].to(device)
            prod_lls_gpu[:,:nb_samples] += ch_lls[prod_ids[:,idx],:]
        prod_lls[:,sid:eid] = prod_lls_gpu[:,:nb_samples].cpu()

    return prod_lls


def get_sum_lls(ns_chs_lls, ch_ids, probs, num_latents, device, batch_size = 2048):
    num_nodes = ch_ids.size(0)
    num_samples = ns_chs_lls.size(1)

    ch_ids = ch_ids.to(device)
    logprobs = probs.log().to(device)

    sum_lls = torch.zeros([num_latents, num_samples], dtype = torch.float32)
    for sid in range(0, num_samples, batch_size):
        eid = min(sid + batch_size, num_samples)
        chs_lls = ns_chs_lls[:,sid:eid].to(device)
        sum_lls_gpu = (chs_lls[ch_ids,:] + logprobs.unsqueeze(2)).logsumexp(dim = 1)
        sum_lls[:,sid:eid] = sum_lls_gpu.cpu()

    return sum_lls


def compute_ns_chs_lls_remote(scope, prod_chs_fnames, all_prod_ids, device):

    ns_chs_lls = []
    for idx, prod_fnames in enumerate(prod_chs_fnames):
        chs_lls = []
        for fname in prod_fnames:
            ch_lls = torch.from_numpy(np.load(fname))
            chs_lls.append(ch_lls)

        prod_ids = all_prod_ids[idx]
        prod_lls = get_prod_lls(chs_lls, prod_ids, device)

        ns_chs_lls.append(prod_lls)

    ns_chs_lls = torch.cat(ns_chs_lls, dim = 0)

    return scope, ns_chs_lls


def get_sum_lls_remote(scope, ns_chs_lls, ch_ids, probs, num_latents, device):

    sum_lls = get_sum_lls(ns_chs_lls, ch_ids, probs, num_latents, device)

    per_dim_ll = sum_lls.max(dim = 0).values.mean() / len(scope)
    print(f"> [Sum nodes with scope size {len(scope)}] Aveg best per dim LL: {per_dim_ll:.2f}")

    return scope, sum_lls


class SimpleWorker(mp.Process):
    def __init__(self, pipe, device_id):
        super(SimpleWorker, self).__init__()

        self.pipe = pipe
        self.device_id = device_id

        self.device = None

    def run(self):

        self.device = torch.device(f"cuda:{self.device_id}")

        while True:

            command, args = self._recv_message()

            if command == "compute_ns_chs_lls":

                task_id, scope, prod_chs_fnames, all_prod_ids = args
                scope, ns_chs_lls = compute_ns_chs_lls_remote(scope, prod_chs_fnames, all_prod_ids, device = self.device)

                self._send_message("compute_ns_chs_lls", (task_id, scope, ns_chs_lls))

            elif command == "get_sum_lls":

                task_id, scope, ns_chs_lls, ch_ids, probs, num_latents, tmp_folder = args
                scope, sum_lls = get_sum_lls_remote(scope, ns_chs_lls, ch_ids, probs, num_latents, device = self.device)

                temp_fname = os.path.join(tmp_folder, str(uuid.uuid4()) + ".npy")
                np.save(temp_fname, sum_lls.numpy())

                self._send_message("get_sum_lls", (task_id, scope, temp_fname))

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


class SimpleManager():
    def __init__(self, num_workers = 4):

        self.num_workers = min(torch.cuda.device_count(), num_workers)

        self.workers = []
        self.pipes = []
        for worker_idx in range(self.num_workers):
            parent_pipe, child_pipe = mp.Pipe()
            worker = SimpleWorker(
                child_pipe, 
                worker_idx
            )

            self.workers.append(worker)
            self.pipes.append(parent_pipe)

        for worker in self.workers:
            worker.start()
    
    def run_tasks(self, tasks):

        workers_status = np.zeros([self.num_workers], dtype = bool)

        rets = [None for _ in range(len(tasks))]

        pbar = tqdm(total = len(tasks))

        while np.any(workers_status) or len(tasks) > 0:

            time.sleep(1.0)

            # Check for completed tasks
            for worker_idx in range(self.num_workers):
                command, ret_kwargs = self._recv_message_nonblocking(worker_idx)
                if command is not None:
                    workers_status[worker_idx] = False
                else:
                    continue

                if command == "compute_ns_chs_lls":
                    task_id, scope, ns_chs_lls = ret_kwargs

                    rets[task_id] = (scope, ns_chs_lls)

                elif command == "get_sum_lls":
                    task_id, scope, temp_fname = ret_kwargs

                    rets[task_id] = (scope, temp_fname)

                else:
                    raise NotImplementedError(f"Got unknown command {command}.")

                pbar.update(1)

            # Assign tasks to empty workers
            while len(tasks) > 0 and not np.all(workers_status):
                worker_idx = np.where(~workers_status)[0][0]

                command, args = tasks.pop()
                task_id = len(tasks)

                self._send_message(worker_idx, command, (task_id, *args))

                workers_status[worker_idx] = True

        pbar.close()

        return rets

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


def apply_lvd_with_pretraining(root_ns, obs_data, scope2group_id, scope_groups,
                               xprod_cids, subset_full_cids, num_latents, num_cats, device = torch.device("cuda:0"), 
                               sparsity = 0.004, tmp_folder = "outputs/tmp/"):

    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
        
    os.mkdir(tmp_folder)

    manager = SimpleManager()

    total_num_vars = obs_data.size(1)
    
    ns_old2new = dict()

    # Mapping from scope to its corresponding input or sum nodes
    scope2ns = dict()
    for ns in root_ns:
        if ns.is_input() or ns.is_sum():
            assert ns.scope not in scope2ns
            scope2ns[ns.scope] = ns

    # Order the group ids such that it can be processed bottom-up
    group2layer_id = dict()
    group_layers = []
    for ns in root_ns:
        if ns.is_input():
            group_id = scope2group_id[ns.scope]
            if group_id not in group2layer_id:
                group2layer_id[group_id] = 0

                if len(group_layers) <= 0:
                    group_layers.append([])
                group_layers[0].append(group_id)
        elif ns.is_prod():
            group_id = scope2group_id[ns.scope]
            ch_group_ids = [scope2group_id[cs.scope] for cs in ns.chs]
            layer_id = max([group2layer_id[gid] for gid in ch_group_ids]) + 1

            if group_id not in group2layer_id:
                group2layer_id[group_id] = layer_id

                if len(group_layers) <= layer_id:
                    group_layers.append([])
                group_layers[layer_id].append(group_id)
            elif layer_id > group2layer_id[group_id]:
                group_layers[group2layer_id[group_id]].remove(group_id)
                group2layer_id[group_id] = layer_id

                if len(group_layers) <= layer_id:
                    group_layers.append([])
                group_layers[layer_id].append(group_id)

    # Sanity check
    for ns in root_ns:
        if ns.is_prod():
            group_id = scope2group_id[ns.scope]
            ch_group_ids = [scope2group_id[cs.scope] for cs in ns.chs]
            assert group2layer_id[group_id] > max([group2layer_id[gid] for gid in ch_group_ids])

    ordered_group_ids = functools.reduce(lambda x, y: x + y, group_layers)

    # Stores temp LLs for all nodes
    scope2lls_fname = dict()

    def get_lls_for_scope(scope):
        fname = scope2lls_fname[scope]
        return torch.from_numpy(np.load(fname))
    
    # Do group-wise bottom-up pretraining
    for group_id in ordered_group_ids:
        scopes = scope_groups[group_id]
        group_ns = [scope2ns[scope] for scope in scopes]

        if group_ns[0].is_input():

            num_samples = obs_data.size(0)

            input_params = torch.zeros([num_latents, num_cats], dtype = torch.float32, device = device)
            weights = torch.ones([num_samples], dtype = torch.float32, device = device) * 0.01
            for old_ns in group_ns:
                v = next(iter(old_ns.scope))

                weighted_counts(
                    input_params, 
                    torch.from_numpy(subset_full_cids[old_ns.scope]).to(device), 
                    obs_data[:,v].contiguous().to(device), 
                    weights
                )

            input_params = input_params.cpu()

            # import pdb; pdb.set_trace()

            input_params += 1e-6
            input_params /= input_params.sum(dim = 1, keepdim = True)

            flag = not torch.any(torch.isnan(input_params)) and torch.all(input_params > 1e-12) and torch.all(input_params < 1.0)
            if not flag:
                import pdb; pdb.set_trace()
                # raise ValueError()

            # Compute outputs of all nodes

            new_group_ns = []
            source_ns = None
            for old_ns in group_ns:
                v = next(iter(old_ns.scope))
                lls = input_params.permute(1, 0)[obs_data[:,v],:].permute(1, 0).log()

                print(f"> [Input var {v}] Aveg best per dim LL: {lls.max(dim = 0).values.mean():.2f}")

                temp_fname = os.path.join(tmp_folder, str(uuid.uuid4()) + ".npy")
                np.save(temp_fname, lls.numpy())
                scope2lls_fname[old_ns.scope] = temp_fname

                new_ns = old_ns.duplicate(v, tie_params = False)

                if not old_ns.is_tied():
                    new_ns.set_params(input_params)
                    source_ns = new_ns

                new_group_ns.append(new_ns)

            for new_ns in new_group_ns:
                if new_ns != source_ns:
                    new_ns.set_source_ns(source_ns)

            for old_ns, new_ns in zip(group_ns, new_group_ns):
                ns_old2new[old_ns] = new_ns

            print(">>> Processed an input node group")

        elif group_ns[0].is_sum():

            print("> Start processing a sum node group")

            is_root = (len(group_ns[0].scope) == total_num_vars)
            
            # Retrieve the product node connections
            all_prod_ids = []
            for prod_ns in group_ns[0].chs:
                par_group_id = scope2group_id[prod_ns.scope]
                chs_group_id = tuple(scope2group_id[cs.scope] for cs in prod_ns.chs)
                group_partition = (par_group_id, chs_group_id)

                prod_ids = torch.tensor(xprod_cids[group_partition])
                all_prod_ids.append(prod_ids)

            print("> Product node connections retrieved")

            # Compute product node LLs and pair them with the node cluster indices
            print("> Computing product node LLs...")

            nscid_chlls_pairs = []
            for old_ns in tqdm(group_ns):
                ns_chs_lls = []
                for idx, old_prod_ns in enumerate(old_ns.chs):
                    chs_lls = []
                    for cs in old_prod_ns.chs:
                        ch_lls = get_lls_for_scope(cs.scope)
                        chs_lls.append(ch_lls)

                    prod_ids = all_prod_ids[idx]
                    prod_lls = get_prod_lls(chs_lls, prod_ids, device = torch.device("cuda:0"))

                    ns_chs_lls.append(prod_lls)

                ns_chs_lls = torch.cat(ns_chs_lls, dim = 0)
                nscid_chlls_pairs.append((torch.from_numpy(subset_full_cids[old_ns.scope]), ns_chs_lls))

            # tasks = []
            # for old_ns in group_ns:
            #     prod_chs_fnames = []
            #     for old_prod_ns in old_ns.chs:
            #         prod_chs_fnames.append([scope2lls_fname[cs.scope] for cs in old_prod_ns.chs])
                
            #     tasks.append(("compute_ns_chs_lls", (old_ns.scope, prod_chs_fnames, all_prod_ids)))

            # results = manager.run_tasks(tasks)
            # nscid_chlls_pairs = []
            # for scope, ns_chs_lls in results:
            #     nscid_chlls_pairs.append((torch.from_numpy(subset_full_cids[scope]), ns_chs_lls))

            # Compute and assign sum edges via layer-wise LVD
            print("> Computing and assigning sum edges...")
            if not is_root:
                num_prod_ns = sum([prod_ids.size(0) for prod_ids in all_prod_ids])
                sum_edge_weights = torch.zeros([num_latents, num_prod_ns], dtype = torch.float32).to(device)
                for cls_ids, chs_lls in tqdm(nscid_chlls_pairs):
                    maxobj = chs_lls.max(dim = 0)
                    max_lls = maxobj.values
                    max_chids = maxobj.indices

                    max_lls -= max_lls.max()
                    max_lls /= 2.0
                    max_ps = max_lls.exp()

                    weighted_counts(sum_edge_weights, cls_ids.to(device), max_chids.to(device), max_ps.to(device))

                num_edges_per_node = int(math.ceil(num_prod_ns * sparsity))
                topk_obj = torch.topk(sum_edge_weights.cpu(), k = num_edges_per_node, dim = 1)
                probs = topk_obj.values # [num_latents, num_edges_per_node]
                ch_ids = topk_obj.indices # [num_latents, num_edges_per_node]
                ns_ids = torch.arange(0, num_latents).unsqueeze(1).repeat(1, num_edges_per_node)

                sum_edges = torch.stack((
                    ns_ids.reshape(-1), 
                    ch_ids.reshape(-1)
                ), dim = 0)

                probs += 0.1
                probs /= probs.sum(dim = 1, keepdim = True)
                sum_params = probs.reshape(-1)

            else:
                num_prod_ns = sum([prod_ids.size(0) for prod_ids in all_prod_ids])
                sum_edge_weights = torch.zeros([1, num_prod_ns], dtype = torch.float32).to(device)
                for cls_ids, chs_lls in tqdm(nscid_chlls_pairs):
                    maxobj = chs_lls.max(dim = 0)
                    max_lls = maxobj.values
                    max_chids = maxobj.indices

                    max_lls -= max_lls.max()
                    max_lls /= 2.0
                    max_ps = max_lls.exp()

                    cls_ids = torch.zeros([max_chids.size(0)], dtype = torch.long, device = device)

                    weighted_counts(sum_edge_weights, cls_ids, max_chids.to(device), max_ps.to(device))

                sum_edge_weights += 1.0
                probs = sum_edge_weights / sum_edge_weights.sum(dim = 1, keepdim = True)
                probs = probs.cpu()
                sum_params = probs.reshape(-1)

                sum_edges = torch.stack((
                    torch.zeros([num_prod_ns], dtype = torch.long),
                    torch.arange(0, num_prod_ns)
                ), dim = 0)

            # if not (not torch.any(torch.isnan(sum_params)) and torch.all(sum_params > 1e-12) and torch.all(sum_params <= 1.0)):
            #     import pdb; pdb.set_trace()

            # Compute LLs of the sum nodes
            print("> Computing LLs of sum nodes")
            # tasks = []
            # for old_ns, nscid_chlls_pair in tqdm(zip(group_ns, nscid_chlls_pairs)):
            #     ns_chs_lls = nscid_chlls_pair[1] # [num_prod_ns, num_samples]

            #     sum_lls = get_sum_lls(ns_chs_lls, ch_ids, probs, num_latents, device = torch.device("cuda:0"))

            #     temp_fname = os.path.join(tmp_folder, str(uuid.uuid4()) + ".npy")
            #     np.save(temp_fname, sum_lls.numpy())
            #     scope2lls_fname[old_ns.scope] = temp_fname

            if not is_root:
                tasks = []
                for old_ns, nscid_chlls_pair in zip(group_ns, nscid_chlls_pairs):
                    ns_chs_lls = nscid_chlls_pair[1] # [num_prod_ns, num_samples]

                    tasks.append(("get_sum_lls", (old_ns.scope, ns_chs_lls, ch_ids, probs, num_latents, tmp_folder)))

                results = manager.run_tasks(tasks)
                for scope, temp_fname in results:
                    scope2lls_fname[scope] = temp_fname

            # Create new nodes
            old_source_ns = None
            new_source_ns = None
            for old_ns in group_ns:
                if not old_ns.is_tied():
                    old_source_ns = old_ns

                    prod_nodes = []
                    for idx, old_pns in enumerate(old_ns.chs):
                        new_chs = [ns_old2new[cs] for cs in old_pns.chs]
                        new_pns = juice.multiply(*new_chs, edge_ids = all_prod_ids[idx])
                        prod_nodes.append(new_pns)
                    
                    new_ns = juice.summate(*prod_nodes, edge_ids = sum_edges, params = sum_params)

                    ns_old2new[old_ns] = new_ns
                    new_source_ns = new_ns

                    break

            for old_ns in group_ns:
                if old_ns.is_tied():
                    prod_nodes = []
                    for idx, old_pns in enumerate(old_ns.chs):
                        new_chs = [ns_old2new[cs] for cs in old_pns.chs]
                        new_pns = new_source_ns.chs[idx].duplicate(*new_chs)
                        prod_nodes.append(new_pns)

                    new_ns = new_source_ns.duplicate(*prod_nodes, tie_params = True)

                    ns_old2new[old_ns] = new_ns

            print(">>> Processed a sum node group")

        else:
            raise ValueError("Expecting ns in `group_ns` to be either input or sum.")

    manager.kill()

    return ns_old2new[root_ns]