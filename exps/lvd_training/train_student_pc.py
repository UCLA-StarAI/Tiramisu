import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import omegaconf
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import wandb
import shutil
import os
from tqdm import tqdm
import yaml
from copy import deepcopy
from pytorch_lightning.strategies import FSDPStrategy
from functools import partial
import tempfile
import math
import subprocess
import pickle
import pyjuice as juice
import ray
import multiprocessing as mp

import sys
sys.path.append("./")
sys.path.append("../../")
sys.path.append("../../external/taming-transformers/")

from distributional import MPManager

from controlled_img_modeling.lvd import pc_lvd_parser
from controlled_img_modeling.lvd import apply_lvd, apply_lvd_with_pretraining

from tools.args import from_argparse_args
from tools.utils import instantiate_from_config, get_obj_from_str
from tools.data import get_input

import warnings
warnings.filterwarnings("ignore")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type = str, default = "models/2021-04-23T18-11-19_celebahq_transformer/")
    parser.add_argument("--latent-stage-config", type = str, default = "configs/latent_stage_models/mlm_medium_256_1024.yaml")
    parser.add_argument("--lvd-config", type = str, default = "configs/lvd_structures/img_pd_1024_no_tie.yaml")
    parser.add_argument("--gpus", type = str, default = "AVAILABLE")
    parser.add_argument("--num-gpus-per-worker", type = int, default = 1)
    parser.add_argument("--batch-size", type = int, default = 64)

    # For Kmeans
    parser.add_argument("--kmeans-niter", type = int, default = 200)
    parser.add_argument("--kmeans-nredo", type = int, default = 1)
    parser.add_argument("--reduced-dim", type = int, default = 32)
    parser.add_argument("--max-kmeans-feature-dim", type = int, default = 1024)
    parser.add_argument("--max-kmeans-num-samples", type = int, default = 400000)

    parser.add_argument("--max-num-scopes", type = int, default = 16)

    # For the product computation stage
    parser.add_argument("--num-samples-per-partition", type = int, default = 160000)
    parser.add_argument("--n-cls-per-target", type = int, default = 16)
    parser.add_argument("--num-configs-per-target-cls", type = int, default = 20)

    # For layer-wise pretraining
    parser.add_argument("--pretrain-num-samples", type = int, default = 20000)

    # Logging
    parser.add_argument("--gpt-log-every", type = int, default = 5000)

    return parser


def load_dataset(opt):

    config = None
    config_base = os.path.join(opt.config, "configs/")
    for config_fname in os.listdir(config_base):
        curr_config = OmegaConf.load(os.path.join(config_base, config_fname))
        if config is None:
            config = curr_config
        else:
            config = OmegaConf.merge(config, curr_config)

    base_folder = "outputs/latents/"
    model_name = os.path.basename(opt.config).split(".")[0]
    data_folder = os.path.join(base_folder, model_name)
    assert os.path.exists(data_folder)

    config.data.params.train.target = "tools.data.ZCNumpyDataset"
    config.data.params.train.params = {
        "data_path": data_folder,
        "split": "train",
        "key_mapping": {"c_latents": "c", "z_latents": "z"}
    }
    config.data.params.validation.target = "tools.data.ZCNumpyDataset"
    config.data.params.validation.params = {
        "data_path": data_folder,
        "split": "validation",
        "key_mapping": {"c_latents": "c", "z_latents": "z"}
    }

    if opt.batch_size > 0:
        config.data.params.batch_size = opt.batch_size

    config.data.params.shuffle_train = False

    dsets = instantiate_from_config(config.data)
    dsets.prepare_data()
    dsets.setup()

    return dsets


def get_available_gpus():
    cmd = 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    output = result.stdout.strip().split('\n')
    gpu_utilization = [int(util.split()[0]) for util in output]

    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != "":
        visible_gpus = [int(gpu) for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    else:
        visible_gpus = [gpu for gpu in range(len(gpu_utilization))]
    
    gpus = []
    for i, util in enumerate(gpu_utilization):
        print(f"GPU {i}: {util}%")
        if util < 10 and i in visible_gpus:
            gpus.append(i)

    print("Selected GPUs:", gpus)

    return gpus


def model_constructor(opt):
    latent_stage_config = OmegaConf.load(opt.latent_stage_config)
    model = instantiate_from_config(latent_stage_config.model)

    sd = torch.load(opt.model_ckpt_path, map_location = "cpu")
    model.load_state_dict(sd["state_dict"])

    return model


def mkdir(path):
    num_levels = len(path.split("/"))
    for i in range(1, num_levels + 1):
        curr_path = "/".join(path.split("/")[:i])
        if not os.path.exists(curr_path):
            os.mkdir(curr_path)


def main():
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # Load datasets
    dsets = load_dataset(opt)
    train_data_z = dsets.datasets["train"].data_dict["z"]
    train_data_c = dsets.datasets["train"].data_dict["c"]

    num_tr_samples = train_data_z.size(0)

    # Get ckpt path
    full_model_name = os.path.basename(opt.latent_stage_config).split(".")[0] + "-" + \
        os.path.basename(opt.config).split(".")[0]
    opt.model_ckpt_path = os.path.join(
        "outputs/latent_model_ckpts/",
        full_model_name,
        "ckpts/last.ckpt"
    )
    assert os.path.exists(opt.model_ckpt_path)

    # Parse device indices
    if opt.gpus == "ALL":
        gpus = [i for i in range(torch.cuda.device_count())]
    elif opt.gpus == "AVAILABLE":
        gpus = get_available_gpus()
    else:
        gpus = [int(i) for i in opt.gpus.split(",")]
    
    device_ids = []
    for i in range(0, len(gpus), opt.num_gpus_per_worker):
        if len(gpus) - i < opt.num_gpus_per_worker:
            continue

        device_ids.append(gpus[i:i+opt.num_gpus_per_worker])

    # MP manager
    manager = MPManager(device_ids, model_constructor, worker_args = opt)

    # Base ckpt path
    lvd_name = os.path.basename(opt.lvd_config).split(".")[0]
    base_ckpt_path = f"outputs/lvd/{full_model_name}/{lvd_name}/"
    mkdir(base_ckpt_path)

    ## Get base PC structure ##

    lvd_config = OmegaConf.load(opt.lvd_config)["lvd"]
    base_pc_struct = partial(get_obj_from_str(lvd_config["target"]), **lvd_config.get("params", dict()))
    
    latent_config = OmegaConf.load(opt.latent_stage_config)["model"]
    vocab_size = latent_config.params.vocab_size
    block_size = latent_config.params.block_size
    height = width = int(math.sqrt(block_size))

    num_latents = lvd_config.num_latents

    root_ns = base_pc_struct(height = height, width = width, num_latents = num_latents, num_cats = vocab_size)

    # Get target 
    scope2group_id, scope_groups, sgroup_partition_scopes = pc_lvd_parser(root_ns)

    ## Collect LV assignment ##

    cls_info_fname = os.path.join(base_ckpt_path, "clusters_info.pkl")
    if os.path.exists(cls_info_fname):
        with open(cls_info_fname, 'rb') as f:
            scopes, lv_centroids, lv_pca_models = pickle.load(f)

    else:
        tasks = []
        for scopes in scope_groups:
            if len(scopes) > opt.max_num_scopes:
                ids = np.random.permutation(len(scopes))[:opt.max_num_scopes]
                scopes = [scopes[i] for i in ids]
            if num_tr_samples * len(scopes) > opt.max_kmeans_num_samples:
                eid = int(opt.max_kmeans_num_samples // len(scopes))
            else:
                eid = num_tr_samples
            tasks.append(("get_subset_cids", (train_data_z[:eid,:], train_data_c[:eid,:], scopes, num_latents)))
        rets = manager.run_tasks(tasks)

        scope_groups = [item[0] for item in rets]
        lv_centroids = [item[2] for item in rets]
        lv_pca_models = [item[3] for item in rets]

        with open(cls_info_fname, "wb") as f:
            pickle.dump((scope_groups, lv_centroids, lv_pca_models), f)

    ## Product layer connection ##

    # Prepare for lvd
    sgroup_partition_cids_mapping = dict()
    scope2task_nsamples = dict()
    for key, scope_partitions in sgroup_partition_scopes.items():
        if len(scope_partitions) > opt.max_num_scopes:
            ids = np.random.permutation(len(scope_partitions))[:opt.max_num_scopes]
            scope_partitions = [scope_partitions[i] for i in ids]

        num_samples_per_scope = min(opt.num_samples_per_partition // len(scope_partitions), num_tr_samples)

        sgroup_partition_cids_mapping[key] = []
        for par_scope, ch_scopes in scope_partitions:

            curr_cids = []
            for scope in [par_scope] + list(ch_scopes):
                group_id = scope2group_id[scope]

                curr_cids.append(("task_id", scope))
                if scope in scope2task_nsamples:
                    scope2task_nsamples[scope] = max(scope2task_nsamples[scope], num_samples_per_scope)
                else:
                    scope2task_nsamples[scope] = num_samples_per_scope

            sgroup_partition_cids_mapping[key].append(curr_cids)

    tasks = []
    total_nsamples = 0
    for scope, nsamples in scope2task_nsamples.items():
        group_id = scope2group_id[scope]
        tasks.append(("pred_cids", (
            train_data_z[:nsamples,:], 
            train_data_c[:nsamples,:], 
            scope, 
            lv_centroids[group_id], 
            lv_pca_models[group_id]
        )))
        total_nsamples += nsamples

    print(f"> Total number of samples: {total_nsamples}")

    lv_cids_fname = os.path.join(base_ckpt_path, "lv_cids.pkl")
    if os.path.exists(lv_cids_fname):
        with open(lv_cids_fname, "rb") as f:
            aux_cids = pickle.load(f)

    else:
        aux_cids = manager.run_tasks(tasks)
        aux_cids = {scope: cids for scope, cids in aux_cids}

        with open(lv_cids_fname, "wb") as f:
            pickle.dump(aux_cids, f)

    # Collect cids and compute product connections
    sgroup_partition_cids = dict()
    for key, scope_partitions in sgroup_partition_scopes.items():
        group_cids = None
        for par_scope, ch_scopes in scope_partitions:
            try:
                par_cids = aux_cids[par_scope]
                ch_cids = [aux_cids[ch_scope] for ch_scope in ch_scopes]
            except KeyError:
                continue

            min_nsamples = min(par_cids.shape[0], min([ch_cid.shape[0] for ch_cid in ch_cids]))

            curr_cids = torch.from_numpy(np.stack(
                [par_cids[:min_nsamples]] + [ch_cid[:min_nsamples] for ch_cid in ch_cids],
                axis = 1
            ))

            if group_cids is None:
                group_cids = curr_cids
            else:
                group_cids = torch.cat((group_cids, curr_cids), dim = 0)

        assert group_cids is not None

        sgroup_partition_cids[key] = group_cids

    # Finally, compute product layer connections
    tasks = []
    sgroup_cids = dict()
    sgroups = []
    for group_partition, sg_cids in sgroup_partition_cids.items():
        target_group_ids = group_partition[0]
        chs_group_ids = group_partition[1]
        target_centroids = lv_centroids[target_group_ids]
        chs_centroids = [lv_centroids[ch_gid] for ch_gid in chs_group_ids]

        target_cids = sg_cids[:,0]
        chs_cids = sg_cids[:,1:]

        tasks.append(("get_xprod_assignments", (
            target_cids, chs_cids, target_centroids, chs_centroids, opt.reduced_dim
        )))
        sgroup_cids[group_partition] = (target_cids, chs_cids)
        sgroups.append(group_partition)

    group_xprod_fname = os.path.join(base_ckpt_path, "xprod.pkl")
    if os.path.exists(group_xprod_fname):
        with open(group_xprod_fname, "rb") as f:
            xprod_cids = pickle.load(f)

    else:
        print("> Running xprod tasks")
        xprod_cids = manager.run_tasks(tasks)
        xprod_cids = {group_partition: xprod for group_partition, xprod in zip(sgroups, xprod_cids)}

        with open(group_xprod_fname, "wb") as f:
            pickle.dump(xprod_cids, f)

    # Input node LVD

    group_id2ids = dict()
    for scope, cids in aux_cids.items():
        if len(scope) == 1:
            v = next(iter(scope))
            cids = torch.from_numpy(cids)
            nsamples = cids.size(0)
            nids = train_data_z[:nsamples,v]

            group_id = scope2group_id[scope]
            if group_id in group_id2ids:
                group_id2ids[group_id] = (
                    torch.cat((group_id2ids[group_id][0], cids), dim = 0),
                    torch.cat((group_id2ids[group_id][1], nids), dim = 0)
                )
            else:
                group_id2ids[group_id] = (cids, nids)

    # We choose a subset of samples and do LVD for every scope
    opt.pretrain_num_samples = min(opt.pretrain_num_samples, train_data_z.size(0))
    tasks = []
    for scope, group_id in scope2group_id.items():
        tasks.append(("pred_cids", (
            train_data_z[:opt.pretrain_num_samples,:], 
            train_data_c[:opt.pretrain_num_samples,:], 
            scope, 
            lv_centroids[group_id], 
            lv_pca_models[group_id]
        )))

    subset_lv_cids_fname = os.path.join(base_ckpt_path, "subset_lv_cids.pkl")
    if os.path.exists(subset_lv_cids_fname):
        with open(subset_lv_cids_fname, "rb") as f:
            subset_full_cids = pickle.load(f)

    else:
        subset_full_cids = manager.run_tasks(tasks)
        subset_full_cids = {scope: cids for scope, cids in subset_full_cids}

        with open(subset_lv_cids_fname, "wb") as f:
            pickle.dump(subset_full_cids, f)

    manager.kill()
    del manager

    ## Apply LVD to the PC ##

    # root_ns = apply_lvd(
    #     root_ns, scope2group_id, scope_groups, sgroup_partition_scopes, 
    #     group_id2ids, sgroup_cids, xprod_cids, lv_centroids, 
    #     num_latents = num_latents, num_cats = vocab_size
    # )

    obs_data = train_data_z[:opt.pretrain_num_samples,:]
    root_ns = apply_lvd_with_pretraining(
        root_ns, obs_data, scope2group_id, scope_groups,
        xprod_cids, subset_full_cids, num_latents, num_cats = vocab_size
    )

    # Save PC
    pc_fname = os.path.join(base_ckpt_path, f"lvd_initialized_pc.jpc")
    juice.io.save(pc_fname, root_ns)

    print("> PC completed and saved.")


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(os.cpu_count())
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    main()

