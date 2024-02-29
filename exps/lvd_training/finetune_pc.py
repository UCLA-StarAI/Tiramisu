import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import omegaconf
from torch.utils.data import TensorDataset, DataLoader
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

import sys
sys.path.append("./")
sys.path.append("../../")
sys.path.append("../../external/taming-transformers/")

from distributional import MPManager

from controlled_img_modeling.lvd import pc_lvd_parser
from controlled_img_modeling.lvd import apply_lvd
from controlled_img_modeling.utils import ProgressBar

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
    parser.add_argument("--batch-size", type = int, default = 64)
    parser.add_argument("--no-save", default = False, action = "store_true")

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


def main():
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # Load datasets
    dsets = load_dataset(opt)
    train_data_z = dsets.datasets["train"].data_dict["z"]
    train_data_c = dsets.datasets["train"].data_dict["c"]

    num_tr_samples = train_data_z.size(0)

    # PC ckpt path
    full_model_name = os.path.basename(opt.latent_stage_config).split(".")[0] + "-" + \
        os.path.basename(opt.config).split(".")[0]
    lvd_name = os.path.basename(opt.lvd_config).split(".")[0]
    base_ckpt_path = f"outputs/lvd/{full_model_name}/{lvd_name}/"
    pc_fname = os.path.join(base_ckpt_path, f"lvd_initialized_pc.jpc")

    ## Finetune PC ##

    device = torch.device("cuda:0")

    ns = juice.io.load(pc_fname)

    print("> Compiling PC...")
    pc = juice.TensorCircuit(ns)
    print("> Done")

    print(f"> Number of nodes: {pc.num_nodes}")
    print(f"> Number of edges: {pc.num_edges}")
    print(f"> Number of sum parameters: {pc.num_sum_params}")

    pc.to(device)

    # Define dataset
    dataset = TensorDataset(train_data_z)
    data_loader = DataLoader(
        dataset = dataset,
        batch_size = opt.batch_size,
        shuffle = True,
        drop_last = True
    )

    finetuned_pc_fname = os.path.join(base_ckpt_path, f"lvd_finetuned_pc.jpc")

    mode = "full_batch"
    if mode == "mini_batch":
        # Define optimizer
        optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.01, pseudocount = 0.001)
        scheduler = juice.optim.CircuitScheduler(
            optimizer, 
            method = "multi_linear", 
            lrs = [0.01, 0.001, 0.0004, 0.0001], 
            milestone_steps = [0, len(data_loader) * 50, len(data_loader) * 40, len(data_loader) * 110]
        )

        print("> Start finetuning...")
        num_epochs = 200
        progress_bar = ProgressBar(num_epochs, len(data_loader), ["LL"], cumulate_statistics = True)
        aveg_lls = []
        for epoch in range(num_epochs):
            progress_bar.new_epoch_begin()
            total_ll = 0.0
            for batch in data_loader:
                x = batch[0].to(device)

                optimizer.zero_grad()

                lls = pc(x)
                lls.mean().backward()

                total_ll += lls.mean().detach().cpu().numpy().item()

                progress_bar.new_batch_done([lls.mean().detach().cpu().numpy().item()])

                optimizer.step()
                scheduler.step()

            progress_bar.epoch_ends()
            aveg_ll = total_ll / len(data_loader)

            aveg_lls.append(aveg_ll)

    elif mode == "full_batch":

        print("> Start finetuning...")
        num_epochs = 200
        progress_bar = ProgressBar(num_epochs, len(data_loader), ["LL"], cumulate_statistics = True)
        aveg_lls = []
        for epoch in range(num_epochs):
            progress_bar.new_epoch_begin()
            total_ll = 0.0
            for batch in data_loader:
                x = batch[0].to(device)

                lls = pc(x)
                pc.backward(x, flows_memory = 1.0)

                total_ll += lls.mean().detach().cpu().numpy().item()

                progress_bar.new_batch_done([lls.mean().detach().cpu().numpy().item()])

            pc.mini_batch_em(step_size = 1.0, pseudocount = 0.1)

            progress_bar.epoch_ends()
            aveg_ll = total_ll / len(data_loader)

            aveg_lls.append(aveg_ll)

            if not opt.no_save:
                print("> Saving PC...")
                juice.io.save(finetuned_pc_fname, pc)
                print("> Done.")

    else:
        raise ValueError()


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    main()

