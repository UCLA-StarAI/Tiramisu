import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import omegaconf
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import Sampler
from typing import Sequence
import wandb
import shutil
import os
from tqdm import tqdm

import sys
sys.path.append("./")
sys.path.append("../../../")
sys.path.append("../../../external/taming-transformers/")
sys.path.append("../../../external/latent-diffusion/")
sys.path.append("../../pixel_space_lvd/")

from tools.args import from_argparse_args
from tools.utils import instantiate_from_config
from tools.data import get_input

import warnings
warnings.filterwarnings("ignore")


class SubsetSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices, generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        for i in torch.arange(0, len(self.indices)):
            yield self.indices[i]

    def __len__(self):
        return len(self.indices)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type = str, default = "models/2021-04-23T18-11-19_celebahq_transformer/")
    parser.add_argument("--batch-size", type = int, default = 0)
    parser.add_argument("--num-workers", type = int, default = 0)
    parser.add_argument("--sid", type = int, default = -1)
    parser.add_argument("--eid", type = int, default = -1)
    parser.add_argument("--val-only", default = False, action = "store_true")
    parser.add_argument("--merge", default = False, action = "store_true")

    return parser


def partial_run(model, opt, dsets, output_folder):
    train_loader = dsets._train_dataloader(sampler = SubsetSampler(torch.arange(opt.sid, opt.eid)))

    with torch.no_grad():
        tr_c_indices = None
        tr_z_indices = None
        sid = 0
        for idx, batch in enumerate(tqdm(train_loader)):
            x = get_input(batch, "image").cuda()
            try:
                c = get_input(batch, model.cond_stage_key).cuda()
            except AttributeError:
                c = None

            eid = sid + x.size(0)

            try:
                quant_z, z_indices = model.first_stage_model(x, call_func = "safe_encode")
            except Exception:
                quant_z, z_indices = model(x, call_func = "safe_encode")

            if not hasattr(model, "cond_stage_model") or model.cond_stage_model is None:
                c_indices = torch.zeros([z_indices.size(0), 1], dtype = torch.long, device = z_indices.device)
            else:
                quant_c, _, (_, _, c_indices) = model.cond_stage_model(c, call_func = "encode")

            discrete_indices = torch.cat((c_indices.clone(), z_indices.clone()), dim = 1)

            if tr_c_indices is None:
                tr_c_indices = torch.zeros([opt.eid - opt.sid, c_indices.size(1)], dtype = torch.long)
                tr_z_indices = torch.zeros([opt.eid - opt.sid, z_indices.size(1)], dtype = torch.long)
            my_eid = min(eid, opt.eid - opt.sid)
            n_s = my_eid - sid
            tr_c_indices[sid:my_eid,:] = c_indices[:n_s,:].detach().cpu()
            tr_z_indices[sid:my_eid,:] = z_indices[:n_s,:].detach().cpu()

            sid = eid

    np.savez(
        os.path.join(output_folder, f"latents_{opt.sid}_{opt.eid}.npz"), 
        tr_c_latents = tr_c_indices.detach().cpu().numpy(),
        tr_z_latents = tr_z_indices.detach().cpu().numpy()
    )


def main():
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    config = None
    config_base = os.path.join(opt.config, "configs/")
    for config_fname in os.listdir(config_base):
        curr_config = OmegaConf.load(os.path.join(config_base, config_fname))
        if config is None:
            config = curr_config
        else:
            config = OmegaConf.merge(config, curr_config)

    # Load dataset
    if opt.batch_size > 0:
        config.data.params.batch_size = opt.batch_size
    if opt.num_workers > 0:
        config.data.params.num_workers = opt.num_workers
    dsets = instantiate_from_config(config.data)
    dsets.shuffle_train = False
    dsets.prepare_data()
    dsets.setup()

    # Load model
    model = instantiate_from_config(config.model)

    if os.path.exists(os.path.join(opt.config, "checkpoints/last.ckpt")):
        ckpt_fname = os.path.join(opt.config, "checkpoints/last.ckpt")
        sd = torch.load(ckpt_fname)

        model.load_state_dict(sd["state_dict"])
    else:
        raise ValueError()

    # To GPU
    device = torch.device("cuda:0")

    model.to(device)

    # Target folder
    base_folder = "outputs/latents/"
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)
    model_name = (opt.config.split("/")[-2]).split(".")[0]
    output_folder = os.path.join(base_folder, model_name)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if opt.merge:
        d = dict()
        l = list()
        for fname in os.listdir(output_folder):
            if os.path.isfile(os.path.join(output_folder, fname)):
                if fname == "latents.npz" or fname == "summary.txt":
                    continue
                data = np.load(os.path.join(output_folder, fname), allow_pickle = True)
                
                if "val" in fname:
                    vl_z_latents = data["vl_z_latents"]
                    vl_c_latents = data["vl_c_latents"]
                else:
                    sid = int(fname.split("_")[1])
                    d[sid] = data
                    l.append(sid)

        tr_z_latents = []
        tr_c_latents = []

        l = sorted(l)
        for sid in l:
            tr_z_latents.append(d[sid]["tr_z_latents"])
            tr_c_latents.append(d[sid]["tr_c_latents"])

        tr_z_latents = np.concatenate(tr_z_latents, axis = 0)
        tr_c_latents = np.concatenate(tr_c_latents, axis = 0)

        with open(os.path.join(output_folder, "summary.txt"), "w") as f:
            f.write(f"Number of training samples: {tr_c_latents.shape[0]}\n")
            f.write(f"Number of validation samples: {vl_c_latents.shape[0]}\n")
            f.write(f"Number of c variables: {tr_c_latents.shape[1]}\n")
            f.write(f"Number of z variables: {tr_z_latents.shape[1]}\n")

        print(np.unique(tr_c_latents))

        np.savez(
            os.path.join(output_folder, "latents.npz"), 
            tr_c_latents = tr_c_latents,
            tr_z_latents = tr_z_latents,
            vl_c_latents = vl_c_latents,
            vl_z_latents = vl_z_latents
        )

        return None

    if not opt.val_only:
        if opt.sid >= 0 and opt.eid >= 0:
            partial_run(model, opt, dsets, output_folder)
            return

    # Run
    with torch.no_grad():
        if not opt.val_only:
            train_loader = dsets._train_dataloader()
            tr_c_indices = None
            tr_z_indices = None
            sid = 0
            for idx, batch in enumerate(tqdm(train_loader)):
                x = get_input(batch, "image").cuda()
                try:
                    c = get_input(batch, model.cond_stage_key).cuda()
                except AttributeError:
                    c = None

                eid = sid + x.size(0)

                try:
                    quant_z, z_indices = model.first_stage_model(x, call_func = "safe_encode")
                except Exception:
                    quant_z, z_indices = model(x, call_func = "safe_encode")

                if not hasattr(model, "cond_stage_model") or model.cond_stage_model is None:
                    c_indices = torch.zeros([z_indices.size(0), 1], dtype = torch.long, device = z_indices.device)
                else:
                    quant_c, _, (_, _, c_indices) = model.cond_stage_model(c, call_func = "encode")

                discrete_indices = torch.cat((c_indices.clone(), z_indices.clone()), dim = 1)

                # Verification
                if idx == 0:
                    try:
                        xrec = model.first_stage_model(quant_z, call_func = "decode")
                    except AttributeError:
                        xrec = model(quant_z, call_func = "decode")

                    import matplotlib.pyplot as plt
                    plt.figure()

                    rescale = lambda x: (x + 1.) / 2.
                    img = np.transpose(rescale(xrec[3,:,:,:].detach().cpu().numpy()), (1, 2, 0))
                    plt.imshow(img)
                    plt.savefig("debug.png")

                if tr_c_indices is None:
                    tr_c_indices = torch.zeros([len(train_loader.dataset), c_indices.size(1)], dtype = torch.long)
                    tr_z_indices = torch.zeros([len(train_loader.dataset), z_indices.size(1)], dtype = torch.long)
                tr_c_indices[sid:eid,:] = c_indices.detach().cpu()
                tr_z_indices[sid:eid,:] = z_indices.detach().cpu()

                sid = eid

        val_loader = dsets._val_dataloader()
        vl_c_indices = None
        vl_z_indices = None
        sid = 0
        for batch in tqdm(val_loader):
            x = get_input(batch, "image").cuda()
            try:
                c = get_input(batch, model.cond_stage_key).cuda()
            except AttributeError:
                c = None

            eid = sid + x.size(0)

            try:
                quant_z, z_indices = model.first_stage_model(x, call_func = "safe_encode")
            except Exception:
                quant_z, z_indices = model(x, call_func = "safe_encode")
            z_indices = z_indices.view(quant_z.shape[0], -1)

            if not hasattr(model, "cond_stage_model") or model.cond_stage_model is None:
                c_indices = torch.zeros([z_indices.size(0), 1], dtype = torch.long, device = z_indices.device)
            else:
                quant_c, _, (_, _, c_indices) = model.cond_stage_model(c, call_func = "encode")

            discrete_indices = torch.cat((c_indices.clone(), z_indices.clone()), dim = 1)

            if vl_c_indices is None:
                vl_c_indices = torch.zeros([len(val_loader.dataset), c_indices.size(1)], dtype = torch.long)
                vl_z_indices = torch.zeros([len(val_loader.dataset), z_indices.size(1)], dtype = torch.long)
            vl_c_indices[sid:eid,:] = c_indices.detach().cpu()
            vl_z_indices[sid:eid,:] = z_indices.detach().cpu()

            sid = eid

    if not opt.val_only:

        with open(os.path.join(output_folder, "summary.txt"), "w") as f:
            f.write(f"Number of training samples: {tr_c_indices.size(0)}\n")
            f.write(f"Number of validation samples: {vl_c_indices.size(0)}\n")
            f.write(f"Number of c variables: {c_indices.size(1)}\n")
            f.write(f"Number of z variables: {z_indices.size(1)}\n")

        np.savez(
            os.path.join(output_folder, "latents.npz"), 
            tr_c_latents = tr_c_indices.detach().cpu().numpy(),
            tr_z_latents = tr_z_indices.detach().cpu().numpy(),
            vl_c_latents = vl_c_indices.detach().cpu().numpy(),
            vl_z_latents = vl_z_indices.detach().cpu().numpy()
        )

    else:

        np.savez(
            os.path.join(output_folder, "latents_val.npz"), 
            vl_c_latents = vl_c_indices.detach().cpu().numpy(),
            vl_z_latents = vl_z_indices.detach().cpu().numpy()
        )

    print("done")


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    main()

