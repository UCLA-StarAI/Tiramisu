import argparse, os, sys, glob, math, time
import torch
import numpy as np
from omegaconf import OmegaConf
import streamlit as st
from PIL import Image
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("./")

from main import instantiate_from_config, DataModuleFromConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


rescale = lambda x: (x + 1.) / 2.


def bchw_to_st(x):
    return rescale(x.detach().cpu().numpy().transpose(0,2,3,1))

def save_img(xstart, fname):
    I = (xstart.clip(0,1)[0]*255).astype(np.uint8)
    Image.fromarray(I).save(fname)



def get_interactive_image(resize=False):
    image = st.file_uploader("Input", type=["jpg", "JPEG", "png"])
    if image is not None:
        image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        print("upload image shape: {}".format(image.shape))
        img = Image.fromarray(image)
        if resize:
            img = img.resize((256, 256))
        image = np.array(img)
        return image


def single_image_to_torch(x, permute=True):
    assert x is not None, "Please provide an image through the upload function"
    x = np.array(x)
    x = torch.FloatTensor(x/255.*2. - 1.)[None,...]
    if permute:
        x = x.permute(0, 3, 1, 2)
    return x


def pad_to_M(x, M):
    hp = math.ceil(x.shape[2]/M)*M-x.shape[2]
    wp = math.ceil(x.shape[3]/M)*M-x.shape[3]
    x = torch.nn.functional.pad(x, (0,wp,0,hp,0,0,0,0))
    return x

# ((16,48), (16,48))

@torch.no_grad()
def run_conditional(model, dsets, rfolder_name):
    base_folder = "latents/"

    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    base_folder = os.path.join(base_folder, rfolder_name)
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    with torch.no_grad():
        train_loader = dsets._train_dataloader()
        tr_indices = None
        sid = 0
        for batch in tqdm(train_loader):
            x = model.get_input("image", batch).to(model.device)

            cond_key = model.cond_stage_key
            c = model.get_input(cond_key, batch).to(model.device)

            eid = sid + x.size(0)

            quant_z, z_indices = model.encode_to_z(x)
            quant_c, c_indices = model.encode_to_c(c)

            discrete_indices = torch.cat((c_indices, z_indices), dim = 1)

            # xrec = model.first_stage_model.decode(quant_z)

            if tr_indices is None:
                tr_indices = torch.zeros([len(train_loader.dataset), discrete_indices.size(1)], dtype = torch.long)
            tr_indices[sid:eid,:] = discrete_indices.detach().cpu()

            sid = eid

        val_loader = dsets._val_dataloader()
        vl_indices = None
        sid = 0
        for batch in tqdm(val_loader):
            x = model.get_input("image", batch).to(model.device)

            cond_key = model.cond_stage_key
            c = model.get_input(cond_key, batch).to(model.device)

            eid = sid + x.size(0)

            quant_z, z_indices = model.encode_to_z(x)
            quant_c, c_indices = model.encode_to_c(c)

            discrete_indices = torch.cat((c_indices, z_indices), dim = 1)

            # xrec = model.first_stage_model.decode(quant_z)

            if vl_indices is None:
                vl_indices = torch.zeros([len(val_loader.dataset), discrete_indices.size(1)], dtype = torch.long)
            vl_indices[sid:eid,:] = discrete_indices.detach().cpu()

            sid = eid

    with open(os.path.join(base_folder, "summary.txt"), "w") as f:
        f.write(f"Number of training samples: {tr_indices.size(0)}\n")
        f.write(f"Number of validation samples: {vl_indices.size(0)}\n")
        f.write(f"Number of c variables: {c_indices.size(1)}\n")
        f.write(f"Number of c variables: {z_indices.size(1)}\n")

    np.savez(
        os.path.join(base_folder, "latents.npz"), 
        tr_latents = tr_indices.detach().cpu().numpy(),
        vl_latents = vl_indices.detach().cpu().numpy()
    )

    print("done")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    return parser


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        st.warning("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        st.warning("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            st.warning("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            st.warning("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        st.info(f"Missing Keys in State Dict: {missing}")
        st.info(f"Unexpected Keys in State Dict: {unexpected}")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model_and_dset(config, ckpt, gpu, eval_mode):
    # get data
    dsets = get_data(config)   # calls data.config ...

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dsets, model, global_step


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]
    elif opt.config == "":
        opt.base = []
        cfg_folder = os.path.join(opt.resume, "configs")
        for fname in os.listdir(cfg_folder):
            if fname.endswith(".yaml"):
                opt.base.append(
                    os.path.join(cfg_folder, fname)
                )

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    if opt.ignore_base_data:
        for config in configs:
            if hasattr(config, "data"): del config["data"]
    config = OmegaConf.merge(*configs, cli)

    rfolder_name = opt.resume.split("/")[-1]

    st.sidebar.text(ckpt)
    gs = st.sidebar.empty()
    gs.text(f"Global step: ?")
    st.sidebar.text("Options")
    #gpu = st.sidebar.checkbox("GPU", value=True)
    gpu = True
    #eval_mode = st.sidebar.checkbox("Eval Mode", value=True)
    eval_mode = True
    #show_config = st.sidebar.checkbox("Show Config", value=False)
    show_config = False
    if show_config:
        st.info("Checkpoint: {}".format(ckpt))
        st.json(OmegaConf.to_container(config))

    dsets, model, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode)
    gs.text(f"Global step: {global_step}")
    run_conditional(model, dsets, rfolder_name = rfolder_name)
