import argparse, os, sys, glob, math, time
import torch
import numpy as np
from omegaconf import OmegaConf
import streamlit as st
from PIL import Image
import sys
import matplotlib.pyplot as plt

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
def run_conditional(model, dsets, condition_type = "none", bbox = ((4, 12), (4, 12)),
                    img_base_folder = "./imgs/", rfolder_name = None):

    if not os.path.exists(img_base_folder):
        os.mkdir(img_base_folder)

    img_base_folder = os.path.join(img_base_folder, rfolder_name)
    if not os.path.exists(img_base_folder):
        os.mkdir(img_base_folder)

    if condition_type == "none":
        conditioning_method_name = "none"
    elif condition_type == "no_generation":
        conditioning_method_name = "no_generation"
    elif condition_type == "top_half":
        conditioning_method_name = "top_half"
    elif condition_type == "bbox":
        conditioning_method_name = "bbox_{}_{}_{}_{}".format(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
    else:
        raise NotImplementedError()

    img_folder = os.path.join(img_base_folder, conditioning_method_name)
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)

    sid = 0
    for batch in dsets._train_dataloader():
        x = model.get_input("image", batch).to(model.device)

        cond_key = model.cond_stage_key
        c = model.get_input(cond_key, batch).to(model.device)

        eid = sid + x.size(0)

        quant_z, z_indices = model.encode_to_z(x)
        quant_c, c_indices = model.encode_to_c(c)

        if z_indices.size(1) == 256:
            H, W = 16, 16
        idx_to_generate_c = torch.zeros([c_indices.size(1)], dtype = torch.bool)
        idx_to_generate_z = torch.ones([H, W], dtype = torch.bool)
        if condition_type == "none":
            pass
        elif condition_type == "no_generation":
            idx_to_generate_z[:,:] = False
        elif condition_type == "top_half":
            idx_to_generate_z[:H//2,:] = False
        elif condition_type == "bbox":
            idx_to_generate_z[bbox[0][0]:bbox[0][1],bbox[1][0]:bbox[1][1]] = False
        else:
            raise NotImplementedError()
        idx_to_generate = torch.cat((idx_to_generate_c, idx_to_generate_z.reshape(-1)), dim = 0)[1:]

        xrec = model.first_stage_model.decode(quant_z)

        if hasattr(model.transformer, "diffusion_based") and model.transformer.diffusion_based:
            out = model.transformer.sample_fast(
                cond_x_token = z_indices.unsqueeze(1).repeat(1, 6, 1).reshape((eid-sid) * 6, -1), 
                cond_mask = ~idx_to_generate_z.reshape(-1), 
                cond_token = c_indices.unsqueeze(1).repeat(1, 6, 1).reshape((eid-sid) * 6, -1), 
                temperature = 1.0, 
                skip_step = 0
            )
            all_z_samples = out["content_token"]
            cshape = quant_z.shape
            cshape = [all_z_samples.size(0), *cshape[1:]]
            all_x_samples = model.decode_to_img(all_z_samples, cshape)
            sizes = all_x_samples.shape
            all_x_samples = all_x_samples.reshape(eid - sid, 6, *sizes[1:]).detach().cpu()

        else:
            # This is slow
            # all_x_samples = []
            # for _ in range(6):
            #     z_samples = model.transformer.conditional_sample(
            #         idx = torch.cat((c_indices, z_indices), dim = 1), 
            #         idx_to_generate = idx_to_generate, 
            #         temperature = 1.0, 
            #         do_sample = True, 
            #         top_k = 10
            #     )[:,c_indices.size(1):]

            #     cshape = quant_z.shape
            #     x_samples = model.decode_to_img(z_samples, cshape)

            #     all_x_samples.append(x_samples.detach().cpu())
            # all_x_samples = torch.stack(all_x_samples, dim = 1)

            # This is faster
            all_z_samples = model.transformer.conditional_sample(
                idx = torch.cat((c_indices, z_indices), dim = 1).unsqueeze(1).repeat(1, 6, 1).reshape((eid-sid) * 6, -1), 
                idx_to_generate = idx_to_generate, 
                temperature = 1.0, 
                do_sample = True, 
                top_k = 10
            )[:,c_indices.size(1):]
            cshape = quant_z.shape
            cshape = [all_z_samples.size(0), *cshape[1:]]
            all_x_samples = model.decode_to_img(all_z_samples, cshape)
            sizes = all_x_samples.shape
            all_x_samples = all_x_samples.reshape(eid - sid, 6, *sizes[1:]).detach().cpu()

        titles = ["Origin", "Sample 1", "Sample 2", "Sample 3", "Reconstruction", "Sample 4", "Sample 5", "Sample 6"]
        for i in range(x.size(0)):
            curr_x = np.transpose(rescale(x[i,:,:,:].detach().cpu().numpy()), (1, 2, 0))
            curr_xrec = np.transpose(rescale(xrec[i,:,:,:].detach().cpu().numpy()), (1, 2, 0))
            curr_samples = np.transpose(rescale(all_x_samples[i,:,:,:,:].numpy()), (0, 2, 3, 1))

            curr_xs = np.concatenate(
                (curr_x[None,:,:,:], curr_samples[:3,:,:,:], curr_xrec[None,:,:,:], curr_samples[3:,:,:,:]),
                axis = 0
            )

            fig, axes = plt.subplots(2, 4)

            for j, ax in enumerate(axes.flatten()):
                ax.imshow(curr_xs[j])
                if j == 0:
                    mask = idx_to_generate_z[:,None,:,None].repeat(1, 16, 1, 16).reshape(256, 256)
                    ax.imshow(1.0 - mask.float().numpy() * 0.5, cmap = "gray", alpha=0.5)
                ax.set_title(titles[j])
                ax.axis('off')

            plt.tight_layout()
            fig.savefig(os.path.join(img_folder, f"img_{sid+i}.png"))

        sid = eid

        if eid > 100:
            break


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


# CUDA_VISIBLE_DEVICES=4 python scripts/sample_conditional.py -r logs/2023-05-20T06-28-38_coco_scene_images_transformer