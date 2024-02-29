import logging
import os

import sys
sys.path.append("./")
sys.path.append("../../")
sys.path.append("../../controlled_img_modeling/")
sys.path.append("../../external/CoPaint/")
sys.path.append("../../external/latent-diffusion/")
sys.path.append("../../external/taming-transformers/")

from tools.args import from_argparse_args
from tools.utils import instantiate_from_config
from omegaconf import OmegaConf

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

import pyjuice as juice

# from datasets import load_lama_celebahq, load_imagenet
from copaint_datasets.utils import normalize
from guided_diffusion import (
    DDIMSampler,
    O_DDIMSampler,
    PC_O_DDIMSampler,
    PC_RepaintSampler,
    DDNMSampler,
    DDRMSampler,
    DPSSampler,
)
from guided_diffusion import dist_util
from guided_diffusion.ddim import R_DDIMSampler
from guided_diffusion.respace import SpacedDiffusion
from guided_diffusion.script_util import (
    model_defaults,
    create_model,
    diffusion_defaults,
    create_gaussian_diffusion,
    select_args,
    create_classifier,
    classifier_defaults,
)
from metrics import LPIPS, PSNR, SSIM, Metric
from copaint_utils import save_grid, save_image, normalize_image
from copaint_utils.config import Config
from copaint_utils.logger import get_logger, logging_info
from copaint_utils.nn_utils import get_all_paths, set_random_seed
from copaint_utils.result_recorder import ResultRecorder
from copaint_utils.timer import Timer


# def normalize(image, shape=(256, 256)):
#     """
#     Given an PIL image, resize it and normalize each pixel into [-1, 1].
#     Args:
#         image: image to be normalized, PIL.Image
#         shape: the desired shape of the image

#     Returns: the normalized image

#     """
#     image = np.array(image.convert("RGB").resize(shape))
#     image = image.astype(np.float32) / 255.0
#     image = image[None].transpose(0, 3, 1, 2)
#     image = torch.from_numpy(image)
#     image = image * 2.0 - 1.0
#     return image


def prepare_model(algorithm, conf, device):
    logging_info("Prepare model...")
    unet = create_model(**select_args(conf, model_defaults().keys()), conf=conf)
    SAMPLER_CLS = {
        "repaint": SpacedDiffusion,
        "pc_repaint": PC_RepaintSampler,
        "ddim": DDIMSampler,
        "o_ddim": O_DDIMSampler,
        "Tiramisu": PC_O_DDIMSampler,
        "resample": R_DDIMSampler,
        "ddnm": DDNMSampler,
        "ddrm": DDRMSampler,
        "dps": DPSSampler,
    }
    sampler_cls = SAMPLER_CLS[algorithm]
    sampler = create_gaussian_diffusion(
        **select_args(conf, diffusion_defaults().keys()),
        conf=conf,
        base_cls=sampler_cls,
    )

    logging_info(f"Loading model from {conf.model_path}...")
    unet.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.model_path), map_location="cpu"
        ), strict=False
    )
    unet.to(device)
    if conf.use_fp16:
        unet.convert_to_fp16()
    unet.eval()
    return unet, sampler


def prepare_classifier(conf, device):
    logging_info("Prepare classifier...")
    classifier = create_classifier(
        **select_args(conf, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.classifier_path), map_location="cpu"
        )
    )
    classifier.to(device)
    classifier.eval()
    return classifier


# def prepare_data(
#     dataset_name, mask_type="half", dataset_starting_index=-1, dataset_ending_index=-1
# ):
#     if dataset_name == "celebahq":
#         datas = load_lama_celebahq(mask_type=mask_type)
#     elif dataset_name == "imagenet":
#         datas = load_imagenet(mask_type=mask_type)
#     elif dataset_name == "imagenet64":
#         datas = load_imagenet(mask_type=mask_type, shape=(64, 64))
#     elif dataset_name == "imagenet128":
#         datas = load_imagenet(mask_type=mask_type, shape=(128, 128))
#     elif dataset_name == "imagenet512":
#         datas = load_imagenet(mask_type=mask_type, shape=(512, 512))
#     else:
#         raise NotImplementedError

#     dataset_starting_index = (
#         0 if dataset_starting_index == -1 else dataset_starting_index
#     )
#     dataset_ending_index = (
#         len(datas) if dataset_ending_index == -1 else dataset_ending_index
#     )
#     datas = datas[dataset_starting_index:dataset_ending_index]

#     logging_info(f"Load {len(datas)} samples")
#     return datas


def all_exist(paths):
    for p in paths:
        if not os.path.exists(p):
            return False
    return True


def main():
    ###################################################################################
    # prepare config, logger and recorder
    ###################################################################################
    config = Config(default_config_file="configs/celeba.yaml", use_argparse=True)
    config.show()

    print("Default config loaded")

    all_paths = get_all_paths(config.outdir)
    config.dump(all_paths["path_config"])
    # get_logger(all_paths["path_log"], force_add_handler=True)
    recorder = ResultRecorder(
        path_record=all_paths["path_record"],
        initial_record=config,
        use_git=config.use_git,
    )
    # set_random_seed(config.seed, deterministic=False, no_torch=False, no_tf=True)

    ###################################################################################
    # prepare data
    ###################################################################################
    if config.input_image == "":  # if input image is not given, load dataset
        # datas = prepare_data(
        #     config.dataset_name,
        #     config.mask_type,
        #     config.dataset_starting_index,
        #     config.dataset_ending_index,
        # )

        config["data_config"]["train"]["params"]["mask_type"] = config["mask_type"]
        config["data_config"]["validation"]["params"]["mask_type"] = config["mask_type"]

        train_dataset = instantiate_from_config(OmegaConf.create(config["data_config"]["train"]))
        val_dataset = instantiate_from_config(OmegaConf.create(config["data_config"]["validation"]))

        tr_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size = 1, shuffle = False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size = 1, shuffle = False
        )
        datas = val_loader
        
    else:
        # NOTE: the model should accepet this input image size
        image = normalize(Image.open(config.input_image).convert("RGB"))
        if config.mode != "super_resolution":
            mask = (
                torch.from_numpy(np.array(Image.open(config.mask).convert("1"), dtype=np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            mask = torch.from_numpy(np.array([0]))  # just a dummy value
        datas = [(image, mask, "sample0")]

    print("> Data loaded")

    ###################################################################################
    # prepare model and device
    ###################################################################################
    # device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:1")
    unet, sampler = prepare_model(config.algorithm, config, device)

    def model_fn(x, t, y=None, gt=None, **kwargs):
        return unet(x, t, y if config.class_cond else None, gt=gt)
    
    cond_fn = None

    METRICS = {
        "lpips": Metric(LPIPS("alex", device)),
        "psnr": Metric(PSNR(), eval_type="max"),
        "ssim": Metric(SSIM(), eval_type="max"),
    }
    final_loss = []

    ###################################################################################
    # prepare save dir
    ###################################################################################
    img_path = os.path.join(config.outdir, "imgs/")
    os.makedirs(img_path, exist_ok=True)
    metric_path = os.path.join(config.outdir, "metrics/")
    os.makedirs(metric_path, exist_ok=True)

    ###################################################################################
    # start sampling
    ###################################################################################
    logging_info("Start sampling")
    timer, num_image = Timer(), 0
    batch_size = config.n_samples

    count = 0
    for data in tqdm(datas):
        if config.class_cond:
            image, mask, image_name, class_id = data
            if isinstance(image_name, tuple):
                image_name = image_name[0]
            class_id = class_id[0]
        else:
            image, mask, image_name = data
            if isinstance(image_name, tuple):
                image_name = image_name[0]
            class_id = None

        # prepare batch data for processing
        batch = {"image": image.to(device), "mask": mask.to(device)}
        model_kwargs = {
            "gt": batch["image"].repeat(batch_size, 1, 1, 1),
            "gt_keep_mask": batch["mask"].repeat(batch_size, 1, 1, 1),
        }

        if config.class_cond:
            if config.cond_y is not None:
                classes = torch.ones(batch_size, dtype=torch.long, device=device)
                model_kwargs["y"] = classes * config.cond_y
            elif config.classifier_path is not None:
                classes = torch.full((batch_size,), class_id, device=device)
                model_kwargs["y"] = classes

        shape = (batch_size, 3, config.image_size, config.image_size)

        orig_img_fname = os.path.join(img_path, f"{image_name}_origin.png")
        img_fname = os.path.join(img_path, f"{image_name}_inpainted.png")
        metric_fname = os.path.join(metric_path, f"{image_name}_metrics.txt")

        if os.path.exists(img_fname) and os.path.exists(metric_fname):
            print(f"skip image {image_name}")
            count += 1
            continue

        # sample images
        result = sampler.p_sample_loop(
            model_fn,
            shape=shape,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=True,
            return_all=True,
            conf=config,
            sample_dir=outpath if config["debug"] else None,
        )

        mask = mask.cpu()
        result["sample"] = (1.0 - mask) * result["sample"].cpu() + mask * batch["image"].cpu()
        batch["image"] = batch["image"].cpu()

        for metric in METRICS.values():
            metric.update(result["sample"], batch["image"])

        inpainted = normalize_image(result["sample"])

        # Save results
        save_grid(inpainted, img_fname)
        save_grid(normalize_image(batch["image"]), orig_img_fname)

        with open(metric_fname, "w") as f:
            for name, metric in METRICS.items():
                batch_metric = metric.report_batch()
                f.write(f"{name}: {batch_metric}\n")

        count += 1
        if count >= 200:
            break



if __name__ == "__main__":
    main()

