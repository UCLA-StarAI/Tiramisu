import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Callback
from overrides import overrides

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
from typing import Optional, Union, Dict, Any
from pytorch_lightning.strategies import FSDPStrategy
import tempfile
import math
import uuid

import sys
sys.path.append("./")
sys.path.append("../../")
sys.path.append("../../external/taming-transformers/")
sys.path.append("../../external/VQ-Diffusion/")

from tools.args import from_argparse_args
from tools.utils import instantiate_from_config
from tools.data import get_input

import warnings
warnings.filterwarnings("ignore")


class EMA(Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """
    def __init__(self, decay: float = 0.9999, ema_device: Optional[Union[torch.device, str]] = None, pin_memory=True):
        super().__init__()
        self.decay = decay
        self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.
        
        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()
        
    @overrides
    def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        # Only keep track of EMA weights in rank zero.
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
            if self.ema_device:
                self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in self.ema_state_dict.items()}

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

        self._ema_state_dict_ready = True

    @rank_zero_only
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs) -> None:
        # Update EMA weights
        with torch.no_grad():
            for key, value in self.get_state_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                ema_value.copy_(self.decay * ema_value + (1. - self.decay) * value, non_blocking=True)

    @overrides
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))
        pl_module.trainer.training_type_plugin.broadcast(self.ema_state_dict, 0)
        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
            f"There are some keys missing in the ema static dictionary broadcasted. " \
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    @overrides
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        # Replace EMA weights with training weights
        pl_module.load_state_dict(self.original_state_dict, strict=False)

    @overrides
    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint["ema_state_dict"] = self.ema_state_dict
        checkpoint["_ema_state_dict_ready"] = self._ema_state_dict_ready

    @overrides
    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        self._ema_state_dict_ready = checkpoint["_ema_state_dict_ready"]
        self.ema_state_dict = checkpoint["ema_state_dict"]


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type = str, default = "models/2021-04-23T18-11-19_celebahq_transformer/")
    parser.add_argument("--latent-stage-config", type = str, default = "configs/latent_stage_models/mlm_medium_256_1024.yaml")
    parser.add_argument("--wandb-online", default = False, action = "store_true")
    parser.add_argument("--resume", default = False, action = "store_true")

    return parser


def img_mask_generator(B, num_vars, device):

    H = W = int(math.sqrt(num_vars))

    max_pow = int(math.floor(math.log2(H)))

    masked_vars = torch.zeros([B, H, W], dtype = torch.bool)
    for i in range(B):
        powx = 2**np.random.randint(1, max_pow + 1)
        powy = 2**np.random.randint(1, max_pow + 1)
        x_s = np.random.randint(0, H - powx + 1)
        y_s = np.random.randint(0, W - powy + 1)

        masked_vars[i,x_s:x_s+powx,y_s:y_s+powy] = True

    masked_vars = masked_vars.reshape(B, H * W).to(device)

    return masked_vars


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
    base_folder = "outputs/latents/"
    model_name = os.path.basename(opt.config).split(".")[0]
    data_folder = os.path.join(base_folder, model_name)
    assert os.path.exists(data_folder)

    config.data.params.train.target = "tools.data.ZCNumpyDataset"
    config.data.params.train.params = {
        "data_path": data_folder,
        "split": "train",
        "key_mapping": {"c_latents": "c", "z_latents": "z"},
    }
    config.data.params.validation.target = "tools.data.ZCNumpyDataset"
    config.data.params.validation.params = {
        "data_path": data_folder,
        "split": "validation",
        "key_mapping": {"c_latents": "c", "z_latents": "z"}
    }

    dsets = instantiate_from_config(config.data)
    dsets.prepare_data()
    dsets.setup()

    # Load teacher model
    latent_stage_config = OmegaConf.load(opt.latent_stage_config)
    model = instantiate_from_config(latent_stage_config.model)

    model.mask_generator = img_mask_generator

    # Trainer config
    if not os.path.exists("outputs/latent_model_ckpts/"):
        os.mkdir("outputs/latent_model_ckpts/")
    full_model_name = os.path.basename(opt.latent_stage_config).split(".")[0] + "-" + model_name
    ckptdir = os.path.join(
        "outputs/latent_model_ckpts/",
        full_model_name
    )
    if not os.path.exists(ckptdir):
        os.mkdir(ckptdir)

    ckpt_models = os.path.join(ckptdir, "ckpts/")
    if not os.path.exists(ckpt_models):
        os.mkdir(ckpt_models)

    ckpt_configs = os.path.join(ckptdir, "configs/")
    if not os.path.exists(ckpt_configs):
        os.mkdir(ckpt_configs)

    OmegaConf.save(config, os.path.join(ckpt_configs, "config.yaml"))
    OmegaConf.save(latent_stage_config, os.path.join(ckpt_configs, "latent_stage_config.yaml"))

    modelckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath = os.path.join(ckptdir, "ckpts/"),
        filename = "{epoch:06}",
        verbose = True,
        save_last = True
    )
    trainer_kwargs = dict()
    trainer_kwargs["callbacks"] = [modelckpt_callback]
    trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"
    # strategy = FSDPStrategy(cpu_offload=False)
    # trainer_kwargs["strategy"] = "fsdp"
    trainer_kwargs["precision"] = 16
    trainer_kwargs["devices"] = [i for i in range(torch.cuda.device_count())]
    trainer_kwargs["accelerator"] = "gpu"
    trainer_kwargs["accumulate_grad_batches"] = latent_stage_config.trainer.accumulate_grad_batches

    for k, v in latent_stage_config.trainer.items():
        if k != "accumulate_grad_batches":
            trainer_kwargs[k] = v

    try:
        trainer_kwargs["callbacks"].append(EMA(decay = latent_stage_config.ema.decay, ema_device = latent_stage_config.ema.device))
    except Exception as e:
        pass

    # Resume if possible
    last_ckpt_path = os.path.join(ckpt_models, "last.ckpt")
    if opt.resume and os.path.exists(last_ckpt_path):
        resume_ckpt_path = last_ckpt_path
    else:
        resume_ckpt_path = None

    wandb_logger = pl.loggers.WandbLogger(
        name = full_model_name + str(uuid.uuid4()),
        save_dir = os.path.join(ckptdir, "wandb/"),
        offline = not opt.wandb_online,
        id = full_model_name + str(uuid.uuid4())
    )
    trainer_kwargs["logger"] = wandb_logger

    trainer = from_argparse_args(Trainer, None, **trainer_kwargs)

    # Set learning rate
    ngpus = torch.cuda.device_count()
    acc_grad_batches = latent_stage_config.trainer.accumulate_grad_batches
    bs = config.data.params.batch_size
    base_lr = latent_stage_config.model.base_learning_rate
    lr = base_lr * bs * ngpus * acc_grad_batches
    model.learning_rate = lr
    print(f">> Learning rate set to {lr}")

    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckpt_models, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb; pudb.set_trace()

    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    try:
        if opt.resume and os.path.exists(last_ckpt_path):
            trainer.fit(model, dsets, ckpt_path = last_ckpt_path)
        else:
            trainer.fit(model, dsets)
    except Exception:
        melk()
        raise


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()