import math
import os
from omegaconf import OmegaConf

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from tqdm import tqdm

from copaint_utils.logger import logging_info
from .gaussian_diffusion import _extract_into_tensor
from .new_scheduler import ddim_timesteps, ddim_repaint_timesteps
from .respace import SpacedDiffusion
from .ddim import O_DDIMSampler, DDIMSampler

import pyjuice as juice

import sys
sys.path.append("./")
sys.path.append("../../external/taming-transformers/")

from tools.utils import instantiate_from_config

# from patchgpt import PatchGPT


def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


def truncated_gaussian_pdf(inputs, mean, variance, k=2.0):
    """
    Compute the truncated Gaussian probability density function (PDF) for a vector of inputs.

    Args:
        inputs (torch.Tensor): The input tensor for which to compute the PDF.
        mean (torch.Tensor): The mean tensor of the Gaussian distribution.
        variance (torch.Tensor): The variance tensor of the Gaussian distribution.
        k (float): The truncation parameter in terms of standard deviation.

    Returns:
        torch.Tensor: The PDF values for each input in the inputs tensor.
    """
    # Calculate the standard deviation from the variance (square root)
    std_dev = torch.sqrt(variance)
    
    # Calculate the lower and upper truncation bounds
    lower_bound = mean - k * std_dev
    upper_bound = mean + k * std_dev
    
    # Calculate the exponential term of the PDF
    exponent = -0.5 * ((inputs - mean) / std_dev)**2
    
    # Calculate the normalization constant for the PDF
    normalization_const = 1.0 / (std_dev * torch.sqrt(torch.tensor(2.0 * 3.14159265358979323846)))  # 2 * pi
    
    # Calculate the final PDF values
    pdf = normalization_const * torch.exp(exponent)
    
    # Apply truncation by setting PDF values to zero for inputs outside the range
    pdf = torch.where((inputs >= lower_bound) & (inputs <= upper_bound), pdf, torch.tensor(0.0))
    
    return pdf


class PCSemanticFusionSampler(O_DDIMSampler):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )

        self.vq_device = torch.device("cuda:0")
        self.pc_device = torch.device("cuda:0")
        self.gpt_device = torch.device("cuda:0")

        # Load VQ
        vq_conf = conf.get("latent_pc.vq_model.config_folder", None)
        config_base = os.path.join(vq_conf, "configs/")
        config = None
        for config_fname in os.listdir(config_base):
            curr_config = OmegaConf.load(os.path.join(config_base, config_fname))
            if config is None:
                config = curr_config
            else:
                config = OmegaConf.merge(config, curr_config)
        self.vq_model = instantiate_from_config(config.model)

        sd = torch.load(os.path.join(vq_conf, "checkpoints/last.ckpt"), map_location = "cpu")
        self.vq_model.load_state_dict(sd["state_dict"])

        self.vq_model.to(self.vq_device)
        self.vq_model.eval()

        # Load PC
        ns = juice.io.load(conf.get("latent_pc.pc_fname", None))

        print("> Compiling PC...")
        self.pc = juice.TensorCircuit(ns)
        self.pc.to(self.pc_device)
        print("> Done")

        print(f"> Number of nodes: {self.pc.num_nodes}")
        print(f"> Number of edges: {self.pc.num_edges}")
        print(f"> Number of sum parameters: {self.pc.num_sum_params}")

        # Other hyperparameters
        self.top_k = conf.get("latent_pc.top_k", 10)

        self.mixing_prior_factor_start = conf.get("semantic_fusion.mixing_prior_factor_start", 0.4)
        self.mixing_prior_factor_end = conf.get("semantic_fusion.mixing_prior_factor_end", 1.0)
        self.mixing_exp_factor = conf.get("semantic_fusion.mixing_exp_factor", 3)

        self.num_latent_samples = conf.get("latent_pc.num_latent_samples", 8)

        self.num_inference_steps = conf.get("ddim.schedule_params.num_inference_steps", 250)

        self.detach_pc_frac = conf.get("ddim.schedule_params.detach_pc_frac", 0.8)

        self.img_temperature = conf.get("semantic_fusion.img_temperature", 0.1)

        self.ref_temperature = conf.get("semantic_fusion.ref_temperature", 0.1)

        self.mixing_ref_frac = conf.get("semantic_fusion.mixing_ref_frac", 0.8)

    def get_latent_prior(self, ref_x, multi_mask):
        B = ref_x.size(0)
        patch_h = 16
        patch_w = 16

        with torch.no_grad():

            multi_mask = multi_mask.cpu().to(self.vq_device)

            try:
                code_probs = self.vq_model.first_stage_model(ref_x.cpu().to(self.vq_device), call_func = "prob_encode").exp()
            except AttributeError:
                code_probs = self.vq_model(ref_x.cpu().to(self.vq_device), call_func = "prob_encode").exp()

            num_patches = (code_probs.size(1), code_probs.size(2))

            scaled_ref_code_prob = torch.softmax(code_probs.log() / self.ref_temperature, dim = -1).reshape(B, num_patches[0], num_patches[1], -1).cpu().to(self.pc_device)

            ref_code_prob = torch.zeros([num_patches[0], num_patches[1], scaled_ref_code_prob.size(3)], device = self.pc_device)
            for i in range(B):
                conditioned_frac = (multi_mask == i).float().reshape(num_patches[0], patch_h, num_patches[1], patch_w).sum(dim = 3).sum(dim = 1) / (patch_h * patch_w)
                ref_code_prob += conditioned_frac.unsqueeze(2) * scaled_ref_code_prob[i,:,:,:]
            ref_code_prob += 1e-6
            ref_code_prob /= ref_code_prob.sum(dim = 2, keepdim = True)

            return ref_code_prob.unsqueeze(0)

    def p_sample(
        self,
        model_fn,
        x,
        t,
        prev_t,
        model_kwargs,
        lr_xt,
        coef_xt_reg,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):
        def process_xstart(_x):
            if denoised_fn is not None:
                _x = denoised_fn(_x)
            if clip_denoised:
                return _x.clamp(-1, 1)
            return _x

        def get_et(_x, _t):
            if self.mid_interval_num > 1:
                res = grad_ckpt(
                    self._get_et, model_fn, _x, _t, model_kwargs, use_reentrant=False
                )
            else:
                res = self._get_et(model_fn, _x, _t, model_kwargs)
            return res

        def get_smart_lr_decay_rate(_t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)

            ret = 1
            time_pairs = list(zip(steps[:-1], steps[1:]))
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                ret *= self.sqrt_recip_alphas_cumprod[_cur_t] * math.sqrt(
                    self.alphas_cumprod[_prev_t]
                )
            return ret

        def multistep_predx0(_x, _et, _t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)
            time_pairs = list(zip(steps[:-1], steps[1:]))
            x_t = _x
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                _cur_t = torch.tensor([_cur_t] * _x.shape[0], device=_x.device)
                _prev_t = torch.tensor(
                    [_prev_t] * _x.shape[0], device=_x.device)
                if i != 0:
                    _et = get_et(x_t, _cur_t)
                x_t = grad_ckpt(
                    get_update, x_t, _cur_t, _prev_t, _et, None, use_reentrant=False
                )
            return x_t

        def get_predx0(_x, _t, _et, interval_num=1):
            if interval_num == 1:
                return process_xstart(self._predict_xstart_from_eps(_x, _t, _et))
            else:
                _pred_x0 = grad_ckpt(
                    multistep_predx0, _x, _et, _t, interval_num, use_reentrant=False
                )
                return process_xstart(_pred_x0)

        def get_update(
            _x,
            cur_t,
            _prev_t,
            _et=None,
            _pred_x0=None,
        ):
            if _et is None:
                _et = get_et(_x=_x, _t=cur_t)
            if _pred_x0 is None:
                _pred_x0 = get_predx0(_x, cur_t, _et, interval_num=1)

            alpha_t = _extract_into_tensor(self.alphas_cumprod, cur_t, _x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, _prev_t, _x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )
            mean_pred = (
                _pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * _et  # dir_xt
            )
            noise = noise_like(_x.shape, _x.device, repeat=False)
            nonzero_mask = (cur_t != 0).float().view(-1,
                                                     *([1] * (len(_x.shape) - 1)))
            _x_prev = mean_pred + noise * sigmas * nonzero_mask
            return _x_prev

        B, C = x.shape[:2]
        assert t.shape == (B,)

        # condition mean
        if cond_fn is not None:
            raise ValueError()

        ## re-sample pred_x0 with PC ##
        with torch.no_grad():
            if t.max() > self.num_inference_steps * self.detach_pc_frac:
                output = self.p_mean_variance(model_fn, x, t)
                pred_x0_mean = output["pred_xstart"].cpu().to(self.vq_device)
                pred_x0_variance = output["variance"].cpu().to(self.vq_device)

                # Apply temperature scaling
                patch_h = 16
                patch_w = 16
                B = x.size(0)
                num_patches = (x.size(2) // patch_h, x.size(3) // patch_w)

                try:
                    code_probs = self.vq_model.first_stage_model(pred_x0_mean, call_func = "prob_encode").exp()
                except AttributeError:
                    code_probs = self.vq_model(pred_x0_mean, call_func = "prob_encode").exp()

                img_code_prob = torch.softmax(code_probs.log() / self.img_temperature, dim = -1).reshape(B, num_patches[0] * num_patches[1], -1).cpu().to(self.pc_device)

                ref_code_prob = model_kwargs["ref_code_prob"].reshape(1, num_patches[0] * num_patches[1], -1)

                conditioned_frac = (model_kwargs["multi_mask"] < 100).float().reshape(num_patches[0], patch_h, num_patches[1], patch_w).sum(dim = 3).sum(dim = 1) / (patch_h * patch_w)
                ref_frac = (self.mixing_ref_frac * conditioned_frac).clamp(min = 0.01, max = 0.99).reshape(num_patches[0] * num_patches[1])[None,:,None].cpu().to(self.pc_device)
                scaled_code_prob = F.softmax(img_code_prob.log() * (1.0 - ref_frac) + ref_code_prob.log() * ref_frac, dim = -1)

                pc_mask = torch.zeros([B, num_patches[0] * num_patches[1]], dtype = torch.bool, device = self.pc_device)
                pz = juice.queries.conditional(self.pc, scaled_code_prob, missing_mask = pc_mask)
                pz = F.softmax(pz.log() * (1.0 - ref_frac) + ref_code_prob.log() * ref_frac, dim = -1).reshape(B, num_patches[0], num_patches[1], -1)


                pz_size = pz.size()
                z_samples = torch.multinomial(pz.flatten(0,2), num_samples = self.num_latent_samples, replacement = True).reshape(*pz_size[:3], self.num_latent_samples)
                z_samples = z_samples.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2).cpu().to(self.vq_device)

                try:
                    quant_z = self.vq_model.first_stage_model.quantize.get_codebook_entry(
                        z_samples, shape = (z_samples.size(0), num_patches[0], num_patches[1], self.vq_model.first_stage_model.quantize.e_dim)
                    )
                    x_samples = self.vq_model.first_stage_model.decode(quant_z)
                    x_samples = x_samples.reshape(B, self.num_latent_samples, 3, num_patches[0] * patch_h, num_patches[1] * patch_w)
                except AttributeError:
                    quant_z = self.vq_model.quantize.get_codebook_entry(
                        z_samples, shape = (z_samples.size(0), num_patches[0], num_patches[1], self.vq_model.quantize.e_dim)
                    )
                    x_samples = self.vq_model.decode(quant_z)
                    x_samples = x_samples.reshape(B, self.num_latent_samples, 3, num_patches[0] * patch_h, num_patches[1] * patch_w)

                vals = torch.linspace(-1.0, 1.0, 256, device = self.vq_device)[None,None,None,None,:]
                prior_ps = truncated_gaussian_pdf(vals, pred_x0_mean[:,:,:,:,None], pred_x0_variance[:,:,:,:,None], k = 4.0)
                prior_ps /= prior_ps.sum(dim = 4, keepdim = True)

                posterior_ps = truncated_gaussian_pdf(vals, x_samples.mean(dim = 1)[:,:,:,:,None], x_samples.var(dim = 1)[:,:,:,:,None], k = 100.0)
                posterior_ps += 1e-3
                posterior_ps /= posterior_ps.sum(dim = 4, keepdim = True)

                # import pdb; pdb.set_trace()

                mixing_prior_factor = (self.mixing_prior_factor_end - self.mixing_prior_factor_start) * (-self.mixing_exp_factor * t / self.num_inference_steps).exp() + self.mixing_prior_factor_start
                mixing_prior_factor = mixing_prior_factor.cpu().to(self.vq_device)[:,None,None,None]

                logits = mixing_prior_factor * prior_ps.log() + (1.0 - mixing_prior_factor) * posterior_ps.log()
                probs = torch.softmax(logits, dim = 4)
                p_size = probs.size()
                pred_x0 = torch.multinomial(probs.flatten(0,3), num_samples=1).reshape(p_size[:4]) / 127.5 - 1.0
                pred_x0 = pred_x0.cpu().to(x.device)

            else:
                e_t = get_et(x, _t=t)
                pred_x0 = get_predx0(
                    x, _t=t, _et=e_t, interval_num=self.mid_interval_num
                )

        # after optimize
        with torch.no_grad():
            e_t = get_et(x, _t=t)
            pred_x0, e_t, x = pred_x0.detach(), e_t.detach(), x.detach()
            x_prev = get_update(
                x,
                t,
                prev_t,
                e_t,
                _pred_x0=pred_x0,
            )

        # import matplotlib.pyplot as plt
        # plt.figure()
        # image = np.transpose(pred_x0[0,:,:,:].cpu().numpy(), (1, 2, 0))
        # image = (image + 1) / 2
        # plt.imshow(image)
        # plt.axis('off')
        # plt.savefig("debug.png")

        # import pdb; pdb.set_trace()

        return {"x": x.detach(), "x_prev": x_prev.detach(), "pred_x0": pred_x0.detach(), "loss": 0.0}

    def p_sample_loop(
        self,
        model_fn,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None,
        sample_dir="",
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            assert not conf["optimize_xt.filter_xT"]
            img = noise
        else:
            xT_shape = (
                shape
                if not conf["optimize_xt.filter_xT"]
                else tuple([20] + list(shape[1:]))
            )
            img = torch.randn(xT_shape, device=device)

        if conf["optimize_xt.filter_xT"]:
            xT_losses = []
            for img_i in img:
                xT_losses.append(
                    self.p_sample(
                        model_fn,
                        x=img_i.unsqueeze(0),
                        t=torch.tensor([self.steps[0]] * 1, device=device),
                        prev_t=torch.tensor([0] * 1, device=device),
                        model_kwargs=model_kwargs,
                        pred_xstart=None,
                        lr_xt=self.lr_xt,
                        coef_xt_reg=self.coef_xt_reg,
                    )["loss"]
                )
            img = img[torch.argsort(torch.tensor(xT_losses))[: shape[0]]]

        time_pairs = list(zip(self.steps[:-1], self.steps[1:]))

        x_t = img
        # set up hyper paramer for this run
        lr_xt = self.lr_xt
        coef_xt_reg = self.coef_xt_reg
        loss = None

        status = None
        for cur_t, prev_t in tqdm(time_pairs):
            if cur_t > prev_t:  # denoise
                status = "reverse"
                cur_t = torch.tensor([cur_t] * shape[0], device=device)
                prev_t = torch.tensor([prev_t] * shape[0], device=device)
                output = self.p_sample(
                    model_fn,
                    x=x_t,
                    t=cur_t,
                    prev_t=prev_t,
                    model_kwargs=model_kwargs,
                    pred_xstart=None,
                    lr_xt=lr_xt,
                    coef_xt_reg=coef_xt_reg,
                )
                x_t = output["x_prev"]
                loss = output["loss"]

            else:  # time travel back
                if status == "reverse" and conf.get(
                    "optimize_xt.optimize_before_time_travel", False
                ):
                    # update xt if previous status is reverse
                    x_t = self.get_updated_xt(
                        model_fn,
                        x=x_t,
                        t=torch.tensor([cur_t] * shape[0], device=device),
                        model_kwargs=model_kwargs,
                        lr_xt=lr_xt,
                        coef_xt_reg=coef_xt_reg,
                    )
                status = "forward"
                assert prev_t == cur_t + 1, "Only support 1-step time travel back"
                prev_t = torch.tensor([prev_t] * shape[0], device=device)
                with torch.no_grad():
                    x_t = self._undo(x_t, prev_t)
                # undo lr decay
                logging_info(f"Undo step: {cur_t}")
                lr_xt /= self.lr_xt_decay
                coef_xt_reg /= self.coef_xt_reg_decay

        x_t = x_t.clamp(-1.0, 1.0)  # normalize
        return {"sample": x_t, "loss": loss}