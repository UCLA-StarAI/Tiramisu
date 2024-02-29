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


class PC_O_DDIMSampler(O_DDIMSampler):
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

        try:
            unique_zs_fname = conf.get("latent_pc.pc_unique_fname", None)
        except ValueError:
            unique_zs_fname = None
        if unique_zs_fname is not None:
            try:
                vq_vocab_size = self.vq_model.first_stage_model.n_embed
            except AttributeError:
                vq_vocab_size = self.vq_model.n_embed

            unique_zs = torch.from_numpy(np.load(unique_zs_fname))
            zs_mapping = torch.zeros([vq_vocab_size], dtype = torch.long)
            zs_mapping[unique_zs] = torch.arange(0, unique_zs.size(0))

            self.unique_zs = unique_zs.to(self.vq_device)
            self.zs_mapping = zs_mapping.to(self.vq_device)

            if self.unique_zs.size(0) == vq_vocab_size:
                self.unique_zs = None
                self.zs_mapping = None
        else:
            self.unique_zs = None
            self.zs_mapping = None

        # Load PC
        ns = juice.io.load(conf.get("latent_pc.pc_fname", None))

        print("> Compiling PC...")
        self.pc = juice.TensorCircuit(ns)
        self.pc.to(self.pc_device)
        print("> Done")

        print(f"> Number of nodes: {self.pc.num_nodes}")
        print(f"> Number of edges: {self.pc.num_edges}")
        print(f"> Number of sum parameters: {self.pc.num_sum_params}")

        # # Load GPT
        # self.gpt_model = PatchGPT(
        #     vocab_size = 256, 
        #     block_size = 256,
        #     cond_vocab_size = self.vq_model.first_stage_model.quantize.n_e,
        #     cond_block_size = 1,
        #     n_layer = 4,
        #     n_head = 16,
        #     n_embd = 256,
        #     embd_pdrop = 0.1,
        #     resid_pdrop = 0.1,
        #     attn_pdrop = 0.1
        # )

        # sd = torch.load(conf.get("latent_pc.gpt_fname", None))
        # self.gpt_model.load_state_dict(sd)
        # self.gpt_model.to(self.gpt_device)
        # self.gpt_model.eval()

        # self.gpt_top_k = conf.get("latent_pc.gpt_top_k", 10)

        # Other hyperparameters
        self.top_k = conf.get("latent_pc.top_k", 10)

        self.mixing_prior_factor_start = conf.get("latent_pc.mixing_prior_factor_start", 0.4)
        self.mixing_prior_factor_end = conf.get("latent_pc.mixing_prior_factor_end", 1.0)
        self.mixing_exp_factor = conf.get("latent_pc.mixing_exp_factor", 3)

        self.num_latent_samples = conf.get("latent_pc.num_latent_samples", 8)

        self.num_inference_steps = conf.get("ddim.schedule_params.num_inference_steps", 250)

        self.detach_pc_frac = conf.get("latent_pc.detach_pc_frac", 0.8)

        self.min_temperature = conf.get("ddim.schedule_params.min_temperature", 0.01)
        self.max_temperature = conf.get("ddim.schedule_params.max_temperature", 0.1)


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
        if self.mode == "inpaint":
            def loss_fn(_x0, _pred_x0, _mask):
                ret = torch.sum((_x0 * _mask - _pred_x0 * _mask) ** 2)
                return ret
        elif self.mode == "super_resolution":
            size = x.shape[-1]
            downop = nn.AdaptiveAvgPool2d(
                (size // self.scale, size // self.scale))

            def loss_fn(_x0, _pred_x0, _mask):
                down_x0 = downop(_x0)
                down_pred_x0 = downop(_pred_x0)
                ret = torch.sum((down_x0 - down_pred_x0) ** 2)
                return ret
        else:
            raise ValueError("Unkown mode: {self.mode}")

        def reg_fn(_origin_xt, _xt):
            ret = torch.sum((_origin_xt - _xt) ** 2)
            return ret

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
        x0 = model_kwargs["gt"]
        mask = model_kwargs["gt_keep_mask"]

        # condition mean
        if cond_fn is not None:
            model_fn = self._wrap_model(model_fn)
            B, C = x.shape[:2]
            assert t.shape == (B,)
            model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            _, model_var_values = torch.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
            with torch.enable_grad():
                gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
                x = x + model_variance * gradient

        if self.use_smart_lr_xt_decay:
            lr_xt /= get_smart_lr_decay_rate(t, self.mid_interval_num)
        # optimize
        with torch.enable_grad():
            origin_x = x.clone().detach()
            x = x.detach().requires_grad_()
            e_t = get_et(_x=x, _t=t)
            pred_x0 = get_predx0(
                _x=x, _t=t, _et=e_t, interval_num=self.mid_interval_num
            )
            prev_loss = loss_fn(x0, pred_x0, mask).item()

            # logging_info(f"step: {t[0].item()} lr_xt {lr_xt:.8f}")
            for step in range(self.num_iteration_optimize_xt):
                loss = loss_fn(x0, pred_x0, mask) + \
                    coef_xt_reg * reg_fn(origin_x, x)
                x_grad = torch.autograd.grad(
                    loss, x, retain_graph=False, create_graph=False
                )[0].detach()
                new_x = x - lr_xt * x_grad

                # logging_info(
                #     f"grad norm: {torch.norm(x_grad, p=2).item():.3f} "
                #     f"{torch.norm(x_grad * mask, p=2).item():.3f} "
                #     f"{torch.norm(x_grad * (1. - mask), p=2).item():.3f}"
                # )

                while self.use_adaptive_lr_xt and True:
                    with torch.no_grad():
                        e_t = get_et(new_x, _t=t)
                        pred_x0 = get_predx0(
                            new_x, _t=t, _et=e_t, interval_num=self.mid_interval_num
                        )
                        new_loss = loss_fn(x0, pred_x0, mask) + coef_xt_reg * reg_fn(
                            origin_x, new_x
                        )
                        if not torch.isnan(new_loss) and new_loss <= loss:
                            break
                        else:
                            lr_xt *= 0.8
                            # logging_info(
                            #     "Loss too large (%.3lf->%.3lf)! Learning rate decreased to %.5lf."
                            #     % (loss.item(), new_loss.item(), lr_xt)
                            # )
                            del new_x, e_t, pred_x0, new_loss
                            new_x = x - lr_xt * x_grad

                x = new_x.detach().requires_grad_()
                e_t = get_et(x, _t=t)
                pred_x0 = get_predx0(
                    x, _t=t, _et=e_t, interval_num=self.mid_interval_num
                )
                del loss, x_grad
                torch.cuda.empty_cache()

        ## re-sample pred_x0 with PC ##
        # print(self.detach_pc_frac, t.max())
        if t.max() > self.num_inference_steps * self.detach_pc_frac:
            output = self.p_mean_variance(model_fn, x, t, model_kwargs = model_kwargs)
            pred_x0_mean = output["pred_xstart"]
            pred_x0_variance = output["variance"]

            pred_x0_mean = (pred_x0_mean * (1.0 - mask) + x0 * mask).cpu().to(self.vq_device)
            pred_x0_variance.masked_fill(mask > 0.999, 1e-4)
            pred_x0_variance = pred_x0_variance.cpu().to(self.vq_device)
            try:
                code_probs = self.vq_model.first_stage_model(pred_x0_mean, call_func = "prob_encode").exp()
            except AttributeError:
                code_probs = self.vq_model(pred_x0_mean, call_func = "prob_encode").exp()
            # code_probs = self.vq_model.first_stage_model(x0.cpu().to(self.vq_device), call_func = "prob_encode").exp() ### debug # (this is cheating)
            if self.unique_zs is not None:
                code_probs = code_probs[:,:,:,self.unique_zs]

            # Apply temperature scaling
            patch_h = 16
            patch_w = 16
            B = mask.size(0)
            num_patches = (mask.size(2) // patch_h, mask.size(3) // patch_w)
            conditioned_frac = mask.reshape(B, 1, num_patches[0], patch_h, num_patches[1], patch_w).sum(dim = 5).sum(dim = 3) / (patch_h * patch_w)
            # temperature = (0.1 + conditioned_frac * (0.01 - 0.1)).squeeze(1).unsqueeze(-1).cpu().to(self.vq_device)
            temperature = (self.max_temperature + conditioned_frac * (self.min_temperature - self.max_temperature)).squeeze(1).unsqueeze(-1).cpu().to(self.vq_device)

            scaled_code_prob = torch.softmax(code_probs.log() / temperature, dim = -1).reshape(B, num_patches[0] * num_patches[1], -1).cpu().to(self.pc_device)

            pc_mask = torch.zeros([B, num_patches[0] * num_patches[1]], dtype = torch.bool, device = self.pc_device)
            pz = juice.queries.conditional(self.pc, scaled_code_prob, missing_mask = pc_mask).reshape(B, num_patches[0], num_patches[1], -1)
            conditioned_frac = conditioned_frac[:,0,:,:,None].cpu().to(self.pc_device)
            pz = F.softmax(pz.log() * (1.0 - conditioned_frac) + scaled_code_prob.reshape(B, num_patches[0], num_patches[1], -1).log() * \
                conditioned_frac, dim = -1).reshape(B, num_patches[0], num_patches[1], -1) # This is newly added

            with torch.no_grad():
                v, _ = torch.topk(code_probs, self.top_k)
                pz[code_probs < v[:,:,:,[-1]]] = 1e-8
                pz_size = pz.size()
                z_samples = torch.multinomial(pz.flatten(0,2), num_samples = self.num_latent_samples, replacement = True).reshape(*pz_size[:3], self.num_latent_samples)
                z_samples = z_samples.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2).cpu().to(self.vq_device)
                if self.unique_zs is not None:
                    z_samples = self.zs_mapping[z_samples]

                try:
                    quant_z = self.vq_model.first_stage_model.quantize.get_codebook_entry(
                        z_samples, shape = (z_samples.size(0), num_patches[0], num_patches[1], self.vq_model.first_stage_model.quantize.e_dim)
                    )
                    x_samples = self.vq_model.first_stage_model.decode(quant_z)
                except AttributeError:
                    quant_z = self.vq_model.quantize.get_codebook_entry(
                        z_samples, shape = (z_samples.size(0), num_patches[0], num_patches[1], self.vq_model.quantize.e_dim)
                    )
                    x_samples = self.vq_model.decode(quant_z)
                x_samples = x_samples.reshape(B, self.num_latent_samples, 3, num_patches[0] * patch_h, num_patches[1] * patch_w)

                vals = torch.linspace(-1.0, 1.0, 256, device = self.vq_device)[None,None,None,None,:]
                prior_ps = truncated_gaussian_pdf(vals, pred_x0_mean[:,:,:,:,None], pred_x0_variance[:,:,:,:,None], k = 2.0)
                prior_ps /= prior_ps.sum(dim = 4, keepdim = True)

                posterior_ps = truncated_gaussian_pdf(vals, x_samples.mean(dim = 1)[:,:,:,:,None], x_samples.var(dim = 1)[:,:,:,:,None], k = 100.0)
                posterior_ps += 1e-3
                posterior_ps /= posterior_ps.sum(dim = 4, keepdim = True)

                mixing_prior_factor = (self.mixing_prior_factor_end - self.mixing_prior_factor_start) * (-self.mixing_exp_factor * t / self.num_inference_steps).exp() + self.mixing_prior_factor_start
                mixing_prior_factor = mixing_prior_factor.cpu().to(self.vq_device)[:,None,None,None]

                logits = mixing_prior_factor * prior_ps.log() + (1.0 - mixing_prior_factor) * posterior_ps.log()
                probs = torch.softmax(logits, dim = 4)
                p_size = probs.size()
                pred_x0 = torch.multinomial(probs.flatten(0,3), num_samples=1).reshape(p_size[:4]) / 127.5 - 1.0
                pred_x0 = pred_x0.cpu().to(x0.device)

        # after optimize
        with torch.no_grad():
            new_loss = loss_fn(x0, pred_x0, mask).item()
            # logging_info("Loss Change: %.3lf -> %.3lf" % (prev_loss, new_loss))
            new_reg = reg_fn(origin_x, new_x).item()
            # logging_info("Regularization Change: %.3lf -> %.3lf" % (0, new_reg))
            pred_x0, e_t, x = pred_x0.detach(), e_t.detach(), x.detach()
            del origin_x, prev_loss
            x_prev = get_update(
                x,
                t,
                prev_t,
                e_t,
                _pred_x0=pred_x0,
            )

        if "record_traj" in model_kwargs and model_kwargs["record_traj"]:
            import matplotlib.pyplot as plt

            ccount = model_kwargs["ccount"]
            if not os.path.exists(f"traj_imgs/pc/{ccount}/"):
                os.mkdir(f"traj_imgs/pc/{ccount}/")

            plt.figure()
            image = np.transpose(pred_x0[0,:,:,:].cpu().numpy(), (1, 2, 0))
            image = (image + 1) / 2
            plt.imshow(image)
            plt.axis('off')
            myt = t.max().cpu().item()
            plt.savefig(f"traj_imgs/pc/{ccount}/recon_timestep_{myt}.png")

            plt.figure()
            image = np.transpose(x_prev[0,:,:,:].cpu().numpy(), (1, 2, 0))
            image = (image + 1) / 2
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(f"traj_imgs/pc/{ccount}/xprev_timestep_{myt}.png")

            try:
                pc_img = x_samples.mean(dim = 1)
                # import pdb; pdb.set_trace()
                plt.figure()
                image = np.transpose(pc_img[0,:,:,:].cpu().numpy(), (1, 2, 0))
                image = (image + 1) / 2
                plt.imshow(image)
                plt.axis('off')
                plt.savefig(f"traj_imgs/pc/{ccount}/x_pc_timestep_{myt}.png")

                dm_img = output["pred_xstart"]
                plt.figure()
                image = np.transpose(dm_img[0,:,:,:].detach().cpu().numpy(), (1, 2, 0))
                image = (image + 1) / 2
                plt.imshow(image)
                plt.axis('off')
                plt.savefig(f"traj_imgs/pc/{ccount}/x_dm_timestep_{myt}.png")
            except:
                pass

        # import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()

        return {"x": x, "x_prev": x_prev, "pred_x0": pred_x0, "loss": new_loss}


class PC_RepaintSampler(SpacedDiffusion):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )
        self.ddim_sigma = conf.get("ddim.ddim_sigma", 0.0)

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

        self.mixing_prior_factor_start = conf.get("latent_pc.mixing_prior_factor_start", 0.4)
        self.mixing_prior_factor_end = conf.get("latent_pc.mixing_prior_factor_end", 1.0)
        self.mixing_exp_factor = conf.get("latent_pc.mixing_exp_factor", 3)

        self.num_latent_samples = conf.get("latent_pc.num_latent_samples", 8)

        self.num_inference_steps = conf.get("ddim.schedule_params.num_inference_steps", 250)

        self.detach_pc_frac = conf.get("ddim.schedule_params.detach_pc_frac", 0.8)

        self.min_temperature = conf.get("ddim.schedule_params.min_temperature", 0.01)
        self.max_temperature = conf.get("ddim.schedule_params.max_temperature", 0.1)

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        conf=None,
        meas_fn=None,
        pred_xstart=None,
        idx_wall=-1,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        noise = th.randn_like(x)
        if conf.repaint["inpa_inj_sched_prev"]:

            if pred_xstart is not None:
                gt_keep_mask = model_kwargs.get("gt_keep_mask")
                gt = model_kwargs["gt"]
                alpha_cumprod = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

                if conf.repaint["inpa_inj_sched_prev_cumnoise"]:
                    weighed_gt = self.get_gt_noised(gt, int(t[0].item()))
                else:
                    gt_weight = th.sqrt(alpha_cumprod)
                    gt_part = gt_weight * gt

                    noise_weight = th.sqrt((1 - alpha_cumprod))
                    noise_part = noise_weight * th.randn_like(x)

                    weighed_gt = gt_part + noise_part

                x = gt_keep_mask * weighed_gt + (1 - gt_keep_mask) * x

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        x0 = model_kwargs.get("gt")
        mask = model_kwargs.get("gt_keep_mask")

        ########################

        ## re-sample pred_x0 with PC ##
        if t.max() > self.num_inference_steps * 0.8:
            output = self.p_mean_variance(
                model, x, t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs
            )
            pred_x0_mean = output["pred_xstart"]
            pred_x0_variance = output["variance"]

            pred_x0_mean = (pred_x0_mean * (1.0 - mask) + x0 * mask).cpu().to(self.vq_device)
            pred_x0_variance.masked_fill(mask > 0.999, 1e-4)
            pred_x0_variance = pred_x0_variance.cpu().to(self.vq_device)
            code_probs = self.vq_model.first_stage_model(pred_x0_mean, call_func = "prob_encode").exp()
            # code_probs = self.vq_model.first_stage_model(x0.cpu().to(self.vq_device), call_func = "prob_encode").exp() ### debug # (this is cheating)

            # Apply temperature scaling
            patch_h = 16
            patch_w = 16
            B = mask.size(0)
            num_patches = (mask.size(2) // patch_h, mask.size(3) // patch_w)
            conditioned_frac = mask.reshape(B, 1, num_patches[0], patch_h, num_patches[1], patch_w).sum(dim = 5).sum(dim = 3) / (patch_h * patch_w)
            # temperature = (0.1 + conditioned_frac * (0.01 - 0.1)).squeeze(1).unsqueeze(-1).cpu().to(self.vq_device)
            temperature = (self.max_temperature + conditioned_frac * (self.min_temperature - self.max_temperature)).squeeze(1).unsqueeze(-1).cpu().to(self.vq_device)

            scaled_code_prob = torch.softmax(code_probs.log() / temperature, dim = -1).reshape(B, num_patches[0] * num_patches[1], -1).cpu().to(self.pc_device)

            pc_mask = torch.zeros([B, num_patches[0] * num_patches[1]], dtype = torch.bool, device = self.pc_device)
            pz = juice.queries.conditional(self.pc, scaled_code_prob, missing_mask = pc_mask).reshape(B, num_patches[0], num_patches[1], -1)
            conditioned_frac = conditioned_frac[:,0,:,:,None].cpu().to(self.pc_device)
            pz = F.softmax(pz.log() * (1.0 - conditioned_frac) + scaled_code_prob.reshape(B, num_patches[0], num_patches[1], -1).log() * \
                conditioned_frac, dim = -1).reshape(B, num_patches[0], num_patches[1], -1) # This is newly added

            with torch.no_grad():
                v, _ = torch.topk(code_probs, self.top_k)
                pz[code_probs < v[:,:,:,[-1]]] = 1e-8
                pz_size = pz.size()
                z_samples = torch.multinomial(pz.flatten(0,2), num_samples = self.num_latent_samples, replacement = True).reshape(*pz_size[:3], self.num_latent_samples)
                z_samples = z_samples.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2).cpu().to(self.vq_device)

                quant_z = self.vq_model.first_stage_model.quantize.get_codebook_entry(
                    z_samples, shape = (z_samples.size(0), num_patches[0], num_patches[1], self.vq_model.first_stage_model.quantize.e_dim)
                )
                x_samples = self.vq_model.first_stage_model.decode(quant_z)
                x_samples = x_samples.reshape(B, self.num_latent_samples, 3, num_patches[0] * patch_h, num_patches[1] * patch_w)

                vals = torch.linspace(-1.0, 1.0, 256, device = self.vq_device)[None,None,None,None,:]
                prior_ps = truncated_gaussian_pdf(vals, pred_x0_mean[:,:,:,:,None], pred_x0_variance[:,:,:,:,None], k = 2.0)
                prior_ps /= prior_ps.sum(dim = 4, keepdim = True)

                posterior_ps = truncated_gaussian_pdf(vals, x_samples.mean(dim = 1)[:,:,:,:,None], x_samples.var(dim = 1)[:,:,:,:,None], k = 100.0)
                posterior_ps += 1e-3
                posterior_ps /= posterior_ps.sum(dim = 4, keepdim = True)

                mixing_prior_factor = (self.mixing_prior_factor_end - self.mixing_prior_factor_start) * (-self.mixing_exp_factor * t / self.num_inference_steps).exp() + self.mixing_prior_factor_start
                mixing_prior_factor = mixing_prior_factor.cpu().to(self.vq_device)[:,None,None,None]

                logits = mixing_prior_factor * prior_ps.log() + (1.0 - mixing_prior_factor) * posterior_ps.log()
                probs = torch.softmax(logits, dim = 4)
                p_size = probs.size()
                pred_x0 = torch.multinomial(probs.flatten(0,3), num_samples=1).reshape(p_size[:4]) / 127.5 - 1.0
                pred_x0 = pred_x0.cpu().to(x0.device)

            sample, _, log_variance = self.q_posterior_mean_variance(
                x_start=pred_x0, x_t=x, t=t
            )
            sample = sample + nonzero_mask * th.exp(0.5 * log_variance) * noise
            out = {"pred_xstart": pred_x0}

        else:
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )

            sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

            pred_x0 = out["pred_xstart"]

        ########################

        if cond_fn is not None:
            raise ValueError()

        # import matplotlib.pyplot as plt
        # plt.figure()
        # image = np.transpose(pred_x0[0,:,:,:].cpu().numpy(), (1, 2, 0))
        # image = (image + 1) / 2
        # plt.imshow(image)
        # plt.axis('off')
        # plt.savefig("debug.png")

        # import pdb; pdb.set_trace()

        result = {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "gt": model_kwargs.get("gt"),
        }

        return result