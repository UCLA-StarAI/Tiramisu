import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import pytorch_lightning as pl
import importlib
from omegaconf import OmegaConf
from copy import deepcopy
from tqdm import tqdm
from argparse import Namespace

from .base import CrossBlock, Block


def random_mask_generator(B, num_vars, device):
    n_cond = torch.randint(1, args.num_vars, [B])
    masked_vars = torch.zeros([B, T], dtype = torch.bool)
    for i in range(B):
        ids = torch.multinomial(torch.ones([args.num_vars]), n_cond[i])
        masked_vars[i,ids] = True
    masked_vars = masked_vars.to(device)

    return masked_vars


def instantiate_from_config(config, **kwargs):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    param_dict = OmegaConf.to_container(config.get("params", dict()), resolve = True)
    for k, v in kwargs.items():
        param_dict[k] = v
    return cls(**param_dict)


class MLM(pl.LightningModule):

    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size, 
                 cond_vocab_size, cond_block_size,
                 embd_pdrop = 0.1, resid_pdrop = 0.1, attn_pdrop = 0.1, new_mode = False,
                 mask_generator = random_mask_generator, content_emb = None, scheduler_config = None):
        super(MLM, self).__init__()

        self.scheduler_config = scheduler_config

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        self.cond_vocab_size = cond_vocab_size
        self.cond_block_size = cond_block_size

        self.mask_generator = mask_generator

        self.new_mode = new_mode

        config = Namespace(
            n_head = n_head, n_embd = n_embd, 
            resid_pdrop = resid_pdrop, attn_pdrop = attn_pdrop,
            block_size = self.block_size
        )

        self.content_emb = instantiate_from_config(content_emb)

        self.transformer = nn.ModuleDict(dict(
            cond_wte = nn.Embedding(self.cond_vocab_size, self.n_embd),
            cond_wpe = nn.Embedding(self.cond_block_size, self.n_embd),
            drop = nn.Dropout(embd_pdrop),
            h = nn.ModuleList([CrossBlock(config, causal = False) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(self.n_embd)
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias = False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("GPT number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 1e-2},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.999))
        
        if self.scheduler_config is None:
            return optimizer

        else:
            scheduler = instantiate_from_config(self.scheduler_config, optimizer = optimizer, warmup_lr = self.learning_rate)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "monitor": "train/loss",
                    "frequency": 1
                },
            }

    def forward(self, idx, cond_idx, masked_vars):
        """
        idx: [B, T]
        cond_idx: [B, CT]
        masked_vars: [B, T]
        """

        device = idx.device

        B, T = idx.size()
        CT = cond_idx.size(1)
        assert T == self.block_size

        H = W = int(math.sqrt(T))

        cond_pos = torch.arange(0, CT, device = device).unsqueeze(0)

        masked_idx = idx.clone()

        flags = torch.rand(masked_vars.size(), device = device)
        masked_idx.masked_fill_(masked_vars & (flags > 0.2), self.vocab_size)
        mask = masked_vars & (flags > 0.1) & (flags <= 0.2)
        masked_idx[mask] = torch.randint(0, self.vocab_size, (mask.sum(),), device = device)

        # forward the GPT model itself
        tok_emb = self.content_emb(masked_idx)
        x = self.transformer.drop(tok_emb)

        c_tok_emb = self.transformer.cond_wte(cond_idx)
        c_pos_emb = self.transformer.cond_wpe(cond_pos)
        c = self.transformer.drop(c_tok_emb + c_pos_emb)

        for i, block in enumerate(self.transformer.h):
            x = block(x, c)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        xids, yids = torch.where(masked_vars)

        target_idx = idx[xids,yids]
        pred_logits = logits[xids,yids,:]
        weights = (1.0 / masked_vars.sum(dim = 1, keepdim = True)).repeat(1, T)[xids,yids]

        if not self.new_mode:
            # loss
            ele_loss = F.cross_entropy(pred_logits, target_idx, reduction = "none")
            loss = (ele_loss * weights).sum() / B

            # per dim LL
            per_dim_ll = (F.log_softmax(pred_logits, dim = 1).gather(1, target_idx.unsqueeze(-1)).squeeze() * weights).sum() / B

            # accuracy
            acc = ((pred_logits.argmax(dim = 1) == target_idx).float() * weights).sum() / B * 100
        else:
            # loss
            loss = F.cross_entropy(pred_logits, target_idx)

            # per dim LL
            per_dim_ll = (F.log_softmax(pred_logits, dim = 1).gather(1, target_idx.unsqueeze(-1)).squeeze()).mean()

            # accuracy
            acc = ((pred_logits.argmax(dim = 1) == target_idx).float()).mean() * 100

        return logits, loss, per_dim_ll, acc

    def forward_fake_gpt(self, idx, cond_idx, masked_vars):
        device = idx.device

        B, T = idx.size()
        CT = cond_idx.size(1)
        assert T == self.block_size

        H = W = int(math.sqrt(T))

        cond_pos = torch.arange(0, CT, device = device).unsqueeze(0)

        masked_idx = idx.clone()

        # forward the GPT model itself
        tok_emb = self.content_emb(masked_idx)
        x = self.transformer.drop(tok_emb)

        c_tok_emb = self.transformer.cond_wte(cond_idx)
        c_pos_emb = self.transformer.cond_wpe(cond_pos)
        c = self.transformer.drop(c_tok_emb + c_pos_emb)
        x = torch.cat((c[:,0:1,:], x[:,:-1,:]), dim = 1)

        for i, block in enumerate(self.transformer.h):
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))

        per_dim_ll = F.log_softmax(logits, dim = 2).gather(2, idx.unsqueeze(-1)).mean()

        acc = ((logits.argmax(dim = 2) == idx).float()).mean() * 100

        return logits, loss, per_dim_ll, acc

    @torch.no_grad()
    def get_subset_embeddings(self, idx, cond_idx, target_vars):
        masked_vars = torch.zeros([idx.size(0), idx.size(1)], dtype = torch.bool, device = idx.device)
        masked_vars[:,target_vars] = True

        device = idx.device

        B, T = idx.size()
        CT = cond_idx.size(1)
        assert T == self.block_size

        cond_pos = torch.arange(0, CT, device = device).unsqueeze(0)

        masked_idx = idx.clone()

        flags = torch.rand(masked_vars.size(), device = device)

        masked_idx.masked_fill_(masked_vars, self.vocab_size)
        # masked_idx.masked_fill_(masked_vars & (flags > 0.2), self.vocab_size)
        # mask = masked_vars & (flags > 0.1) & (flags <= 0.2)
        # masked_idx[mask] = torch.randint(0, self.vocab_size, (mask.sum(),), device = device)

        # forward the GPT model itself
        tok_emb = self.content_emb(masked_idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)

        c_tok_emb = self.transformer.cond_wte(cond_idx)
        c_pos_emb = self.transformer.cond_wpe(cond_pos)
        c = self.transformer.drop(c_tok_emb + c_pos_emb)

        for i, block in enumerate(self.transformer.h):
            x = block(x, c)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        h = x[:,target_vars,:]

        target_idx = idx[masked_vars]
        pred_logits = logits[masked_vars,:]
        weights = (1.0 / masked_vars.sum(dim = 1, keepdim = True)).repeat(1, T)[masked_vars]

        # per dim LL
        per_dim_ll = (F.log_softmax(pred_logits, dim = 1).gather(1, target_idx.unsqueeze(-1)).squeeze() * weights).sum() / B

        return h, per_dim_ll

    def shared_step(self, batch, batch_idx):
        z = batch["z"]
        c = batch["c"]
        B = z.size(0)

        masked_vars = self.mask_generator(B, self.block_size, z.device)
        
        logits, loss, per_dim_ll, acc = self(z, c, masked_vars)

        return loss, per_dim_ll, acc

    def training_step(self, batch, batch_idx):
        loss, per_dim_ll, acc = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/per_dim_ll", per_dim_ll, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.last_tr_loss = loss.detach().cpu().item()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, per_dim_ll, acc = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("val/per_dim_ll", per_dim_ll, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("val/acc", acc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, per_dim_ll, acc = self.shared_step(batch, batch_idx)
        self.log("test/loss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/per_dim_ll", per_dim_ll, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/acc", acc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.last_tr_loss)