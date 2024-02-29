import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import pytorch_lightning as pl
import importlib
import itertools
from omegaconf import OmegaConf
from copy import deepcopy
from tqdm import tqdm
from argparse import Namespace

from .base import CrossBlock, Block
from .vq import VectorQuantizer


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


def random_bbox_generator(B, num_vars, device):
    H = W = int(math.sqrt(num_vars))

    max_pow = int(math.floor(math.log2(H)))

    all_bboxes = []
    for _ in range(B):
        bboxes = []
        curr_map = torch.zeros([H, W], dtype = torch.bool)
        while not curr_map.all():
            xids, yids = torch.where(~curr_map)
            i = np.random.randint(0, xids.size(0))
            xid, yid = xids[i], yids[i]

            powx = 2**np.random.randint(1, max_pow + 1)
            powy = 2**np.random.randint(1, max_pow + 1)

            hsid = min(xid, H - powx)
            wsid = min(yid, W - powy)
            heid = hsid + powx
            weid = wsid + powy

            curr_map[hsid:heid,wsid:weid] = True

            bboxes.append([hsid, heid-1, wsid, weid-1])

        all_bboxes.append(bboxes)

    max_num_bboxes = max([len(bboxes) for bboxes in all_bboxes])

    bboxes = torch.zeros([B, max_num_bboxes, 4], dtype = torch.long)
    bbox_mask = torch.zeros([B, max_num_bboxes], dtype = torch.bool)
    for i in range(B):
        curr_bboxes = torch.tensor(all_bboxes[i])
        bboxes[i,:curr_bboxes.size(0),:] = curr_bboxes
        bbox_mask[i,:curr_bboxes.size(0)] = True

    return bboxes.to(device), bbox_mask.to(device)


class SAE(pl.LightningModule):

    def __init__(self, n_enc_layer, n_dec_layer, n_head, n_embd,
                 vocab_size, block_size, embd_pdrop = 0.1, resid_pdrop = 0.1, 
                 attn_pdrop = 0.1, n_clusters = None,
                 bbox_generator = random_bbox_generator, content_emb = None, scheduler_config = None):
        super(SAE, self).__init__()

        self.scheduler_config = scheduler_config

        self.bbox_generator = bbox_generator

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        enc_config = Namespace(
            n_head = n_head, n_embd = n_embd, 
            resid_pdrop = resid_pdrop, attn_pdrop = attn_pdrop
        )
        dec_config = Namespace(
            n_head = n_head, n_embd = n_embd, 
            resid_pdrop = resid_pdrop, attn_pdrop = attn_pdrop
        )

        self.content_emb = instantiate_from_config(content_emb)

        H = W = int(math.sqrt(self.block_size))

        # Encoder
        self.enc_transformer = nn.ModuleDict(dict(
            hs_wpe = nn.Embedding(H, self.n_embd),
            he_wpe = nn.Embedding(H, self.n_embd),
            ws_wpe = nn.Embedding(W, self.n_embd),
            we_wpe = nn.Embedding(W, self.n_embd),
            drop = nn.Dropout(embd_pdrop),
            h = nn.ModuleList([Block(enc_config, causal = False) for _ in range(n_enc_layer)]),
            ln_f = nn.LayerNorm(self.n_embd)
        ))

        if n_clusters is not None and n_clusters > 0:
            self.vq = VectorQuantizer(n_clusters, self.n_embd, beta = 0.2, legacy = False)
        else:
            self.vq = None

        # Decoder
        self.dec_transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(self.block_size, self.n_embd),
            drop = nn.Dropout(embd_pdrop),
            h = nn.ModuleList([CrossBlock(dec_config, causal = False) for _ in range(n_dec_layer)]),
            ln_f = nn.LayerNorm(self.n_embd)
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias = False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(n_enc_layer + n_dec_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in itertools.chain(self.enc_transformer.parameters(), self.dec_transformer.parameters()))
        print("SAE number of parameters: %.2fM" % (n_params/1e6,))

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

    def forward(self, idx, bboxes, bbox_mask):
        """
        idx:       [B, T]
        bboxes:    [B, H, 4]
        bbox_mask: [B, H]
        """

        device = idx.device

        B, T = idx.size()
        H = bboxes.size(1)

        token_embds = self.content_emb(idx)

        query_embds = self.enc_transformer.hs_wpe(bboxes[:,:,0]) + self.enc_transformer.he_wpe(bboxes[:,:,1]) + \
            self.enc_transformer.ws_wpe(bboxes[:,:,2]) + self.enc_transformer.we_wpe(bboxes[:,:,3])

        enc_mask = torch.ones([H + T, H + T], dtype = torch.bool, device = device)
        enc_mask[:H,:H] = False
        enc_mask.view(-1)[::(H+T) + 1][:H] = True

        x = self.enc_transformer.drop(torch.cat((query_embds, token_embds), dim = 1))

        for block in self.enc_transformer.h:
            x = block(x, mask = enc_mask)

        z = self.enc_transformer.ln_f(x[:,:H,:])

        if self.vq is not None:
            z, code, vq_loss = self.vq(z)
        else:
            vq_loss = 0.0

        pos = torch.arange(0, T, device = device)[None,:].repeat(B, 1)
        dec_embds = self.dec_transformer.wpe(pos)

        h = w = int(math.sqrt(T))
        dec_mask = torch.ones([B, H, h, w], dtype = torch.bool, device = device)
        h_range = torch.arange(0, h, device = device)[None,None,:,None]
        w_range = torch.arange(0, w, device = device)[None,None,None,:]

        bboxes = bboxes[:,:,:,None,None]
        dec_mask &= (h_range >= bboxes[:,:,0,:,:])
        dec_mask &= (h_range <= bboxes[:,:,1,:,:])
        dec_mask &= (w_range >= bboxes[:,:,2,:,:])
        dec_mask &= (w_range <= bboxes[:,:,3,:,:])
        dec_mask &= bbox_mask[:,:,None,None]
        
        dec_mask = dec_mask.reshape(B, H, T).permute(0, 2, 1)

        x = self.dec_transformer.drop(dec_embds)

        z = z + query_embds
        for block in self.dec_transformer.h:
            x = block(x, z, xmask = dec_mask)

        x = self.dec_transformer.ln_f(x)
        logits = self.lm_head(x)

        # loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1)) + vq_loss

        # per dim LL
        ll = F.log_softmax(logits, dim = 2).gather(2, idx.unsqueeze(-1)).sum(dim = 1).mean()

        # accuracy
        acc = (logits.argmax(dim = 2) == idx).float().mean()

        return logits, loss, ll, acc

    def shared_step(self, batch, batch_idx):
        z = batch["z"]
        c = batch["c"]
        B = z.size(0)

        bboxes, bbox_mask = self.bbox_generator(B, self.block_size, z.device)
        
        logits, loss, ll, acc = self(z, bboxes, bbox_mask)

        return loss, ll, acc

    def training_step(self, batch, batch_idx):
        loss, ll, acc = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/ll", ll, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.last_tr_loss = loss.detach().cpu().item()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ll, acc = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("val/ll", ll, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("val/acc", acc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, ll, acc = self.shared_step(batch, batch_idx)
        self.log("test/loss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/ll", ll, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/acc", acc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.last_tr_loss)

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
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 1e-4},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        
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