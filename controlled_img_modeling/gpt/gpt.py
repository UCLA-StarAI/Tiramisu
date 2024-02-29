import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Sequence, Dict, Optional

import pytorch_lightning as pl

from argparse import Namespace


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, mask = None, get_attn_map = False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        if mask is not None:
            if mask.dim() == 2:
                assert mask.size(0) == T and mask.size(1) == T
                att = att.masked_fill(mask[None,None,:,:] == 0, float('-inf'))
            else:
                raise NotImplementedError()

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        if not get_attn_map:
            return y
        else:
            return y, att

    def forward_kv(self, x, k_in, v_in, mask = None):
        B, T_x, C = x.size()
        T_in = k_in.size(1)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k_curr, v_curr = self.c_attn(x).split(self.n_embd, dim=2)
        kk = k_curr.view(B, T_x, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_x, hs)
        q = q.view(B, T_x, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_x, hs)
        vv = v_curr.view(B, T_x, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_x, hs)

        kk_in = k_in.view(B, T_in, self.n_head, C // self.n_head).transpose(1, 2)
        vv_in = v_in.view(B, T_in, self.n_head, C // self.n_head).transpose(1, 2)
        k = torch.cat((kk_in, kk), dim = 2)
        v = torch.cat((vv_in, vv), dim = 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,T_in:T_in+T_x,:T_in+T_x] == 0, float('-inf'))
        if mask is not None:
            if mask.dim() == 2:
                assert mask.size(0) == T_x and mask.size(1) == T_in + T_x
                att = att.masked_fill(mask[None,None,:,:] == 0, float('-inf'))
            else:
                raise NotImplementedError()

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T_x, T) x (B, nh, T, hs) -> (B, nh, T_x, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_x, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, k_curr, v_curr


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        # self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x, mask = None):
        x = x + self.attn(self.ln_1(x), mask = mask)
        # x = x + self.mlpf(self.ln_2(x))
        x = x + self.mlp.dropout(self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(self.ln_2(x)))))
        return x

    def forward_kv(self, x, k_in, v_in, mask = None):
        h, k_curr, v_curr = self.attn.forward_kv(self.ln_1(x), k_in, v_in, mask = mask)
        x = x + h
        x = x + self.mlp.dropout(self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(self.ln_2(x)))))
        return h, k_curr, v_curr

    def get_attn_map(self, x, mask = None):
        h, attn_map = self.attn(self.ln_1(x), mask = mask, get_attn_map = True)
        # x = x + self.mlpf(self.ln_2(x + h))
        x = x + self.mlp.dropout(self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(self.ln_2(x + h)))))
        return x, attn_map


class GPT(pl.LightningModule):
    """ GPT Language Model """

    def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head, embd_pdrop, resid_pdrop, attn_pdrop):
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.block_size = block_size

        self.vocab_size = vocab_size
        self.n_embd = n_embd

        params_given = all([n_layer is not None, n_head is not None, n_embd is not None])

        config = Namespace()
        config.vocab_size = vocab_size
        config.block_size = block_size
        config.n_embd = n_embd
        config.n_head = n_head
        config.n_layer = n_layer
        config.embd_pdrop = embd_pdrop
        config.resid_pdrop = resid_pdrop
        config.attn_pdrop = attn_pdrop

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd)
        ))
        self.lm_head = nn.Linear(
            n_embd, 
            vocab_size, 
            bias = False
        )

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

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        train_config = Namespace()
        train_config.weight_decay = 1e-3
        train_config.learning_rate = self.learning_rate
        train_config.betas = (0.9, 0.95)

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
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets = None, mask = None, replace_features = None, move_features = None, 
                actual_func = None, first_token_embd = None, **kwargs):
        if actual_func is not None:
            if actual_func == "generate":
                return self.generate(idx, **kwargs)
            elif actual_func == "generate_fast":
                return self.generate_fast(idx, **kwargs)
            elif actual_func == "get_block_output":
                return self.get_block_output(idx, mask = mask, **kwargs)
            else:
                raise NotImplementedError()
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        additional_loss = 0.0

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        if first_token_embd is not None:
            tok_emb[:,0,:] = first_token_embd
        x = self.transformer.drop(tok_emb + pos_emb)
        for i, block in enumerate(self.transformer.h):

            if mask is None:
                x = block(x)
            elif mask.dim() == 3:
                x = block(x, mask = mask[i,:t,:t])
            elif mask.dim() == 2:
                x = block(x, mask = mask[:t,:t])
            else:
                raise NotImplementedError()

            if move_features is not None and f"{i}" in move_features:
                idxs_source = move_features[f"{i}"]["idxs_source"]
                idxs_target = move_features[f"{i}"]["idxs_target"]
                x2 = x.clone()
                x2[:,idxs_target,:] = x[:,idxs_source,:]
                x = x2

            if replace_features is not None and i in replace_features:
                idxs = replace_features[i]["indices"]
                features = replace_features[i]["features"]
                x = x.clone()
                x[:,idxs,:] = features

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = 0.0
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

        per_dim_ll = F.log_softmax(logits, dim = 2).gather(2, targets.unsqueeze(-1)).mean()

        acc = ((logits.argmax(dim = 2) == targets).float()).mean() * 100

        return logits, loss, per_dim_ll, acc

    def forward_with_states(self, idx, k_in, v_in, mask = None, assemble_kv = True):
        device = idx.device
        t_kv = k_in[0].size(1)
        b, t_idx = idx.size()

        pos = torch.arange(t_kv, t_kv + t_idx, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for i, block in enumerate(self.transformer.h):

            if mask is None:
                x, k_curr, v_curr = block.forward_kv(x, k_in[i], v_in[i])
            elif mask.dim() == 3:
                x, k_curr, v_curr = block.forward_kv(x, k_in[i], v_in[i], mask = mask[i,t_kv:t_kv+t_idx,:t_kv+t_idx])
            elif mask.dim() == 2:
                x, k_curr, v_curr = block.forward_kv(x, k_in[i], v_in[i], mask = mask[t_kv:t_kv+t_idx,:t_kv+t_idx])
            else:
                raise NotImplementedError()

            if assemble_kv:
                k_in[i] = torch.cat((k_in[i], k_curr), dim = 1)
                v_in[i] = torch.cat((v_in[i], v_curr), dim = 1)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits, k_in, v_in

    def forward_with_init_embds(self, idx, init_embds, targets = None, mask = None, 
                                replace_features = None, move_features = None):
        device = idx.device
        b, t = idx.size()
        t_init = init_embds.size(1)
        assert t_init + t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t_init + t, dtype = torch.long, device = device)

        additional_loss = 0.0

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        tok_emb = torch.cat((init_embds, tok_emb), dim = 1) # (b, t_init + t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for i, block in enumerate(self.transformer.h):

            if mask is None:
                x = block(x, mask = mask)
            elif mask.dim() == 3:
                x = block(x, mask = mask[i])
            elif mask.dim() == 2:
                x = block(x, mask = mask)
            else:
                raise NotImplementedError()

            if move_features is not None and f"{i}" in move_features:
                idxs_source = move_features[f"{i}"]["idxs_source"]
                idxs_target = move_features[f"{i}"]["idxs_target"]
                x2 = x.clone()
                x2[:,idxs_target,:] = x[:,idxs_source,:]
                x = x2

            if replace_features is not None and i in replace_features:
                idxs = replace_features[i]["indices"]
                features = replace_features[i]["features"]
                x[:,idxs,:] = features

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)[:,t_init-1:,:]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss + additional_loss

    def get_block_output(self, idx, block_idxs: Sequence, mask = None, get_attn_map = False,
                         replace_features = None, move_features = None, **kwargs):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        block_outputs = []
        if get_attn_map:
            block_attn_maps = []

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block_id, block in enumerate(self.transformer.h):
            if mask is None:
                curr_mask = None
            elif mask.dim() == 3:
                curr_mask = mask[block_id,:,:]
            elif mask.dim() == 2:
                curr_mask = mask
            else:
                raise NotImplementedError()
            
            if block_id in block_idxs:
                if get_attn_map:
                    x, attn_map = block.get_attn_map(x, mask = curr_mask)
                    block_attn_maps.append(attn_map)
                else:
                    x = block(x, mask = curr_mask)
                block_outputs.append(x)
            else:
                x = block(x, mask = curr_mask)

            if move_features is not None and f"{i}" in move_features:
                idxs_source = move_features[f"{i}"]["idxs_source"]
                idxs_target = move_features[f"{i}"]["idxs_target"]
                x2 = x.clone()
                x2[:,idxs_target,:] = x[:,idxs_source,:]
                x = x2

            if replace_features is not None and block_id in replace_features:
                idxs = replace_features[block_id]["indices"]
                features = replace_features[block_id]["features"]
                x[:,idxs,:] = features

        if not get_attn_map:
            return block_outputs
        else:
            return block_outputs, block_attn_maps

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, mask = None, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for i in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, mask = mask)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def get_initial_states(self, B, device):
        k_in = []
        v_in = []
        for _ in range(len(self.transformer.h)):
            k_in.append(torch.zeros([B, 0, self.n_embd], device = device))
            v_in.append(torch.zeros([B, 0, self.n_embd], device = device))

        return k_in, v_in

    @torch.no_grad()
    def generate_fast(self, idx, max_new_tokens, mask = None, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        k_in, v_in = self.get_initial_states(idx.size(0), idx.device)
        for i in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, k_in, v_in = self.forward_with_states(idx_cond[:,-1:], k_in, v_in, mask = mask)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def shared_step(self, batch, batch_idx):
        z = batch["z"]
        c = batch["c"]
        B = z.size(0)
        
        logits, loss, per_dim_ll, acc = self(torch.cat((c, z[:,:-1]), dim = 1), targets = z)

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