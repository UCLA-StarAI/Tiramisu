"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import top_k_top_p_filtering
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present   # TODO: check that this does not break anything

    def forward_kv(self, x, k_in, v_in, mask = None):
        B, T_x, C = x.size()
        T_in = k_in.size(1)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T_x, C) # (B, T, C)
        k_curr = self.key(x).view(B, T_x, C) # (B, T, C)
        v_curr = self.value(x).view(B, T_x, C) # (B, T, C)
        kk = k_curr.view(B, T_x, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_x, hs)
        q = q.view(B, T_x, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_x, hs)
        vv = v_curr.view(B, T_x, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_x, hs)

        kk_in = k_in.view(B, T_in, self.n_head, C // self.n_head).transpose(1, 2)
        vv_in = v_in.view(B, T_in, self.n_head, C // self.n_head).transpose(1, 2)
        k = torch.cat((kk_in, kk), dim = 2)
        v = torch.cat((vv_in, vv), dim = 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,T_in:T_in+T_x,:T_in+T_x] == 0, float('-inf'))
        if mask is not None:
            if mask.dim() == 2:
                assert mask.size(0) == T_x and mask.size(1) == T_in + T_x
                att = att.masked_fill(mask[None,None,:,:] == 0, float('-inf'))
            else:
                raise NotImplementedError()

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T_x, T) x (B, nh, T, hs) -> (B, nh, T_x, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_x, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        return y, k_curr, v_curr


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False, debug = False):
        # TODO: check that training still works
        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x

    def forward_kv(self, x, k_in, v_in, mask = None, debug = False):
        h, k_curr, v_curr = self.attn.forward_kv(self.ln1(x), k_in, v_in, mask = mask)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, k_curr, v_curr


class GPT(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        # input embedding stem
        self.n_embd = config.n_embd
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def get_subset_embeddings(self, x, cond_x, target_vars):
        orig_xidx = x
        idx = torch.cat((cond_x, x), dim = 1)[:,:-1]

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)[:,cond_x.size(1)-1:]

        x = x[:,cond_x.size(1)-1:,:]
        embds = x[:,target_vars,:]

        per_dim_ll = F.log_softmax(logits[:,target_vars,:], dim = 2).gather(2, orig_xidx[:,target_vars].unsqueeze(2)).mean(dim = 1).mean()

        return embds, per_dim_ll

    def get_zc(self, batch):
        z = batch["z"]
        c = batch["c"]
        return z, c

    def shared_step(self, batch, batch_idx):
        z, c = self.get_zc(batch)
        
        logits, _ = self(torch.cat((c, z), dim = 1)[:,:-1])
        logits = logits[:,c.size(1)-1:]

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z.reshape(-1))
        ll = F.log_softmax(logits, dim = 2).gather(2, z.unsqueeze(2)).sum(dim = 1).mean()

        return loss, ll

    def training_step(self, batch, batch_idx):
        loss, ll = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/ll", ll, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ll = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/ll", ll, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, ll = self.shared_step(batch, batch_idx)
        self.log("test/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("test/ll", ll, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def get_per_token_features(self, idx, embeddings=None, block_idxs=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)

        if isinstance(block_idxs, str):
            block_idxs = eval(block_idxs)
        B = idx.size(0)
        features = torch.zeros([B, t, self.n_embd * len(block_idxs)], device = idx.device)
        fsid = 0
        
        block_generator = self.blocks.children()
        for layer_id, block in enumerate(block_generator):
            x = block(x)
            if layer_id in block_idxs:
                feid = fsid + self.n_embd
                features[:,:,fsid:feid] = x
                fsid = feid

        return features

    def get_feature_dim(self, block_idxs=None):
        if isinstance(block_idxs, str):
            block_idxs = eval(block_idxs)

        return self.n_embd * len(block_idxs)

    def forward_with_past(self, idx, embeddings=None, targets=None, past=None, past_length=None):
        # inference only
        assert not self.training
        token_embeddings = self.tok_emb(idx)    # each index maps to a (learnable) vector
        if embeddings is not None:              # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)   # n_layer, 2, b, nh, len_past, dim_head
            past_shape = list(past.shape)
            expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head, past_length, self.config.n_embd//self.config.n_head]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[:, past_length, :]  # each position maps to a (learnable) vector
        else:
            position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        presents = []  # accumulate over layers
        for i, block in enumerate(self.blocks):
            x, present = block(x, layer_past=past[i, ...] if past is not None else None, return_present=True)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, torch.stack(presents)  # _, _, n_layer, 2, b, nh, 1, dim_head

    def forward_with_states(self, idx, k_in, v_in, mask = None, assemble_kv = True):
        assert not self.training

        device = idx.device
        t_kv = k_in[0].size(1)
        b, t_idx = idx.size()

        pos = torch.arange(t_kv, t_kv + t_idx, dtype=torch.long, device=device) # shape (t,)

        # forward the GPT model itself
        tok_emb = self.tok_emb(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.pos_emb[0,pos,:].unsqueeze(0) # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        for i, block in enumerate(self.blocks):

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

        x = self.ln_f(x)
        logits = self.head(x)

        return logits, k_in, v_in

    def get_initial_states(self, B, device):
        k_in = []
        v_in = []
        for _ in range(len(self.blocks)):
            k_in.append(torch.zeros([B, 0, self.n_embd], device = device))
            v_in.append(torch.zeros([B, 0, self.n_embd], device = device))

        return k_in, v_in

    @torch.no_grad()
    def sample(self, idx, max_new_tokens, mask = None, temperature=1.0, do_sample=False, top_k=None):
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

    @torch.no_grad()
    def conditional_sample(self, idx, idx_to_generate, temperature = 1.0, do_sample = False, top_k = None):
        k_in, v_in = self.get_initial_states(idx.size(0), idx.device)
        for i in range(idx.size(1)-1):
            # forward the model to get the logits for the index in the sequence
            logits, k_in, v_in = self.forward_with_states(idx[:,i].unsqueeze(1), k_in, v_in)
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
            if idx_to_generate[i]:
                idx[:,i+1] = idx_next.reshape(-1)

        return idx

    def configure_optimizers(self):
        """
        Following minGPT:
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

        if "pos_emb" in param_dict:
            # special case the position embedding parameter in the root GPT module as not decayed
            no_decay.add('pos_emb')
        elif "transformer.pos_emb" in param_dict:
            no_decay.add('transformer.pos_emb')

        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer


class DummyGPT(nn.Module):
    # for debugging
    def __init__(self, add_value=1):
        super().__init__()
        self.add_value = add_value

    def forward(self, idx):
        return idx + self.add_value, None


class CodeGPT(nn.Module):
    """Takes in semi-embeddings"""
    def __init__(self, vocab_size, block_size, in_channels, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        # input embedding stem
        self.tok_emb = nn.Linear(in_channels, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.taming_cinln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss



#### sampling utils

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


@torch.no_grad()
def sample_with_past(x, model, steps, temperature=1., sample_logits=True,
                     top_k=None, top_p=None, callback=None):
    # x is conditioning
    sample = x
    cond_len = x.shape[1]
    past = None
    for n in range(steps):
        if callback is not None:
            callback(n)
        logits, _, present = model.forward_with_past(x, past=past, past_length=(n+cond_len-1))
        if past is None:
            past = [present]
        else:
            past.append(present)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, dim=-1)
        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            x = torch.multinomial(probs, num_samples=1)
        # append to the sequence and continue
        sample = torch.cat((sample, x), dim=1)
    del past
    sample = sample[:, cond_len:]  # cut conditioning off
    return sample


#### clustering utils

class KMeans(nn.Module):
    def __init__(self, ncluster=512, nc=3, niter=10):
        super().__init__()
        self.ncluster = ncluster
        self.nc = nc
        self.niter = niter
        self.shape = (3,32,32)
        self.register_buffer("C", torch.zeros(self.ncluster,nc))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def is_initialized(self):
        return self.initialized.item() == 1

    @torch.no_grad()
    def initialize(self, x):
        N, D = x.shape
        assert D == self.nc, D
        c = x[torch.randperm(N)[:self.ncluster]] # init clusters at random
        for i in range(self.niter):
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a==k].mean(0) for k in range(self.ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            print('done step %d/%d, re-initialized %d dead clusters' % (i+1, self.niter, ndead))
            c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters

        self.C.copy_(c)
        self.initialized.fill_(1)


    def forward(self, x, reverse=False, shape=None):
        if not reverse:
            # flatten
            bs,c,h,w = x.shape
            assert c == self.nc
            x = x.reshape(bs,c,h*w,1)
            C = self.C.permute(1,0)
            C = C.reshape(1,c,1,self.ncluster)
            a = ((x-C)**2).sum(1).argmin(-1) # bs, h*w indices
            return a
        else:
            # flatten
            bs, HW = x.shape
            """
            c = self.C.reshape( 1, self.nc,  1, self.ncluster)
            c = c[bs*[0],:,:,:]
            c = c[:,:,HW*[0],:]
            x =      x.reshape(bs,       1, HW,             1)
            x = x[:,3*[0],:,:]
            x = torch.gather(c, dim=3, index=x)
            """
            x = self.C[x]
            x = x.permute(0,2,1)
            shape = shape if shape is not None else self.shape
            x = x.reshape(bs, *shape)

            return x
