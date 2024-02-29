import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Sequence, Dict, Optional


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


class SelfAttention(nn.Module):
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


class CrossAttention(nn.Module):
    """
    A vanilla multi-head masked cross-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd)
        self.c_attn_kv = nn.Linear(config.n_embd, 2 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x_q, x_kv, mask = None, get_attn_map = False):
        B, T_q, C = x_q.size() # batch size, sequence length, embedding dimensionality (n_embd)
        T_kv = x_kv.size(1)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.c_attn_q(x_q)
        k, v = self.c_attn_kv(x_kv).split(self.n_embd, dim = 2)
        q = q.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_q, hs)
        k = k.view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_kv, hs)
        v = v.view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_kv, hs)

        # self-attention; Self-attend: (B, nh, T_q, hs) x (B, nh, hs, T_kv) -> (B, nh, T_q, T_kv)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask is not None:
            if mask.dim() == 2:
                assert mask.size(0) == T and mask.size(1) == T
                att = att.masked_fill(mask[None,None,:,:] == 0, float('-inf'))
            else:
                raise NotImplementedError()

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T_q, T_kv) x (B, nh, T_kv, hs) -> (B, nh, T_q, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        if not get_attn_map:
            return y
        else:
            return y, att


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, causal = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        if causal:
            self.attn = CausalSelfAttention(config)
        else:
            self.attn = SelfAttention(config)
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


class CrossBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, causal = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        if causal:
            self.attn = CausalSelfAttention(config)
        else:
            self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        # self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x, h_enc, mask = None):
        x = x + self.attn(self.ln_1(x), mask = mask)
        x = x + self.cross_attn(self.ln_2(x), h_enc, mask = mask)
        # x = x + self.mlpf(self.ln_2(x))
        x = x + self.mlp.dropout(self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(self.ln_3(x)))))
        return x

