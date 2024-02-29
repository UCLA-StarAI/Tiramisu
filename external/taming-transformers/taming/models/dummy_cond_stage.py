from torch import Tensor
import torch.nn as nn


class DummyCondStage(nn.Module):
    def __init__(self, conditional_key):
        super().__init__()
        self.conditional_key = conditional_key
        self.train = None

    def forward(self, *args, call_func = None, **kwargs):
        if call_func == "encode":
            return self.encode(*args, **kwargs)
        elif call_func == "decode":
            return self.decode(*args, **kwargs)

    def eval(self):
        return self

    @staticmethod
    def encode(c: Tensor):
        return c, None, (None, None, c)

    @staticmethod
    def decode(c: Tensor):
        return c

    @staticmethod
    def to_rgb(c: Tensor):
        return c
