import torch
import torch.nn as nn
import numpy as np
from tltorch.factorized_tensors import TTTensor, BlockTT


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def xavier_init(layer):
    with torch.no_grad():
        if type(layer) == nn.Linear:
            if hasattr(layer, 'weight'):
                nn.init.xavier_normal_(layer.weight)
        else:
            raise TypeError(f'Expecting nn.Linear got type={type(layer)} instead')


def tt_xavier_init(layer, std=1.0):
    if not isinstance(layer.weight, TTTensor):
        raise TypeError(
            f'Expecting the given layer has weight of type TTTensor, got type={type(layer.weight)} instead.')

    r = np.sqrt(np.prod(layer.weight.rank))
    std_factors = (std / r) ** (1 / layer.weight.order)

    for factor in layer.weight.factors:
        factor.data.normal_(0, std_factors)


def blocktt_xavier_init(layer, std=1.0):
    if not isinstance(layer.weight, BlockTT):
        raise TypeError(
            f'Expecting the given layer has weight of type BlockTT, got type={type(layer.weight)} instead.')

    r = np.prod(layer.weight.rank)
    d = len(layer.weight.factors)
    std_factors = (std / r) ** (1 / d)

    for factor in layer.weight.factors:
        factor.data.normal_(0, std_factors)


nl_init_dict = dict(
    sine=(Sine(), xavier_init),
    silu=(nn.SiLU(), xavier_init),
    tanh=(nn.Tanh(), xavier_init),
    relu=(nn.ReLU(), xavier_init),
    gelu=(nn.GELU(), xavier_init)
)
