import torch
from torch import nn
import numpy as np
import math
from tltorch.factorized_tensors import FactorizedTensor, TensorizedTensor
from .tensorized_fwd_bwd import tensorized_linear_fwd_init
from .nl_init import tt_xavier_init, blocktt_xavier_init


# Author: Alvin Liu


class TensorizedLinear(nn.Module):

    def __init__(self, in_tensorized_features, out_tensorized_features, factorization='cp', rank='same',
                 bias=True, device=None, dtype=None):
        super(TensorizedLinear, self).__init__()

        factorization_list = ['cp', 'tt', 'blocktt']

        if not isinstance(factorization, str) or not (factorization in factorization_list):
            raise ValueError(f'Currently only accept factorization types={factorization_list} but got {factorization}.')

        self.factorization = factorization
        self.in_features = int(np.prod(in_tensorized_features))
        self.out_features = int(np.prod(out_tensorized_features))
        self.in_tensorized_features = in_tensorized_features
        self.out_tensorized_features = out_tensorized_features
        self.rank = rank

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, device=device, dtype=dtype))
            self.has_bias = True
        else:
            self.register_parameter('bias', None)

        if factorization == 'cp' or factorization == 'blocktt':
            tensor_shape = (in_tensorized_features, out_tensorized_features)
            self.ndim = len(in_tensorized_features)
            self.weight = TensorizedTensor.new(tensor_shape, rank=rank, factorization=factorization,
                                               device=device, dtype=dtype)
        elif factorization == 'tt':
            tensor_shape = (*in_tensorized_features, *out_tensorized_features)
            self.ndim = len(in_tensorized_features)
            self.weight = FactorizedTensor.new(tensor_shape, rank=rank, factorization=factorization,
                                               device=device, dtype=dtype)
        self.rank = self.weight.rank
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.factorization == 'cp':
                self.weight.normal_(0, math.sqrt(5) / math.sqrt(self.in_features))

            elif self.factorization == 'tt':
                std = math.sqrt(2 / (self.in_features + self.out_features))
                tt_xavier_init(self, std=std)

            elif self.factorization == 'blocktt':
                std = math.sqrt(5) / math.sqrt(self.in_features)
                blocktt_xavier_init(self, std=std)

            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.shape[1] != self.in_features:
            x = torch.cat( (x, torch.zeros(x.shape[0], (self.in_features-x.shape[1]), device=x.device) ), 1)
            # raise ValueError(f'Expected fan in {self.in_features} but got {x.shape} instead.')

        fwd_func = tensorized_linear_fwd_init(self.factorization)

        if self.factorization == 'blocktt':
            output = fwd_func(self.weight, x.T)
        elif self.factorization == 'tt':
            output = fwd_func(self.weight, x, row_d=len(self.in_tensorized_features))
        else:
            output = fwd_func(self.weight, x)

        if self.bias is not None:
            return torch.add(output, self.bias)
        else:
            return output

