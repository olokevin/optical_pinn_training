from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import math
import numpy as np
import torch
from pyutils.general import logger
from torch import Tensor, nn
from torch.types import Device
from torch.nn import Parameter, init
import torch.nn.functional as F

from .layers.activation import ReLUN
from .layers.custom_linear import MZIBlockLinear
from .sparse_bp_base import SparseBP_Base

from core.tensor_layers.emb_utils import get_cum_prod,tensorized_lookup

# LowRankTensor
from abc import abstractmethod, ABC
import torch.distributions as td
Parameter = torch.nn.Parameter
from core.tensor_layers.truncated_normal import TruncatedNormal

__all__ = ["TTM_Linear_module", "TTM_LinearBlock","SparseBP_MZI_TTM_MLP"]

def manual_unflatten(tensor, dim, sizes):
    # Handle negative indexing
    if dim < 0:
        dim += tensor.dim()
    shape = list(tensor.shape)
    new_shape = shape[:dim] + list(sizes) + shape[dim+1:]
    return tensor.view(*new_shape)
    
########## For MNIST Training
########## 240919 Also for PINN training
class TTM_LinearBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        miniblock: int = 8,
        bias: bool = False,
        mode: str = "weight",
        v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
        v_pi: float = 4.36,
        w_bit: int = 16,
        in_bit: int = 16,
        photodetect: bool = False,
        device: Device = torch.device("cuda"),
        activation: bool = True,
        act_thres: int = 6,
        in_shape:  List = [2,2,2,2],
        out_shape: List = [2,2,2,2],
        tt_rank:   List = [1,4,4,4,1],
    ) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.miniblock = miniblock
        self.mode = mode
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.order = len(in_shape)

        self.activation = ReLUN(act_thres, inplace=True) if activation else None
        self.fast_forward_flag = False

        if bias:
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter("bias", None)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
            # init.uniform_(self.bias, 0, 0)
            fan_in = self.in_channel
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)


        if len(in_shape) != len(out_shape):
            raise (ValueError("the input dim and output dim should be the same"))
        
        if math.prod(in_shape) != in_channel or math.prod(out_shape) != out_channel:
            raise (ValueError("The decomposition shape does not match input/output channel"))
        
        if isinstance(tt_rank, int):
            self.tt_rank = [1]+(self.order-1)*[tt_rank]+[1]
        elif isinstance(tt_rank, list):
            if len(tt_rank) != self.order + 1:
                raise (ValueError("The rank list length does not match the decomposition dim"))
            else:
                self.tt_rank = tt_rank

        self.tt_cores = nn.ModuleList()

        for i in range(self.order):
            # tt_core[i]: r[i]*m[i]*n[i]*r[i+1]
            
            out_tt_core = self.tt_rank[i]  * out_shape[i]
            in_tt_core  = in_shape[i] * self.tt_rank[i+1]
            self.tt_cores.append(
                MZIBlockLinear(
                    in_tt_core, out_tt_core, miniblock, False, mode, v_max, v_pi, w_bit, in_bit, photodetect, device
                )
            )

        self.reset_tt_cores()
        self.gamma_noise_std = 0
        self.crosstalk_factor = 0
        
        print(f"TTM_LinearBlock: {in_channel} -> {out_channel}, in_shape: {in_shape}, out_shape: {out_shape}, tt_rank: {self.tt_rank}")

    def reset_tt_cores(self):
        std = math.sqrt(5) / math.sqrt(self.in_channel)
        r = np.prod(self.tt_rank)
        d = self.order

        std_factors = (std / r) ** (1 / d)
        for tt_core in self.tt_cores:
            tt_core.reset_parameters_ttm(std_factors)
            tt_core.sync_parameters(src=self.mode)

    def enable_fast_forward(self):
        self.fast_forward_flag = True
    
    def disable_fast_forward(self):
        self.fast_forward_flag = False
    
    def fast_forward(self, x: Tensor) -> Tensor:
        """TTM times matrix forward

        Parameters
        ----------
        tensor : BlockTT
            TTM tensorized weight matrix
        matrix : Parameter object
            input matrix x

        Returns
        -------
        output
            tensor times matrix
            equivalent to x@tensor.to_matrix()
        saved_tensors
            tensorized input matrix
        """
        
        # Prepare tensor shape
        shape_x = tuple(x.shape)
        # shape_x = matrix.shape
        tensorized_shape_x, tensorized_shape_y = tuple(self.in_shape), tuple(self.out_shape)

        num_batch = shape_x[0]
        order = self.order

        tt_rank = self.tt_rank

        # Reshape transpose of input matrix to input tensor
        input_tensor = torch.reshape(x, (shape_x[0:-1] + tensorized_shape_x))

        # Compute left partial sum
        # saved_tensor[k+1] = x.T * G1 * ... * Gk
        # saved_tensor[k+1].shape: num_batch * (i1*...*ik-1) * ik * (jk+1*...*jd) * rk
        saved_tensors = []
        current_i_dim = 1
        saved_tensors.append(
            input_tensor.reshape(shape_x[0:-1]+(current_i_dim, 1, -1, tt_rank[0]))
        )

        for k in range(order):
            current_tensor = saved_tensors[k]
            saved_tensors.append(
                # torch.einsum(
                #     '...ibcd,dbef->...iecf',
                #     current_tensor.reshape(shape_x[0:-1]+(current_i_dim, tensorized_shape_x[k], -1, tt_rank[k])), 
                #     self.tt_cores[k].get_2d_weight().reshape(tt_rank[k], tensorized_shape_y[k], tensorized_shape_x[k], tt_rank[k+1]).permute([0, 2, 1, 3]).contiguous()))
                torch.einsum(
                    '...ibcd,debf->...iecf',
                    current_tensor.reshape(shape_x[0:-1]+(current_i_dim, tensorized_shape_x[k], -1, tt_rank[k])), 
                    self.tt_cores[k].get_2d_weight().reshape(tt_rank[k], tensorized_shape_y[k], tensorized_shape_x[k], tt_rank[k+1])))
            current_i_dim *= tensorized_shape_y[k]

        # Forward Pass
        # y[i1,...,id] = sum_j1_..._jd G1[i1,j1] * G2[i2,j2] * ... * Gd[id,jd] * x[j1,...,jd]
        output = saved_tensors[order].reshape(shape_x[0:-1]+(-1,))
        # return output, saved_tensors
        return output
    
    def forward(self, x: Tensor) -> Tensor:
        if self.fast_forward_flag:
            output = self.fast_forward(x)
        else:
            x_shape = tuple(x.shape)
            new_shape = x_shape[0:-1] + tuple(self.in_shape)

            # x: decomposed as [bz,N_1,N_2,...,N_d]
            output = torch.reshape(x,new_shape).contiguous()
            i_tp = -3

            for i in reversed(range(self.order)):
                # [bz,N1,N2,...,Nd] -> [bzN1N2...Nd-1,Nd]
                temp_shape = output.shape[:-1]
                output = torch.flatten(output, 0, -2)
                # print('1',output.shape)
                # [bzN1N2...Nd-1,Nd] * [NdRd+1 * RdMd]
                # output = torch.matmul(output, self.tt_cores[i].T)
                output = self.tt_cores[i](output)
                # print('2',output.shape)
                # [bzN1N2...Nd-1,RdMd] -> [bz,N1,N2,...Nd-1,RdMd]
                # output = torch.unflatten(output, 0, temp_shape)
                output = manual_unflatten(output, 0, temp_shape)
                # [bz,N1,N2,...Nd-1,RdMd] -> [bz,N1,N2,...Nd-1,Rd,Md]
                # output = torch.unflatten(output, -1, (self.tt_rank[i],self.out_shape[i]))
                output = manual_unflatten(output, -1, (self.tt_rank[i],self.out_shape[i]))
                # print('3',output.shape)
                if i > 0:
                    # [bz,N1,N2,...Nd-1,Rd,Md] -> [bz,N1,N2,...,Md,Nd-1,Rd]
                    output = torch.transpose(torch.transpose(output,-1,-2), -2, i_tp)
                    # print('4',output.shape)
                    # [bz,N1,N2,...,Md,Nd-1,Rd] -> [bz,N1,N2,...,Md,Nd-1Rd]
                    output = torch.flatten(output, -2, -1)
                    # print('5',output.shape)
                    i_tp = i_tp - 1
                else:
                    output = torch.unsqueeze(output, i_tp)
                    output = torch.transpose(output, -1, i_tp).contiguous()

            # # permute batch_sz to the dim 0
            # output = torch.permute(output, (-1,)+tuple(range(self.order))).contiguous()
            # print('end',output.shape)
            output = torch.reshape(output, x_shape[0:-1]+(-1,)).contiguous()

            # fast_output = self.fast_forward(x)
            # print(torch.norm(output-fast_output))

        if self.bias is not None:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output

"""
    (TODO) orthogonal tensor train layer
"""

"""
    (deprecated) TTM MLP
"""

class SparseBP_MZI_TTM_MLP(SparseBP_Base):
    """MZI TTM MLP (Xian+, APL Photonics 2021). Support sparse backpropagation. Blocking matrix multiplication."""
    
    def __init__(
        self,
        n_feat: int,
        n_class: int,
        hidden_list: List[int] = [32],
        block_list: List[int] = [8],
        in_shape: List[int] = [4,7,7,4],
        hidden_shape_list: List[List[int]] = [[4,4,4,4],[4,4,4,4]],
        out_shape: List[int] = [1,5,2,1],
        max_rank_list: List[int] = [4],
        in_bit: int = 32,
        w_bit: int = 32,
        mode: str = "usv",
        v_max: float = 10.8,
        v_pi: float = 4.36,
        act_thres: float = 6.0,
        photodetect: bool = True,
        bias: bool = False,
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.hidden_list = hidden_list
        self.block_list = block_list

        self.in_shape = in_shape
        self.hidden_shape_list = hidden_shape_list
        self.out_shape = out_shape
        self.max_rank_list = max_rank_list

        self.in_bit = in_bit
        self.w_bit = w_bit
        self.mode = mode
        self.v_max = v_max
        self.v_pi = v_pi
        self.act_thres = act_thres

        self.photodetect = photodetect
        self.bias = bias

        self.device = device

        self.build_layers()
        self.drop_masks = None

        # self.reset_parameters()
        self.gamma_noise_std = 0
        self.crosstalk_factor = 0

    def build_layers(self) -> None:
        self.classifier = OrderedDict()

        '''input & hidden layers'''
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx + 1)
            in_channel = self.n_feat if idx == 0 else self.hidden_list[idx - 1]
            out_channel = hidden_dim
            in_shape = self.in_shape if idx == 0 else self.hidden_shape_list[idx-1]
            out_shape = self.hidden_shape_list[idx]

            # self.max_rank_list: store as a tuple, 1st element is the original list
            max_rank = self.max_rank_list[idx]
            if type(max_rank)==int:
                tt_rank = [1]+(len(in_shape)-1)*[max_rank]+[1]
            else:
                assert(type(max_rank)==list)
                tt_rank = max_rank

            self.classifier[layer_name] = TTM_LinearBlock(
                in_channel,
                out_channel,
                miniblock=self.block_list[idx],
                in_shape=in_shape,
                out_shape=out_shape,
                tt_rank=tt_rank,
                bias=self.bias,
                mode=self.mode,
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                photodetect=self.photodetect,
                device=self.device,
                activation=True,
                act_thres=self.act_thres,
            )

        '''output classifier'''
        layer_name = "fc" + str(len(self.hidden_list) + 1)
        max_rank = self.max_rank_list[-1]
        if type(max_rank)==int:
            tt_rank = [1]+(len(in_shape)-1)*[max_rank]+[1]
        else:
            assert(type(max_rank)==list)
            tt_rank = max_rank
        self.classifier[layer_name] = TTM_LinearBlock(
                self.hidden_list[-1] if len(self.hidden_list) > 0 else self.n_feat,
                self.n_class,
                miniblock=self.block_list[idx],
                in_shape=self.hidden_shape_list[-1],
                out_shape=self.out_shape,
                tt_rank=tt_rank,
                bias=self.bias,
                mode=self.mode,
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                photodetect=self.photodetect,
                device=self.device,
                activation=False,
                act_thres=self.act_thres,
            )
        self.classifier = nn.Sequential(self.classifier)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

########## For PINN training (deprecated)
class TTM_Linear_module(nn.Module):
    def __init__(
        self,
        # TensorizedLinear_module
        in_features: int,
        out_features: int,
        bias: bool = False,
        shape=[[2,2,2,2],[2,2,2,2]],
        tensor_type='TensorTrainMatrix',
        max_rank=4,
        em_stepsize=1.0,
        prior_type='log_uniform',
        eta = None,
        device=None,
        dtype=None,
        # L2ight
        miniblock: int = 8,
        # tt_rank:   List = [1,4,4,4,1],
        mode: str = "weight",
        v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
        v_pi: float = 4.36,
        w_bit: int = 16,
        in_bit: int = 16,
        photodetect: bool = False,
        activation: bool = True,
        act_thres: int = 6
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.miniblock = miniblock
        self.in_shape = shape[0]
        self.out_shape = shape[1]
        self.max_rank = max_rank
        self.order = len(self.in_shape)

        if len(self.in_shape) != len(self.out_shape):
            raise (ValueError("the input dim and output dim should be the same"))
        
        if math.prod(self.in_shape) != in_features or math.prod(self.out_shape) != out_features:
            raise (ValueError("The decomposition shape does not match input/output channel"))
        
        if isinstance(max_rank, int):
            self.tt_rank = [1]+(self.order-1)*[max_rank]+[1]
        else:
            if len(max_rank) != self.order + 1:
                raise (ValueError("The rank list length does not match the decomposition dim"))
            else:
                self.tt_rank = max_rank

        self.tt_cores = nn.ModuleList()
        # self.tt_cores = OrderedDict()

        for i in range(self.order):
            # tt_core[i]: r[i]*m[i]*n[i]*r[i+1]
            
            out_tt_core = self.tt_rank[i]  * self.out_shape[i]
            in_tt_core  = self.in_shape[i] * self.tt_rank[i+1]
            # self.tt_cores[str(i)] = MZIBlockLinear(
            #         in_tt_core, out_tt_core, miniblock, bias, mode, v_max, v_pi, w_bit, in_bit, photodetect, device
            #     )
            self.tt_cores.append(
                MZIBlockLinear(
                    in_tt_core, out_tt_core, miniblock, bias, mode, v_max, v_pi, w_bit, in_bit, photodetect, device
                )
            )
        # self.tt_cores = nn.Sequential(self.tt_cores)

        self.activation = ReLUN(act_thres, inplace=True) if activation else None

        self.reset_tt_cores()
        self.gamma_noise_std = 0
        self.crosstalk_factor = 0

    def reset_tt_cores(self):
        std = math.sqrt(5) / math.sqrt(self.in_features)
        r = np.prod(self.tt_rank)
        d = self.order

        std_factors = (std / r) ** (1 / d)
        for tt_core in self.tt_cores:
            tt_core.reset_parameters_ttm(std_factors)
        
    def extra_repr(self):
        model_str = f"MyModel(\n"
        
        for idx, tt_core in enumerate(self.tt_cores):
            model_str += f" tt_core{idx} shape: {tt_core.in_channel}, {tt_core.out_channel}\n"    
        
        model_str += ")"
        
        return model_str
    
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.in_features:
            x = torch.cat( (x, torch.zeros(x.shape[0], (self.in_features-x.shape[1]), device=x.device) ), 1)
            # raise ValueError(f'Expected fan in {self.in_features} but got {x.shape} instead.')
        
        x_shape = tuple(x.shape)
        batch_sz = x_shape[0]
        new_shape = x_shape[0:-1] + tuple(self.in_shape)

        # x: decomposed as [bz,N_1,N_2,...,N_d]
        output = torch.reshape(x,new_shape).contiguous()
        i_tp = -3

        for i in reversed(range(self.order)):
            # [bz,N1,N2,...,Nd] -> [bzN1N2...Nd-1,Nd]
            temp_shape = output.shape[:-1]
            output = torch.flatten(output, 0, -2)
            # print('1',output.shape)
            # [bzN1N2...Nd-1,Nd] * [NdRd+1 * RdMd]
            # output = torch.matmul(output, self.tt_cores[i].T)
            output = self.tt_cores[i](output)
            # print('2',output.shape)
            # [bzN1N2...Nd-1,RdMd] -> [bz,N1,N2,...Nd-1,Rd,Md]
            output = torch.unflatten(output, 0, temp_shape)
            output = torch.unflatten(output, -1, (self.tt_rank[i],self.out_shape[i]))
            # print('3',output.shape)
            # [bz,N1,N2,...Nd-1,Rd,Md] -> [bz,N1,N2,...,Md,Nd-1,Rd]
            output = torch.transpose(torch.transpose(output,-1,-2), -2, i_tp)
            # print('4',output.shape)
            # [bz,N1,N2,...,Md,Nd-1,Rd] -> [bz,N1,N2,...,Md,Nd-1Rd]
            output = torch.flatten(output, -2, -1)
            # print('5',output.shape)
            i_tp = i_tp - 1

        # permute batch_sz to the dim 0
        output = torch.permute(output, (-1,)+tuple(range(self.order))).contiguous()
        # print('end',output.shape)
        output = torch.reshape(output, (batch_sz, -1))

        if self.activation is not None:
            output = self.activation(output)
        return output