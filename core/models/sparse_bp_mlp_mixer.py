"""
Description:
Author:
Date: 2021-10-24 16:24:50
LastEditors:
LastEditTime: 2021-10-24 16:24:50
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pyutils.general import logger
from torch import Tensor, nn
from torch.nn.modules.activation import ReLU
from torch.types import Device, _size

from .layers.activation import ReLUN
from .layers.custom_linear import MZIBlockLinear
from .layers.custom_conv2d import MZIBlockConv2d
from .sparse_bp_base import SparseBP_Base
from.sparse_bp_ttm_mlp import TTM_LinearBlock

from einops.layers.torch import Rearrange

__all__ = ["SparseBP_MZI_MLPMixer", "SparseBP_MZI_MLPMixer_B16_224", "SparseBP_MZI_MLPMixer_L16_224"]

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,

        miniblock: int = 8,
        bias: bool = False,
        mode: str = "weight",
        v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
        v_pi: float = 4.36,
        w_bit: int = 16,
        in_bit: int = 16,
        photodetect: bool = False,
        activation: bool = True,
        act_thres: int = 6,
        device: Device = torch.device("cuda"),
        tensorize_config: Dict = None,  
    ) -> None:
        super().__init__()
        if tensorize_config is not None:
            self.fc1 = TTM_LinearBlock(dim, hidden_dim, miniblock, bias, mode, v_max, v_pi, w_bit, in_bit, photodetect, device=device, activation=False, in_shape=tensorize_config.shape[0], out_shape=tensorize_config.shape[1], tt_rank=tensorize_config.tt_rank)
            self.fc2 = TTM_LinearBlock(hidden_dim, dim, miniblock, bias, mode, v_max, v_pi, w_bit, in_bit, photodetect, device=device, activation=False, in_shape=tensorize_config.shape[1], out_shape=tensorize_config.shape[0], tt_rank=tensorize_config.tt_rank)
        else:
            self.fc1 = MZIBlockLinear(dim, hidden_dim, miniblock, bias, mode, v_max, v_pi, w_bit, in_bit, photodetect, device)
            self.fc2 = MZIBlockLinear(hidden_dim, dim, miniblock, bias, mode, v_max, v_pi, w_bit, in_bit, photodetect, device)
        self.net = nn.Sequential(
            self.fc1,
            nn.GELU(),
            nn.Dropout(dropout),
            self.fc2,
            nn.Dropout(dropout),
        )
        self.gamma_noise_std = 0
        self.crosstalk_factor = 0

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.,
                miniblock: int = 8,
                bias: bool = False,
                mode: str = "weight",
                v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
                v_pi: float = 4.36,
                w_bit: int = 16,
                in_bit: int = 16,
                photodetect: bool = False,
                activation: bool = True,
                act_thres: int = 6,
                device: Device = torch.device("cuda"),
                tensorize_config: Dict = None,
        ):
        super().__init__()

        mlp_tokens_tensorize_config=None
        mlp_channels_tensorize_config=None
        
        if tensorize_config is not None:
            mlp_tokens_tensorize_config = tensorize_config.get("mlp_tokens", None)
            mlp_channels_tensorize_config = tensorize_config.get("mlp_channels", None)
    
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_tokens = FeedForward(num_patch, token_dim, dropout, miniblock, bias, mode, v_max, v_pi, w_bit, in_bit, photodetect, activation, act_thres, device, tensorize_config=mlp_tokens_tensorize_config)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_channels = FeedForward(dim, channel_dim, dropout, miniblock, bias, mode, v_max, v_pi, w_bit, in_bit, photodetect, activation, act_thres, device, tensorize_config=mlp_channels_tensorize_config)
    
    def forward(self, x):
        x = x + Rearrange('b d n -> b n d')(self.mlp_tokens(Rearrange('b n d -> b d n')(self.norm1(x))))
        x = x + self.mlp_channels(self.norm2(x))

        return x

class SparseBP_MZI_MLPMixer(SparseBP_Base):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim, dropout,
                miniblock: int = 8,
                bias: bool = False,
                mode: str = "weight",
                v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
                v_pi: float = 4.36,
                w_bit: int = 16,
                in_bit: int = 16,
                photodetect: bool = False,
                activation: bool = True,
                act_thres: int = 6,
                device: Device = torch.device("cuda"),
                tensorize_config: Dict = None,
        ):
        super().__init__()

        self.dim = dim
        self.bias = bias

        self.miniblock = miniblock
        self.mode = mode
        self.v_max = v_max
        self.v_pi = v_pi
        self.w_bit = w_bit
        self.in_bit = in_bit

        self.photodetect = photodetect
        self.activation = activation
        self.act_thres = act_thres
        self.device = device

        self.tensorize_config = tensorize_config

        self.total_num_layers = depth + 2

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            # MZIBlockConv2d(
            #     in_channel=in_channels,
            #     out_channel=dim,
            #     kernel_size=patch_size,
            #     miniblock=miniblock,
            #     bias=True,
            #     stride=patch_size,
            #     padding=0,
            #     mode=mode,
            #     v_max=v_max,
            #     v_pi=v_pi,
            #     w_bit=w_bit,
            #     in_bit=in_bit,
            #     photodetect=photodetect,
            #     device=device,
            # ),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(
                MixerBlock(
                    dim, self.num_patch, token_dim, channel_dim, dropout,
                    miniblock, bias, mode, v_max, v_pi, w_bit, in_bit, photodetect, activation, act_thres, device,
                    tensorize_config=self.tensorize_config
                )
            )

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.head_drop = nn.Dropout(dropout)

        if self.tensorize_config is not None and hasattr(self.tensorize_config, "head"):
            head_tensorize_config = self.tensorize_config.head
            self.head = TTM_LinearBlock(dim, num_classes, miniblock, bias, mode, v_max, v_pi, w_bit, in_bit, photodetect, device=device, activation=False, act_thres=act_thres, in_shape=head_tensorize_config.shape[0], out_shape=head_tensorize_config.shape[1], tt_rank=head_tensorize_config.tt_rank)
        else:
            self.head = MZIBlockLinear(dim, num_classes, miniblock, bias, mode, v_max, v_pi, w_bit, in_bit, photodetect, device)

    def replace_head(self, num_classes):
        self.head = MZIBlockLinear(self.dim, num_classes, self.head.miniblock, self.bias, self.head.mode, self.head.v_max, self.head.v_pi, self.head.w_bit, self.head.in_bit, self.head.photodetect, self.head.device)
    
    def enable_fast_forward(self):
        for module in self.modules():
            if type(module) == TTM_LinearBlock:
                assert module.mode == "weight"
                module.enable_fast_forward()
    
    def disable_fast_forward(self):
        for module in self.modules():
            if type(module) == TTM_LinearBlock:
                assert module.mode == "weight"
                module.disable_fast_forward()
    
    def forward(self, x):
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        x = x.mean(dim=1)

        x = self.head_drop(x)

        return self.head(x)

    def partial_forward(self, x, num_layers):
        
        layer_outputs = []
        i_layer = 0

        x = self.stem(x)
        i_layer += 1
        layer_outputs.append(x)

        if i_layer == num_layers:
            return layer_outputs, x

        for block in self.blocks:
            x = block(x)
            i_layer += 1
            layer_outputs.append(x)
            if i_layer == num_layers:
                return layer_outputs, x
        
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head_drop(x)
        x = self.head(x)

        return layer_outputs, x

def SparseBP_MZI_MLPMixer_B16_224(num_classes=1000, *args, **kwargs):
    return SparseBP_MZI_MLPMixer(in_channels=3, image_size=224, patch_size=16, num_classes=num_classes,
                 dim=768, depth=12, token_dim=384, channel_dim=3072, dropout=0.0, *args, **kwargs)

def SparseBP_MZI_MLPMixer_L16_224(num_classes=1000, *args, **kwargs):
    return SparseBP_MZI_MLPMixer(in_channels=3, image_size=224, patch_size=16, num_classes=num_classes,
                 dim=1024, depth=24, token_dim=512, channel_dim=4096, dropout=0.0, *args, **kwargs)