import torch
import torch.nn as nn


class FourierMappingLayer(nn.Module):
    def __init__(self, in_features, out_features, freq=30, freq_trainable=False, device=None):
        super(FourierMappingLayer, self).__init__()
        self.freq = nn.Parameter(torch.ones(1, device=device) * freq, requires_grad=freq_trainable)
        self.fourier_features = nn.parameter.Parameter(torch.ones((in_features, out_features // 2),
                                                                  device=device).normal_() * self.freq,
                                                       requires_grad=False)

    def forward(self, x):
        return torch.cat((torch.sin(x @ self.fourier_features), torch.cos(x @ self.fourier_features)), -1)
