import os
import torch
import numpy as np

__all__ = [
    "load_model",
    "to_numpy",
    "to_device",
    "gradients",
]


def load_model(model, model_dir, epoch):
    checkpoint_dir = os.path.join(model_dir, 'checkpoints', 'model_epoch_{}.pth'.format(epoch))
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '
                        'np.ndarray, but got {}'.format(type(tensor)))


def to_device(data, device, requires_grad=False):
    return torch.tensor(data, requires_grad=requires_grad).float().to(device)

import torch.autograd as autograd
def gradients(y, x):
    return autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)