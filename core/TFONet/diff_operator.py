import torch
import torch.autograd as autograd


def gradients(y, x):
    return autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)
