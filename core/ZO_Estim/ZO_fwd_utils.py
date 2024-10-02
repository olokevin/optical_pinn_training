import torch

opt_able_layers_dict = {
    'Conv2d': torch.nn.Conv2d,
}

def get_iterable_block_name():
    return 'net'

def pre_block_forward(model, x):
    return x

def post_block_forward(model, x):
    return x