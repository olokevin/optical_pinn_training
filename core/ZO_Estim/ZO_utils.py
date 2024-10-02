import torch
import torch.nn as nn

class SplitedLayer(nn.Module):
    def __init__(self, idx, name, layer):
        super().__init__()
        self.idx = idx
        self.name = name
        self.layer = layer

class SplitedParam(nn.Module):
    def __init__(self, idx, name, layer, param, commit_fn=None):
        super().__init__()
        self.idx = idx
        self.name = name
        assert isinstance(param, torch.Tensor)
        self.layer = layer
        self.param = param
        if commit_fn is not None:
            assert callable(commit_fn)
            self.commit_fn = commit_fn

def split_model(model, iterable_block_name=None):
    modules = []
    # full model split
    if iterable_block_name is None:
        for m in model.children():
            if isinstance(m, (torch.nn.Sequential, torch.nn.ModuleList)):
                modules += split_model(m)
            else:
                modules.append(m)
    # only split iterable block
    else:
        iterable_block = getattr(model, iterable_block_name)
        assert isinstance(iterable_block, (torch.nn.Sequential, torch.nn.ModuleList))
        for m in iterable_block.children():
            modules.append(m)
    return modules

def split_named_model(model, parent_name=''):
    named_modules = {}
    for name, module in model.named_children():
    # for name, module in model.named_modules():    # Error: non-stop recursion
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
            named_modules.update(split_named_model(module, parent_name + name + '.'))
        # elif hasattr(module, 'conv') and isinstance(module.conv, torch.nn.Sequential):
        #     named_modules.update(split_named_model(module.conv, parent_name + name + '.conv.'))
        else:
            named_modules[parent_name + name] = module
    return named_modules