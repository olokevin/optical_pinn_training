import torch
import torch.nn as nn
import torch.optim as optim

from ZO_optimizer import ZO_SCD_mask, ZO_SGD_mask
from utils.common_utils import PresetLRScheduler

def build_optimizer(config, net, criterion, named_masks, learning_rate, weight_decay):
    if config.optimizer.name == 'ZO_SCD_mask':
        net.requires_grad_(False)
        optimizer = ZO_SCD_mask(
            model = net, 
            criterion = criterion,
            masks = named_masks,
            lr = learning_rate,
            grad_sparsity = config.optimizer.grad_sparsity,
            tensorized = config.model.tensorized
        )
        return optimizer
    elif config.optimizer.name == 'ZO_SGD_mask':
        if config.optimizer.debug == True:
            net.requires_grad_(True)
        else:
            net.requires_grad_(False)
        optimizer = ZO_SGD_mask(
            model = net, 
            criterion = criterion,
            masks = named_masks,
            lr = learning_rate,
            sigma = config.optimizer.sigma,
            n_sample  = config.optimizer.n_sample,
            signsgd = config.optimizer.signsgd,
            layer_by_layer = config.optimizer.layer_by_layer,
            tensorized = config.model.tensorized
        )
        return optimizer
    elif config.optimizer.name == 'ZO_mix':
        optimizer_SGD = ZO_SGD_mask(
            model = net, 
            criterion = criterion,
            masks = named_masks,
            lr = learning_rate,
            sigma = config.optimizer.sigma,
            n_sample  = config.optimizer.n_sample,
            signsgd = config.optimizer.signsgd,
            layer_by_layer = config.optimizer.layer_by_layer,
            tensorized = config.model.tensorized
        )
        optimizer_SCD = ZO_SCD_mask(
            model = net, 
            criterion = criterion,
            masks = named_masks,
            lr = config.optimizer.SCD_lr,
            grad_sparsity = config.optimizer.grad_sparsity,
            tensorized = config.model.tensorized
        )
        return optimizer_SGD, optimizer_SCD
    elif config.optimizer.name == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        return optimizer
    elif config.optimizer.name == 'ADAM':
        # optimizer = optim.Adam(list(net.parameters()), lr=learning_rate, weight_decay=weight_decay)
        optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.98), eps=1e-06, lr = learning_rate)
        return optimizer
    else:
        raise ValueError(f"Wrong optimizer_name {config.optimizer.name}") 

def build_scheduler(config, optimizers, learning_rate):
    if config.scheduler.name == 'PresetLRScheduler':
        lr_schedule = dict()
        for n_epoch,lr_coef in dict(config.scheduler.lr_schedule).items():
            lr_schedule[n_epoch] = learning_rate * lr_coef
    
        lr_scheduler = PresetLRScheduler(lr_schedule)
        return lr_scheduler
    elif config.scheduler.name == 'ExponentialLR':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizers, gamma=config.scheduler.gamma)
        return lr_scheduler
    elif config.scheduler.name == 'ZO_mix':
        lr_scheduler_SGD = optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=config.scheduler.gamma)
        lr_schedule = dict()
        for n_epoch,lr_coef in dict(config.scheduler.lr_schedule).items():
            lr_schedule[n_epoch] = config.optimizer.SCD_lr * lr_coef
    
        lr_scheduler_SCD = PresetLRScheduler(lr_schedule)

        return lr_scheduler_SGD, lr_scheduler_SCD

    else:
        raise ValueError(f"Wrong scheduler_name {config.scheduler.name}")