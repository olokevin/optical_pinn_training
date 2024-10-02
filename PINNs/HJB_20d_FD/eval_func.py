import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import torch
import logging
from val_func import val_func_init_from_config_ADFD, val_func_init_from_config_stein,val_func_init_from_config_sparse_grids
from core.TFONet import int_or_float, to_numpy, load_model

# plt.rcParams.update(plt.rcParamsDefault)
# plt.rcParams.update({'font.size': 16,
#                      'lines.linewidth': 2,
#                      'axes.labelsize': 15,  # fontsize for x and y labels (was 10)
#                      'axes.titlesize': 15,
#                      'xtick.labelsize': 16,
#                      'ytick.labelsize': 16,
#                      'legend.fontsize': 16,
#                      'axes.linewidth': 2,
#                      'text.latex.preamble': [r'\usepackage{amsmath,amssymb,bm,physics,lmodern}'],
#                      "font.family": "serif"})


def eval_func_init_from_config(config, model_dir):
    eval_config = config['evaluation']
    plot = eval_config.getboolean('plot')

    if config.get('loss_func', 'gradient_func') == 'gradients_auto_diff' or config.get('loss_func', 'gradient_func') == 'gradients_cd2':
        val_func = val_func_init_from_config_ADFD(config) # use this when using auto diff or finite difference
    elif config.get('loss_func', 'gradient_func') == 'gradients_stein':
        val_func = val_func_init_from_config_stein(config) # use this when using stein
    elif config.get('loss_func', 'gradient_func') == 'gradients_sparse_grid':
        val_func = val_func_init_from_config_sparse_grids(config) #use this when using sparse grids
    else:
        raise ValueError('wrong gradient_func setting in .ini')



    training_epochs = config['training'].get('epochs')

    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    fh = logging.FileHandler('{}/eval_log.log'.format(model_dir))
    fh.setLevel(logging.CRITICAL)
    logger.addHandler(fh)

    def eval_func(model, dataset, eval_epoch):

        if not eval_epoch:
            eval_epoch = training_epochs

        model = load_model(model, model_dir, eval_epoch)
        model.eval()

        test_mse = val_func(model, dataset)
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.critical('eval epoch: {}, test loss: {}, # of parameters: {}'.
                        format(eval_epoch, test_mse, num_param))

    return eval_func
