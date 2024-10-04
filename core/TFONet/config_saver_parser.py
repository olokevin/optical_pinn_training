import os
import shutil
import ast
import configparser
import time
import torch
import sys
import logging
from torch.utils.tensorboard import SummaryWriter
from pyutils.config import configs as yml_config
from core.TFONet import train
from core.TFONet.modules import FCBlock, TensorizedFCBlock, ONN_FCBlock, TensorizedONN_FCBlock
from core.optimizer import FLOPSOptimizer, MixedTrainOptimizer, ScheduledOptim, ZO_SGD_mask, ZO_SCD_mask

from core.GraSP.model_base import ModelBase
from core.GraSP.utils.init_utils import weights_init
from core.GraSP.GraSP_pinn import GraSP_pinn
from core.GraSP.utils.common_utils import PresetLRScheduler
# from pyutils.torch_train import (
#     get_learning_rate,
#     get_random_state,
#     set_torch_deterministic,
#     set_torch_stochastic,
# )
import random
import numpy as np

from core.ZO_Estim.ZO_Estim_entry import build_ZO_Estim

def set_torch_deterministic(random_state: int = 0) -> None:
    random_state = int(random_state) % (2 ** 32)
    # random_state = int(random_state)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(random_state)
    random.seed(random_state)


def load_config_ini(ini_file_path):
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(ini_file_path)
    return config


def save_config_to_ini(config, model_dir):
    if isinstance(config, dict):
        parser = configparser.ConfigParser(allow_no_value=True)
        for key, value in config.items():
            parser[key] = value
    elif isinstance(config, configparser.ConfigParser):
        parser = config
    else:
        raise TypeError(f'Expected ConfigParser or Dict, got type={type(config)}')
    with open('{}/config.ini'.format(model_dir), 'w') as configfile:
        parser.write(configfile)


def int_or_float(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def convert_value(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.lower() == 'null':
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    return value

def dnn_model_parser(model_config, device):
    in_features = model_config.getint('in_features')
    out_features = model_config.getint('out_features')
    hidden_features = model_config.getint('hidden_features')
    num_layers = model_config.getint('num_layers')
    nonlinearity = model_config.get('nonlinearity')
    nl_last_layer = model_config.getboolean('nl_last_layer')

    fourier_mapping = model_config.getboolean('fourier_mapping')
    if fourier_mapping:
        freq = model_config.getint('freq')
        freq_trainable = model_config.getboolean('freq_trainable')
    else:
        freq = None
        freq_trainable = None

    bias = model_config.getboolean('bias')
    dtype = model_config.get('dtype')

    model = FCBlock(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                    num_layers=num_layers, nonlinearity=nonlinearity, nl_last_layer=nl_last_layer,
                    fourier_mapping=fourier_mapping, freq=freq, freq_trainable=freq_trainable,
                    bias=bias, device=device, dtype=dtype)

    return model

def tensorized_dnn_model_parser(model_config, device):
    in_features = model_config.getint('in_features')
    out_features = model_config.getint('out_features')
    hidden_features = model_config.getint('hidden_features')
    num_layers = model_config.getint('num_layers')
    nonlinearity = model_config.get('nonlinearity')
    order = model_config.getint('order')
    min_dim = model_config.getint('min_dim')
    factorization = model_config.get('factorization')

    rank = model_config.get('rank')
    if rank is not None and rank != 'same':
        rank = int_or_float(rank)
    
    raw_shape_list = model_config.get('shape_list') if not None else None
    if raw_shape_list is not None:
        shape_list = ast.literal_eval(raw_shape_list)
    else:
        shape_list = None

    tensorize_first = model_config.getboolean('tensorize_first')
    tensorize_last = model_config.getboolean('tensorize_last')
    bias = model_config.getboolean('bias')
    dtype = model_config.get('dtype')

    model = TensorizedFCBlock(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                              num_layers=num_layers, nonlinearity=nonlinearity, shape_list=shape_list, order=order, 
                              min_dim=min_dim, factorization=factorization, rank=rank, tensorize_first=tensorize_first,
                              tensorize_last=tensorize_last, bias=bias, device=device, dtype=dtype)

    return model

def onn_model_parser(model_config, device):
    in_features = model_config.getint('in_features')
    out_features = model_config.getint('out_features')
    hidden_features = model_config.getint('hidden_features')
    num_layers = model_config.getint('num_layers')
    nonlinearity = model_config.get('nonlinearity')

    bias = model_config.getboolean('bias')
    dtype = model_config.get('dtype')

    in_bit=yml_config.quantize.input_bit
    w_bit=yml_config.quantize.weight_bit
    mode=yml_config.model.mode
    v_max=yml_config.quantize.v_max
    v_pi=yml_config.quantize.v_pi
    act_thres=yml_config.model.act_thres

    model = ONN_FCBlock(
        in_features=in_features, 
        out_features=out_features, 
        hidden_features=hidden_features,
        num_layers=num_layers, 
        nonlinearity=nonlinearity, 
        bias=bias, 
        device=device, 
        dtype=dtype,
        in_bit=in_bit,
        w_bit=w_bit,
        mode=mode,
        v_max=v_max,
        v_pi=v_pi,
        act_thres=act_thres,
        photodetect=False
    )

    return model

##### MLNCP #####

# def tonn_model_parser(model_config, device):
#     in_features = model_config.getint('in_features')
#     out_features = model_config.getint('out_features')
#     hidden_features = model_config.getint('hidden_features')
#     num_layers = model_config.getint('num_layers')
#     nonlinearity = model_config.get('nonlinearity')
#     order = model_config.getint('order')
#     min_dim = model_config.getint('min_dim')
#     factorization = model_config.get('factorization')

#     # rank = model_config.get('rank')
#     # if rank is not None and rank != 'same':
#     #     rank = int_or_float(rank)

#     tensorize_first = model_config.getboolean('tensorize_first')
#     tensorize_last = model_config.getboolean('tensorize_last')
#     bias = model_config.getboolean('bias')
#     dtype = model_config.get('dtype')

#     max_rank_list=yml_config.model.max_rank_list
#     in_bit=yml_config.quantize.input_bit
#     w_bit=yml_config.quantize.weight_bit
#     mode=yml_config.model.mode
#     v_max=yml_config.quantize.v_max
#     v_pi=yml_config.quantize.v_pi
#     act_thres=yml_config.model.act_thres

#     model = TensorizedONN_FCBlock(
#         in_features=in_features, 
#         out_features=out_features, 
#         hidden_features=hidden_features,
#         num_layers=num_layers, 
#         nonlinearity=nonlinearity, 
#         order=order, 
#         min_dim=min_dim,
#         factorization=factorization, 
#         # rank=rank, 
#         rank=max_rank_list,
#         tensorize_first=tensorize_first,
#         tensorize_last=tensorize_last, 
#         bias=bias, 
#         device=device, 
#         dtype=dtype,
#         in_bit=in_bit,
#         w_bit=w_bit,
#         mode=mode,
#         v_max=v_max,
#         v_pi=v_pi,
#         act_thres=act_thres,
#         photodetect=False,
#         yml_config=yml_config
#     )

#     return model

def tonn_model_parser(model_config, device):
    in_features = model_config.getint('in_features')
    out_features = model_config.getint('out_features')
    hidden_features = model_config.getint('hidden_features')
    num_layers = model_config.getint('num_layers')
    nonlinearity = model_config.get('nonlinearity')
    order = model_config.getint('order')
    min_dim = model_config.getint('min_dim')
    factorization = model_config.get('factorization')
    
    in_features = model_config.getint('in_features')
    out_features = model_config.getint('out_features')
    hidden_features = model_config.getint('hidden_features')
    num_layers = model_config.getint('num_layers')
    nonlinearity = model_config.get('nonlinearity')
    order = model_config.getint('order')
    min_dim = model_config.getint('min_dim')
    factorization = model_config.get('factorization')

    rank = model_config.get('rank')
    if rank is not None and rank != 'same':
        rank = convert_value(model_config.get('rank'))
        # rank = int_or_float(rank)
    
    raw_shape_list = model_config.get('shape_list') if not None else None
    if raw_shape_list is not None:
        shape_list = ast.literal_eval(raw_shape_list)
    else:
        shape_list = None

    tensorize_first = model_config.getboolean('tensorize_first')
    tensorize_last = model_config.getboolean('tensorize_last')
    bias = model_config.getboolean('bias')
    dtype = model_config.get('dtype')
    
    # mode = model_config.get('mode')
    # act_thres = model_config.getfloat('act_thres')
    
    mode=yml_config.model.mode
    act_thres=yml_config.model.act_thres

    in_bit=yml_config.quantize.input_bit
    w_bit=yml_config.quantize.weight_bit
    v_max=yml_config.quantize.v_max
    v_pi=yml_config.quantize.v_pi
    
    

    model = TensorizedONN_FCBlock(
        in_features=in_features, 
        out_features=out_features, 
        hidden_features=hidden_features,
        num_layers=num_layers, 
        nonlinearity=nonlinearity, 
        shape_list=shape_list,
        order=order, 
        min_dim=min_dim,
        factorization=factorization, 
        rank=rank, 
        tensorize_first=tensorize_first,
        tensorize_last=tensorize_last, 
        bias=bias, 
        device=device, 
        dtype=dtype,
        
        in_bit=in_bit,
        w_bit=w_bit,
        mode=mode,
        v_max=v_max,
        v_pi=v_pi,
        act_thres=act_thres,
        photodetect=False,
    )

    return model

def mrr_model_parser(model_config, device):
    in_features = model_config.getint('in_features')
    out_features = model_config.getint('out_features')
    hidden_features = model_config.getint('hidden_features')
    num_layers = model_config.getint('num_layers')
    nonlinearity = model_config.get('nonlinearity')

    bias = model_config.getboolean('bias')
    in_bit = model_config.getint('in_bit')
    dtype = model_config.get('dtype')
    
    mode=model_config.get('mode')
    mrr_config_type=model_config.get('mrr_config_type')
    
    model = MRR_FCBlock(
        in_features=in_features, 
        out_features=out_features, 
        hidden_features=hidden_features,
        num_layers=num_layers, 
        nonlinearity=nonlinearity, 
        bias=bias, 
        in_bit=in_bit,
        mode=mode,
        mrr_config_type=mrr_config_type,
        device=device, 
    )

    return model

def build_model_from_config(config, device):
    model_config = config['model']
    model_type = model_config['type']

    model_type_list = ['dnn', 'tensorized_dnn', 'onn', 'tonn', 'mrr']
    if model_type not in model_type_list:
        raise TypeError(f'Supported model types are {model_type_list}, got {model_type} instead')

    parser_dict = dict(dnn=dnn_model_parser,
                       tensorized_dnn=tensorized_dnn_model_parser,
                       onn=onn_model_parser,
                       tonn=tonn_model_parser,
                       mrr=mrr_model_parser
                       )

    return parser_dict[model_type](model_config, device)

def print_mask_information(mb):
    ratios = mb.get_ratio_at_each_layer()
    print('** Mask information of %s. Overall Remaining: %.2f%%' % (mb.get_name(), ratios['ratio']))
    count = 0
    for k, v in ratios.items():
        if k == 'ratio':
            continue
        if isinstance(v, dict):
            for idx, v_ratio in enumerate(v.values()):
                print('  (%d) tt-core %d: Remaining: %.2f%%' % (count, idx, v_ratio))
        else:
            v_ratio = v
            print('  (%d) %s: Remaining: %.2f%%' % (count, k, v_ratio))
        count += 1

def GraSP_from_ini(config, model, dataset, loss_fn, device):
    num_iterations = config['GraSP'].getint('iterations')
    target_ratio = config['GraSP'].getfloat('target_ratio')
    normalize = config['GraSP'].getboolean('normalize')
    ratio = 1 - (1 - target_ratio) ** (1.0 / num_iterations)
    
    # build ModelBase 
    mb = ModelBase('tt-pinn', '3', 'helmholtz', model)
    mb.cuda()

    # ================== start pruning ==================  
    print('** Target ratio: %.4f, iter ratio: %.4f, iteration: %d/%d.' % (target_ratio,
                                                                                ratio,
                                                                                1,
                                                                                num_iterations))

    
    if config['pretrained'].getboolean('incre') == True:
        pass
    else:
        mb.model.apply(weights_init)
    
    iteration = 0
    print("Iteration of: %d/%d" % (iteration, num_iterations))

    masks, named_masks = GraSP_pinn(
        mb.model, ratio, dataset, loss_fn, device,
        num_iters=config['GraSP'].getint('num_iters'),
        T=config['GraSP'].getint('T'),
        tensorized=False
    )

    print('=> Using GraSP')
    # ========== register mask ==================
    # pretrained model, do not register forward hook (remain original value)
    if config['pretrained'].getboolean('incre') == True:
        mb.register_mask(masks, forward_hook=False)
    # from scratch, register forward hook (keep pruned params as 0)
    else:
        mb.register_mask(masks, forward_hook=True)

    # ========== print pruning details ============
    print('**[%d] Mask and training setting: ' % iteration)
    ratios = mb.get_ratio_at_each_layer()
    print('** Mask information of %s. Overall Remaining: %.2f%%' % (mb.get_name(), ratios['ratio']))
    count = 0
    for k, v in ratios.items():
        if k == 'ratio':
            continue
        if isinstance(v, dict):
            for idx, v_ratio in enumerate(v.values()):
                print('  (%d) tt-core %d: Remaining: %.2f%%' % (count, idx, v_ratio))
        else:
            v_ratio = v
            print('  (%d) %s: Remaining: %.2f%%' % (count, k, v_ratio))
        count += 1
    
    return masks, named_masks

def def_optim_from_config(config, model, loss_fn, named_masks=None):
    training_config = config['training']
    optimizer = training_config.get('optimizer') if not None else 'adam'
    scheduler = training_config.get('scheduler')
    lr = training_config.getfloat('lr')

    if optimizer == 'adam':
        optim = torch.optim.Adam(lr=lr, params=model.parameters())
    elif optimizer == 'sgd':
        optim = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == 'scheduledoptim':
        if config['model'].get('type') != 'attention':
            raise TypeError(f'Expected attention model to use ScheduledOptim, got {type(model)} instead')

        lr_mul = training_config.getfloat('lr_mul') if not None else 0.5
        n_warmup_steps = training_config.getint('n_warmup_steps') if not None else 4000
        d_model = config['model'].getint('d_model')
        optim = ScheduledOptim(
            torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            lr_mul=lr_mul, d_model=d_model, n_warmup_steps=n_warmup_steps
        )
    elif optimizer == 'ZO_SGD_mask':
        raw_opt_layers_strs = config['training'].get('opt_layers_strs') if not None else None
        if raw_opt_layers_strs is not None:
            opt_layers_strs = ast.literal_eval(raw_opt_layers_strs)
        # opt_layers_strs = ['TensorizedLinear', 'nn.Linear']

        if config['loss_func'].get('gradient_func') == 'gradients_auto_diff'\
        or config['training'].getboolean('debug') == True:
            model.requires_grad_(True)
        else:
            model.requires_grad_(False)
        optim = ZO_SGD_mask(
            model = model, 
            criterion = loss_fn,
            masks = named_masks,
            lr = lr,
            sigma = config['training'].getfloat('sigma'),
            n_sample  = config['training'].getint('n_sample'),
            signsgd = config['training'].getboolean('signsgd'),
            layer_by_layer = config['training'].getboolean('layer_by_layer'),
            # opt_layers_strs = config['training'].get('opt_layers_strs')
            opt_layers_strs = opt_layers_strs,
            sample_method = config['training'].get('sample_method')
        )
    elif 'ZO_SCD' in optimizer:
        raw_opt_layers_strs = config['training'].get('opt_layers_strs') if not None else None
        if raw_opt_layers_strs is not None:
            opt_layers_strs = ast.literal_eval(raw_opt_layers_strs)
        # opt_layers_strs = ['TensorizedLinear', 'nn.Linear']
        
        if config['loss_func'].get('gradient_func') == 'gradients_auto_diff'\
        or config['training'].getboolean('debug') == True:
            model.requires_grad_(True)
        else:
            model.requires_grad_(False)
               
        optim = ZO_SCD_mask(
            model = model, 
            criterion = loss_fn,
            masks = named_masks,
            lr = lr,
            grad_sparsity = config['training'].getfloat('grad_sparsity'),
            h_smooth = config['training'].getfloat('h_smooth') if not None else 0.1,
            grad_estimator = config['training'].get('grad_estimator') if not None else 'sign',
            opt_layers_strs = opt_layers_strs,
            # STP = training_config.getboolean('STP') if not None else False,
            momentum = config['training'].getfloat('momentum') if not None else 0
            # weight_decay = config['training'].getfloat('weight_decay') if not None else 0,
            # dampening = config['training'].getfloat('dampening') if not None else 0,
            # adam = config['training'].getboolean('adam') if not None else False,
            # beta_1 = config['training'].getfloat('beta_1') if not None else 0.9,
            # beta_2 = config['training'].getfloat('beta_2') if not None else 0.98
        )
    elif optimizer == "mixedtrain":
        optim = MixedTrainOptimizer(
            model,
            lr=lr,
            param_sparsity=config['training'].getfloat('param_sparsity'),
            grad_sparsity=config['training'].getfloat('grad_sparsity'),
            criterion=loss_fn,
            random_state=config['training'].getint('random_state'),
            STP = config['training'].getboolean('STP') if not None else False,
            momentum = config['training'].getfloat('momentum') if not None else 0,
            weight_decay = config['training'].getfloat('weight_decay') if not None else 0,
            dampening = config['training'].getfloat('dampening') if not None else 0,
        )
        if config['loss_func'].get('gradient_func') == 'gradients_auto_diff'\
        or config['training'].getboolean('debug') == True:
            model.requires_grad_(True)
        else:
            model.requires_grad_(False)
    elif optimizer == "flops":
        optim = FLOPSOptimizer(
            model,
            lr=lr,
            sigma=config['training'].getfloat('sigma'),
            n_sample=config['training'].getint('n_sample'),
            criterion=loss_fn,
            random_state=config['training'].getint('random_state'),
            signsgd = config['training'].getboolean('signsgd') if not None else False,
            momentum = config['training'].getfloat('momentum') if not None else 0,
            weight_decay = config['training'].getfloat('weight_decay') if not None else 0,
            dampening = config['training'].getfloat('dampening') if not None else 0,
        )
        if config['loss_func'].get('gradient_func') == 'gradients_auto_diff'\
        or config['training'].getboolean('debug') == True:
            model.requires_grad_(True)
        else:
            model.requires_grad_(False)
    else:
        raise ValueError(f'No optimizer type specified')

    if scheduler == 'steplr':
        epochs_til_decay = training_config.getint('epochs_til_decay')
        gamma=config['training'].getfloat('gamma') if not None else 0.9
        schedu = torch.optim.lr_scheduler.StepLR(optim, step_size=epochs_til_decay, gamma=gamma)
    elif scheduler == 'reducelronplateau':
        schedu = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
    elif scheduler == 'ExponentialLR':
        schedu = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=config['training'].getfloat('gamma'))
    elif scheduler == 'PresetLRScheduler':
        lr_schedule = dict()
        lr_coef_dict_string = config['training'].get('lr_schedule')
        lr_coef_dict = ast.literal_eval(lr_coef_dict_string)

        for n_epoch,lr_coef in lr_coef_dict.items():
            lr_schedule[n_epoch] = lr * lr_coef
    
        schedu = PresetLRScheduler(lr_schedule)
    else:
        schedu = None

    return optim, schedu

def set_nonlinearity_from_init(yml_config, model):
    # deterministic phase bias
    if(yml_config.noise.phase_bias):
        model.assign_random_phase_bias(random_state=int(yml_config.noise.random_state))
    # deterministic phase shifter gamma noise
    model.set_gamma_noise(float(yml_config.noise.gamma_noise_std),
                          random_state=int(yml_config.noise.random_state))
    # deterministic phase shifter crosstalk
    model.set_crosstalk_factor(float(yml_config.noise.crosstalk_factor))
    # deterministic phase quantization
    model.set_weight_bitwidth(int(yml_config.quantize.weight_bit))
    # enable/disable noisy identity
    model.set_noisy_identity(int(yml_config.sl.noisy_identity))

def training_from_ini(config, model, dataset, model_dir, loss_fn, val_fn, named_masks, device, overwrite, in_logger, writer):
    training_config = config['training']

    epochs = training_config.getint('epochs')
    epochs_til_checkpoints = training_config.getint('epochs_til_checkpoints')
    start_epoch = training_config.getint('start_epoch')
    lr_decay = training_config.getboolean('lr_decay')
    epochs_til_val = training_config.getint('epochs_til_val')
    verbose = training_config.getboolean('verbose')

    optimizer, scheduler = def_optim_from_config(config, model, loss_fn, named_masks)
    
    # if config['pretrained'].getboolean('incre') == True:
    #     checkpoint = torch.load(config['pretrained'].get('load_model_path'))
    #     optimizer.load_state_dict(checkpoint['optim'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    
    ZO_Estim = None
    if 'ZO_Estim'in config and config['ZO_Estim'].getboolean('en'):
        ZO_Estim_config = dict()
        for key, value in config['ZO_Estim'].items():
            ZO_Estim_config[key] = convert_value(value)
            
        from easydict import EasyDict
        ZO_Estim_config = EasyDict(ZO_Estim_config)
        ZO_Estim = build_ZO_Estim(ZO_Estim_config, model=model, )
        
        in_logger.critical('trainable params:')
        for param in ZO_Estim.splited_param_list:
            in_logger.critical(f'{param.name}')

    train(config=config, model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler, ZO_Estim=ZO_Estim, epochs=epochs,
          epochs_til_checkpoints=epochs_til_checkpoints, model_dir=model_dir,
          loss_fn=loss_fn, val_fn=val_fn, device=device,
          start_epoch=start_epoch, lr_decay=lr_decay, epochs_til_val=epochs_til_val,
          verbose=verbose, in_logger=in_logger, writer=writer)

def start_from_ini(sys_args, dataio_from_config, loss_fn_from_config, val_fn_from_config, eval_fn_from_config,
                   *args):

    # ================= Set up PINN model =================
    mode = sys_args.mode
    config = load_config_ini(sys_args.config_file_path)
    model_type = config['model'].get('type')
    if hasattr(sys_args, 'yml_config_path') and sys_args.yml_config_path is not None:
        yml_config.load(sys_args.yml_config_path, recursive=False)

    set_torch_deterministic(config['training'].get('random_state'))

    # ================= Set up PINN model =================
    device = config['basic'].get('device')
    dataset = dataio_from_config(config, mode)
    loss_fn = loss_fn_from_config(config)
    val_fn = val_fn_from_config(config)

    model = build_model_from_config(config, device=device).to(device)
    # model = build_model_from_config(config, device='cpu').to(device)
    
    if hasattr(sys_args, 'yml_config_path') and sys_args.yml_config_path is not None and yml_config.model.mode == 'phase':
        model.switch_mode_to("usv")
        model.sync_parameters(src="phase")
    
    # for name, param in model.named_parameters():
    #     if 'weight' in name:
    #         param.requires_grad = False
    
    # for name, param in model.named_parameters():
    #     print(name)
        
    # for name, module in model.named_modules():
    #     print(name)
    
    # ================= Set up experiment path =================
    if mode == 'eval' or mode == 'evaluate':
        model_dir = os.path.abspath(os.path.join(sys_args.config_file_path, ".."))
    else:
        if config['pretrained'].getboolean('incre') == True:
            t_type = 'inc'
        else:
            t_type = 'scr'

        name_append = '{}_lr_{}'.format(t_type, config['training'].get('lr'))
        
        if config['training'].get('optimizer') == 'ZO_SGD_mask':
            sample_method = 'gaussian' if config['training'].get('sample_method') is None else config['training'].get('sample_method')
            name_append = name_append + '_{}_sign{}_K{}'.format(sample_method, config['training'].get('signsgd'), config['training'].get('n_sample'))
        elif config['training'].get('optimizer') == 'ZO_SCD_mask':
            name_append = name_append + '_{}_grad{}_mom{}'.format(config['training'].get('grad_estimator'), config['training'].get('grad_sparsity'), config['training'].get('momentum'))

        experiment_name = config['basic'].get('experiment_name') + name_append
        experiment_root = config['basic'].get('experiment_root')

        if not experiment_root:
            experiment_root = './exp'

        # For Helmholtz
        k0_coef = config['dataio'].get('k0_coef')
        if k0_coef is not None:
            experiment_root = os.path.join(experiment_root, 'k0='+k0_coef)
        
        # model_dir = os.path.join('./experiments', experiment_name)
        model_type = config['model'].get('type')
        if model_type == 'tensorized_dnn':
            model_dir = os.path.join(
                experiment_root,
                config['model'].get('type'), 
                config['model'].get('hidden_features'), 
                config['model'].get('factorization'), 
                config['model'].get('rank'), 
                config['loss_func'].get('gradient_func'),
                config['training'].get('optimizer'), 
                config['training'].get('scheduler'), 
                experiment_name)
        elif model_type == 'dnn':
            model_dir = os.path.join(
                experiment_root, 
                config['model'].get('type'), 
                config['model'].get('hidden_features'), 
                config['GraSP'].get('pruner'), 
                config['GraSP'].get('target_ratio'), 
                config['loss_func'].get('gradient_func'),
                config['training'].get('optimizer'), 
                config['training'].get('scheduler'), 
                experiment_name)
        elif model_type == 'tonn':
            model_dir = os.path.join(
                experiment_root, 
                config['model'].get('type'), 
                config['loss_func'].get('gradient_func'),
                config['training'].get('optimizer'), 
                config['training'].get('scheduler'), 
                experiment_name)
        else:
            model_dir = os.path.join('./{}'.format(experiment_root), experiment_name)

        # Add unique identifier
        # model_dir = os.path.join(model_dir, time.strftime("%Y%m%d-%H%M%S"))
        model_dir = os.path.join(model_dir, str(os.getpid()))
        
        # Make path
        start_epoch = config['training'].getint('start_epoch')
        overwrite=sys_args.overwrite
        # if not start_epoch:
        
        if os.path.exists(model_dir) and overwrite:
            shutil.rmtree(model_dir)
        elif os.path.exists(model_dir) and not overwrite:
            while os.path.exists(model_dir):
                # model_dir = os.path.join(model_dir, time.strftime("%Y%m%d-%H%M%S"))
                model_dir = model_dir + '_new'

        os.makedirs(model_dir)
        config['basic']['experiment_name'] = model_dir.split('/')[-1]

        save_config_to_ini(config, model_dir)
        
        # setup logger
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        fh = logging.FileHandler('{}/log.log'.format(model_dir))
        fh.setLevel(logging.CRITICAL)  # or any level you want
        logger.addHandler(fh)

        # show in console
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.CRITICAL) 
        # logger.addHandler(console_handler)

        # setup summarywriter
        writer = SummaryWriter(model_dir)
        
        # ================= GraSP prune =================
        pruner = config['GraSP'].getboolean('pruner') if not None else False

        if pruner == True:
            masks, named_masks = GraSP_from_ini(
                config=config,
                model=model,
                dataset=dataset,
                loss_fn=loss_fn,
                device=device
            )
        else:
            masks = None
            named_masks = None
        
    if config['pretrained'].getboolean('incre') == True:
        checkpoint = torch.load(config['pretrained'].get('load_model_path'))
        model.load_state_dict(checkpoint['model'])

        ##### offchip Pretrain
        # if hasattr(model.net.fc1, 'mode') and model.net.fc1.mode == "weight":
        #     model.switch_mode_to("usv")
        #     model.sync_parameters(src="weight")
            
        # pretrain_loss = val_fn(model=model, dataset=dataset)
        # logger.critical('Pretrained model val loss: {}'.format(pretrain_loss))

    # ================= Setup nonlinearity =================
    if config['model'].get('type') in ('onn', 'tonn'):
        if config['model'].getboolean('mzi_noise') == True:
            set_nonlinearity_from_init(yml_config, model)
    else:
        pass

    if mode == 'train' or mode == 'training' or mode == 'both':
        
        training_from_ini(config=config, model=model, dataset=dataset, model_dir=model_dir,
                          loss_fn=loss_fn, val_fn=val_fn, named_masks=named_masks, device=device, overwrite=sys_args.overwrite,
                          in_logger=logger, writer=writer)

    # if mode == 'eval' or mode == 'evaluate' or mode == 'both':
    #     eval_epoch_list = sys_args.eval_epoch_list
    #     eval_fn = eval_fn_from_config(config, model_dir=model_dir)

    #     for eval_epoch in eval_epoch_list:
    #         eval_fn(model=model, dataset=dataset, eval_epoch=eval_epoch)


def debug_from_ini(config_file_path, mode, dataio_from_config, loss_fn_from_config, val_fn_from_config,
                   eval_fn_from_config):
    config = load_config_ini(config_file_path)
    dataset = dataio_from_config(config, mode)
    device = config['basic'].get('device')

    model = build_model_from_config(config, device)
    loss_fn = loss_fn_from_config(config)

    experiment_name = config['basic'].get('experiment_name')
    experiment_root = config['basic'].get('experiment_root')

    if not experiment_root:
        model_dir = os.path.join('./experiments', experiment_name)
    else:
        model_dir = os.path.join('./{}'.format(experiment_root), experiment_name)

    val_fn = val_fn_from_config(config)
    eval_fn = eval_fn_from_config(config, model_dir=model_dir)

    return model, dataset, model_dir, loss_fn, val_fn, eval_fn, device

