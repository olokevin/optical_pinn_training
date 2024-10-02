import torch
import torch.nn as nn

from .ZO_utils import SplitedLayer, SplitedParam, split_model
from .ZO_Estim_MC import ZO_Estim_MC

### Model specific utils ###
from .ZO_fwd_utils import opt_able_layers_dict, get_iterable_block_name, pre_block_forward, post_block_forward

def create_opt_layer_list(layer_list, opt_able_layers_dict):
    if isinstance(layer_list, str):
        return opt_able_layers_dict[layer_list]
    elif isinstance(layer_list, list):
        opt_layers = []
        for layer_str in layer_list:
            opt_layers.append(opt_able_layers_dict[layer_str])
        return tuple(opt_layers)
    else:
        raise (ValueError("opt_layers_strs should either be a string of a list of strings"))

def fwd_hook_get_output_shape(module, input, output):
    module.output_shape = output.shape

def MZI_commit_fn(layer, param, param_name) -> None:
    def _commit_fn():
        if param_name == "phase_U":
            phase_bias = layer.phase_bias_U
            delta_list = layer.delta_list_U
            quantizer = layer.phase_U_quantizer
            layer.U.data.copy_(
                layer.decomposer.reconstruct(
                    delta_list,
                    layer.decomposer.v2m(quantizer(param.view(phase_bias.size(0), phase_bias.size(1), -1)) + phase_bias),
                )
            )
        elif param_name == "phase_V":
            phase_bias = layer.phase_bias_V
            delta_list = layer.delta_list_V
            quantizer = layer.phase_V_quantizer

            layer.V.data.copy_(
                layer.decomposer.reconstruct(
                    delta_list,
                    layer.decomposer.v2m(quantizer(param.view(phase_bias.size(0), phase_bias.size(1), -1)) + phase_bias),
                )
            )

        elif param_name == "phase_S":
            layer.S.data.copy_(param.data.cos().view_as(layer.S).mul_(layer.S_scale))
        else:
            raise ValueError(f"Wrong param_name {param_name}")
    return _commit_fn

def build_ZO_Estim(config, model):
    if config.name == 'ZO_Estim_MC':
        ### split model
        iterable_block_name = get_iterable_block_name()
        split_modules_list = split_model(model, iterable_block_name)

        splited_param_list = None
        splited_layer_list = None

        ### Param perturb 
        if config.param_perturb_param_list is not None:
            param_perturb_param_list = config.param_perturb_param_list
            if config.param_perturb_block_idx_list == 'all':
                param_perturb_block_idx_list = list(range(len(split_modules_list)))
            else:
                param_perturb_block_idx_list = config.param_perturb_block_idx_list
            
            splited_param_list = []

            for block_idx in param_perturb_block_idx_list:
                if block_idx < 0:
                    block_idx = len(split_modules_list) + block_idx
                block = split_modules_list[block_idx]
                
                from core.models.layers.custom_conv2d import MZIBlockConv2d
                from core.models.layers.custom_linear import MZIBlockLinear
                from core.tensor_fwd_bwd.tensorized_linear import TensorizedLinear
                from core.models.sparse_bp_mlp import LinearBlock
                from core.models.sparse_bp_ttm_mlp import TTM_LinearBlock
                
                for layer_name, layer in block.named_modules():
                    if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d)):
                        # for param_name in ["S",]:
                        #     param = getattr(layer, param_name)
                        #     commit_fn = None
                        #     splited_param_list.append(SplitedParam(idx=block_idx, name=f'{block_idx}.{layer_name}.{param_name}', layer=layer, param=param, commit_fn=commit_fn))  
                        for param_name in ["phase_U", "phase_S", "phase_V"]:
                        # for param_name in ["phase_S",]:
                            param = getattr(layer, param_name)
                            if layer.mode == 'phase':
                                commit_fn = None
                            else:
                                commit_fn = MZI_commit_fn(layer, param, param_name)
                            splited_param_list.append(SplitedParam(idx=block_idx, name=f'{block_idx}.{layer_name}.{param_name}', layer=layer, param=param, commit_fn=commit_fn))  
                    elif isinstance(layer, (TTM_LinearBlock)):
                        for param_name in ["bias",]:
                            param = getattr(layer, param_name)
                            splited_param_list.append(SplitedParam(idx=block_idx, name=f'{block_idx}.{layer_name}.{param_name}', layer=layer, param=param))
                    elif isinstance(layer, (TensorizedLinear, nn.Linear, nn.Conv2d)):
                        for param_name, param in layer.named_parameters():
                            if param.requires_grad:
                                splited_param_list.append(SplitedParam(idx=block_idx, name=f'{block_idx}.{layer_name}.{param_name}', layer=layer, param=param))
                
                # for name, param in block.named_parameters():
                #     for keyword in param_perturb_param_list:
                #         # if keyword in name:
                #         index = name.find(keyword)
                #         if index == -1:
                #             continue
                #         elif index == 0:
                #             layer = block
                #         else: # index > 0
                #             layer = block._modules[name[:index-1]]       
                        
                #         splited_param_list.append(SplitedParam(idx=block_idx, name=f'{block_idx}.{name}', layer=layer, param=param))   
                
        ### Actv perturb 
        if config.actv_perturb_layer_list is not None:
            # actv_perturb_layer_list = create_opt_layer_list(config.actv_perturb_layer_list, opt_able_layers_dict)
            actv_perturb_layer_list = config.actv_perturb_layer_list

            if config.actv_perturb_block_idx_list == 'all':
                actv_perturb_block_idx_list = list(range(len(split_modules_list)))
            else:
                actv_perturb_block_idx_list = config.actv_perturb_block_idx_list

            splited_layer_list = []

            for block_idx in actv_perturb_block_idx_list:
                if block_idx < 0:
                    block_idx = len(split_modules_list) + block_idx
                block = split_modules_list[block_idx]
                for name, layer in block.named_modules():
                    if any(keyword in name for keyword in actv_perturb_layer_list):
                        splited_layer_list.append(SplitedLayer(idx=block_idx, name=name, layer=layer))
                        fwd_hook_handle = layer.register_forward_hook(fwd_hook_get_output_shape)

        ZO_Estim = ZO_Estim_MC(
            model = model, 
            obj_fn_type = config.obj_fn_type,
            splited_param_list = splited_param_list,
            splited_layer_list = splited_layer_list,

            sigma = config.sigma,
            n_sample  = config.n_sample,
            signsgd = config.signsgd,
            
            quantized = config.quantized,
            estimate_method = config.estimate_method,
            sample_method = config.sample_method,
            normalize_perturbation = config.normalize_perturbation,
            
            scale = config.scale,
            
            en_layerwise_perturbation = config.en_layerwise_perturbation,
            en_partial_forward = config.en_partial_forward,
            en_param_commit = config.en_param_commit,

            get_iterable_block_name = get_iterable_block_name,
            pre_block_forward = pre_block_forward,
            post_block_forward = post_block_forward
        )
        return ZO_Estim
    else:
        return NotImplementedError

def build_obj_fn(obj_fn_type, **kwargs):
    if obj_fn_type == 'classifier':
        obj_fn = build_obj_fn_classifier(**kwargs)
    elif obj_fn_type == 'classifier_layerwise':
        obj_fn = build_obj_fn_classifier_layerwise(**kwargs)
    elif obj_fn_type == 'classifier_acc':
        obj_fn = build_obj_fn_classifier_acc(**kwargs)
    elif obj_fn_type == 'pinn':
        obj_fn = build_obj_fn_pinn(**kwargs)
    else:
        raise NotImplementedError
    return obj_fn

def build_obj_fn_pinn(model, dataset, loss_fn, inputs=None):
    def _obj_fn(return_loss_reduction='mean'):
        train_loss = loss_fn(model=model, dataset=dataset, inputs=inputs, return_loss_reduction=return_loss_reduction)
        ##### Calculate the penalty for weights outside the range
        # max_val = 0.9
        # # min_val = 0.02
        # min_val = -0.9
        # penalty = torch.tensor(0.0, device=train_loss.device)
        # for name, param in model.named_parameters():
        #     if 'weight' in name:
        #         penalty += torch.sum(torch.relu(param - max_val) + torch.relu(min_val - param))
        # train_loss = train_loss + penalty
        
        y = False
        return y, train_loss

    return _obj_fn
  
def build_obj_fn_classifier(data, target, model, criterion):
    def _obj_fn():
        y = model(data)
        return y, criterion(y, target)
    
    return _obj_fn

def build_obj_fn_classifier_acc(data, target, model, criterion):
    def _obj_fn():
        outputs = model(data)
        _, predicted = outputs.max(1)
        total = target.size(0)
        correct = predicted.eq(target).sum().item()
        err = 1 - correct / total

        return outputs, err
    
    return _obj_fn

def build_obj_fn_classifier_layerwise(data, target, model, criterion, get_iterable_block_name=None, pre_block_forward=None, post_block_forward=None):
    if get_iterable_block_name is not None:
        iterable_block_name = get_iterable_block_name()
    else:
        iterable_block_name = None
    split_modules_list = split_model(model, iterable_block_name)
    
    # if no attribute for _obj_fn: same as build_obj_fn_classifier
    def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx == None:
            ending_idx = len(split_modules_list)

        if starting_idx == 0:
            y = data
            ### pre_block_forward when start from image input
            if pre_block_forward is not None:
                y = pre_block_forward(model, y)
        else:
            assert input is not None
            y = input
        
        if detach_idx is not None and detach_idx < 0:
            detach_idx = len(split_modules_list) + detach_idx
        
        for i in range(starting_idx, ending_idx):
            y = split_modules_list[i](y)
            if detach_idx is not None and i == detach_idx:
                y = y.detach()
                y.requires_grad = True
            
        ### post_block_forward when end at classifier head
        if ending_idx == len(split_modules_list):
            if post_block_forward is not None:
                y = post_block_forward(model, y)
           
        if return_loss_reduction == 'mean':
            criterion.reduction = 'mean'
            return y, criterion(y, target)
        elif return_loss_reduction == 'none':
            criterion.reduction = 'none'
            loss = criterion(y, target)
            criterion.reduction = 'mean'
            return y, loss
        elif return_loss_reduction == 'no_loss':
            return y
        else:
            raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
    return _obj_fn

# def build_obj_fn_classifier_layerwise(data, target, model, criterion, iterable_block_name=None):
#     split_modules_list = split_model(model, iterable_block_name)
    
#     # if no attribute for _obj_fn: same as build_obj_fn_classifier
#     def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
#         if ending_idx == None:
#             ending_idx = len(split_modules_list)

#         if starting_idx == 0:
#             y = data
#         else:
#             assert input is not None
#             y = input
        
#         if detach_idx is not None and detach_idx < 0:
#             detach_idx = len(split_modules_list) + detach_idx
        
#         for i in range(starting_idx, ending_idx):
#             y = split_modules_list[i](y)
#             if detach_idx is not None and i == detach_idx:
#                 y = y.detach()
#                 y.requires_grad = True
           
#         if return_loss_reduction == 'mean':
#             criterion.reduction = 'mean'
#             return y, criterion(y, target)
#         elif return_loss_reduction == 'none':
#             criterion.reduction = 'none'
#             loss = criterion(y, target)
#             criterion.reduction = 'mean'
#             return y, loss
#         elif return_loss_reduction == 'no_loss':
#             return y
#         else:
#             raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
#     return _obj_fn