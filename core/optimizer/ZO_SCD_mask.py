"""
Description:
Author:
Date: 2021-10-24 16:27:47
LastEditors:
LastEditTime: 2021-10-24 16:27:47
"""
from typing import Callable

import copy
import numpy as np
import torch
from pyutils.general import logger
from pyutils.torch_train import (
    get_learning_rate,
    get_random_state,
    set_torch_deterministic,
    set_torch_stochastic,
)
from torch import nn
from torch.functional import Tensor
from torch.optim import Optimizer
from core.tensor_layers.layers import TensorizedLinear_linear, TensorizedLinear_module, TensorizedLinear_module_tonn
from core.tensor_fwd_bwd.tensorized_linear import TensorizedLinear
# from tensorized_linear import TensorizedLinear

__all__ = ["ZO_SCD_mask"]

opt_able_layers_dict = {
    'nn.Linear': nn.Linear,
    'nn.Conv2d': nn.Conv2d,
    'TensorizedLinear': TensorizedLinear,
    'TensorizedLinear_module': TensorizedLinear_module,
    'TensorizedLinear_module_tonn': TensorizedLinear_module_tonn
}

class ZO_SCD_mask(Optimizer):
    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        masks,
        lr: float = 0.1,
        grad_sparsity: float = 0.1,
        h_smooth: float = 0.001,
        grad_estimator: str = 'sign',
        opt_layers_strs: list = [],
        STP: bool = True,
        momentum: float = 0,
        weight_decay: float = 0,
        dampening: float = 0
    ):
        defaults = dict(lr=lr)
        super().__init__(model.parameters(), defaults)
        self.forward_counter = 0
        self.global_step = 0
        self.model = model
        self.criterion = criterion
        self.masks = masks
        self.grad_sparsity = grad_sparsity
        self.h_smooth = h_smooth
        self.grad_estimator = grad_estimator
        self.opt_layers_strs = opt_layers_strs
        self.STP = STP
        
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening

        # initialize
        self.m_t = None

        self.init_state()

    def init_state(self):
        self.opt_layers = self.create_opt_layers()
        self.modules = self.extract_modules(self.model)
        self.trainable_params = self.extract_trainable_parameters(self.model)
        # self.untrainable_params = self.extract_untrainable_parameters(self.model)

        if self.masks is not None:
            self.enable_mixedtraining(self.masks)
        else:
            self.disable_mixedtraining()
        
        # if self.patience_table == True:
        #     self.enable_patience_table()

    def create_opt_layers(self):
        if isinstance(self.opt_layers_strs, str):
            return opt_able_layers_dict[self.opt_layers_strs]
        elif isinstance(self.opt_layers_strs, list):
            opt_layers = []
            for layer_str in self.opt_layers_strs:
                opt_layers.append(opt_able_layers_dict[layer_str])
            return tuple(opt_layers)
        else:
            raise (ValueError("opt_layers_strs should either be a string of a list of strings"))
        
    def extract_modules(self, model):
        return {
            layer_name: layer
            for layer_name, layer in model.named_modules()
            if isinstance(layer, self.opt_layers)
        }

    def extract_trainable_parameters(self, model):
    # flatten the parameters
        trainable_parameters = dict()
        for layer_name, layer in self.modules.items():
            if isinstance(layer, (TensorizedLinear_module_tonn)):
                trainable_parameters[layer_name] = {
                    "tt_cores-"+str(i): getattr(layer,'tt_cores')[i].weight.data.view(-1)
                    for i in range(getattr(layer, 'order'))
                }
            elif isinstance(layer, (TensorizedLinear_module)):
                trainable_parameters[layer_name] = {
                    str(i): getattr(layer.tensor,'factors')[i].view(-1)
                    for i in range(getattr(layer.tensor, 'order'))
                }
                if layer.bias is not None:
                    trainable_parameters[layer_name]["bias"] = layer.bias.data
            elif isinstance(layer, (TensorizedLinear)):
                trainable_parameters[layer_name] = {
                    str(i): getattr(layer.weight.factors, 'factor_'+str(i)).view(-1)
                    for i in range(getattr(layer, 'ndim'))
                }
                if layer.bias is not None:
                    trainable_parameters[layer_name]["bias"] = layer.bias.data
            elif isinstance(layer, (nn.Linear, nn.Conv2d)):
                trainable_parameters[layer_name] = {  
                    param_name: getattr(layer, param_name).data.view(-1)
                    for param_name in ["weight"]
                }
                if layer.bias is not None:
                    trainable_parameters[layer_name]["bias"] = layer.bias.data
        return trainable_parameters
    
    # def extract_untrainable_parameters(self, model):
    #     return {
    #         layer_name: {
    #             param_name: getattr(layer, param_name)
    #             for param_name in ["phase_bias_U", "phase_bias_V", "delta_list_U", "delta_list_V"]
    #         }
    #         for layer_name, layer in model.named_modules()
    #         if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))
    #     }

    def enable_mixedtraining(self, masks):
        # different from ZO_SGD!
        # need to change to index 
        # matrix: [4, 5, 6, 7]
        # mask:   [0, 1, 0, 1]
        #  =>     [1, 3]
        self.mixedtrain_masks = {
            layer_name: {
                p_name: torch.arange(p.numel(), device=p.device)[
                    masks[layer_name][p_name].to(p.device).bool().view(-1)
                ] if p_name != "bias"\
                else torch.arange(p.numel(), device=p.device)
                for p_name, p in layer_params.items()
            }
            for layer_name, layer_params in self.trainable_params.items()
        }
        print(self.mixedtrain_masks)
    
    def disable_mixedtraining(self):
        self.mixedtrain_masks = {
            layer_name: {
                p_name: torch.arange(p.numel(), device=p.device) for p_name, p in layer_params.items()
            }
            for layer_name, layer_params in self.trainable_params.items()
        }
    
    def get_forward_cnt(self):
        return self.forward_counter

    def get_param_num(self):
        total_num = 0
        for layer_name, layer_params in self.mixedtrain_masks.items():
            for p_name, p in layer_params.items():
                total_num = total_num + p.numel()      
        return total_num
    
    def extract_grad_fo(self, model):
    # flatten the parameters
        grad_fo = dict()
        for layer_name, layer in self.modules.items():
            if isinstance(layer, (TensorizedLinear_module_tonn)):
                grad_fo[layer_name] = {
                    "tt_cores-"+str(i): getattr(layer,'tt_cores')[i].weight.grad.view(-1)
                    for i in range(getattr(layer, 'order'))
                }
            elif isinstance(layer, (TensorizedLinear_module)):
                grad_fo[layer_name] = {
                    str(i): getattr(layer.tensor,'factors')[i].grad.view(-1)
                    for i in range(getattr(layer.tensor, 'order'))
                }
                if layer.bias is not None:
                    grad_fo[layer_name]["bias"] = layer.bias.grad
            elif isinstance(layer, (TensorizedLinear)):
                grad_fo[layer_name] = {
                    str(i): getattr(layer.weight.factors, 'factor_'+str(i)).grad.view(-1)
                    for i in range(getattr(layer, 'ndim'))
                }
                if layer.bias is not None:
                    grad_fo[layer_name]["bias"] = layer.bias.grad
            elif isinstance(layer, (nn.Linear, nn.Conv2d)):
                grad_fo[layer_name] = {  
                    param_name: getattr(layer, param_name).grad.view(-1)
                    for param_name in ["weight"]
                }
                if layer.bias is not None:
                    grad_fo[layer_name]["bias"] = layer.bias.data
        return grad_fo

    def cal_grad_err(self, params, grad_zo, grad_fo):
        return {
            layer_name: {
                p_name: grad_zo[layer_name][p_name] - grad_fo[layer_name][p_name]
                for p_name, p in layer_params.items()
            }
            for layer_name, layer_params in params.items()
        }
    
    def enable_STP(self):
        self.STP = True
    
    def disable_stp(self):
        self.STP = False

    def commit(self, layer_name: str, param_name: str, param: Tensor) -> None:
        '''
            layer_name:{param_name: p}
        '''
        layer = self.modules[layer_name]
        if isinstance(layer, (TensorizedLinear_module_tonn)):
            raise NotImplementedError
        elif isinstance(layer, (TensorizedLinear_module)):
            if param_name == 'bias':
                layer.bias.data = param
            else:
                idx = int(param_name)
                ttm_shape = layer.tensor.factors[idx].shape
                layer.tensor.factors[idx].data = param.reshape(ttm_shape)
        elif isinstance(layer, (TensorizedLinear)):
            if param_name == 'bias':
                layer.bias.data = param
            else:
                idx = int(param_name)
                if layer.factorization == 'tt':
                    tt_shape = (layer.weight.rank[idx],
                                layer.weight.shape[idx],
                                layer.weight.rank[idx+1])
                elif layer.factorization == 'blocktt':
                    tt_shape = (layer.weight.rank[idx],
                                layer.weight.tensorized_shape[0][idx],
                                layer.weight.tensorized_shape[1][idx],
                                layer.weight.rank[idx+1])
                # t_param = torch.nn.parameter.Parameter(param.reshape(tt_shape), requires_grad=False)
                # setattr(layer.weight.factors, 'factor_'+param_name, t_param)
                setattr(getattr(layer.weight.factors, 'factor_'+param_name), 'data', param.reshape(tt_shape))
        elif isinstance(layer, (nn.Linear, nn.Conv2d)):
            if param_name == "weight":
                layer.weight.data = param.reshape(layer.out_features, layer.in_features)
            else:
                raise ValueError(f"Wrong param_name {param_name}")

    def zo_coordinate_descent_sign(self, obj_fn, params):
        """
        description: stochastic coordinate-wise descent.
        (2020 DAC) sparse fine-tuning
        
        A more efficient sign SGD
        """
        # evaluate objective on the current parameters
        with torch.no_grad():
            y, old_loss = obj_fn()
            self.forward_counter += 1
        lr = get_learning_rate(self)

        for layer_name, layer_params in params.items():
            layer_masks = self.mixedtrain_masks[layer_name]
            for p_name, p in layer_params.items():
                selected_indices = layer_masks[p_name]
                for idx in selected_indices:
                    # ============ SparseTune in FLOPS+ [Gu+, DAC 2020] ============
                    cur_seed = get_random_state()
                    set_torch_stochastic()
                    seed = np.random.rand()
                    set_torch_deterministic(cur_seed)
                    if seed < self.grad_sparsity:
                        continue
                    old_value = copy.deepcopy(p.data[idx])
                    pos_perturbed_value = old_value + lr
                    neg_perturbed_value = old_value - lr
                    
                    with torch.no_grad():  # training=True to enable profiling, but do not save graph
                        p.data[idx] = pos_perturbed_value
                        # self.commit(layer_name, p_name, p)
                        y, new_loss = obj_fn()
                        self.forward_counter += 1

                    if new_loss < old_loss:           # pos works
                        old_loss = new_loss
                    else:
                        with torch.no_grad():
                            p.data[idx] = neg_perturbed_value
                            # self.commit(layer_name, p_name, p)
                            y, new_loss = obj_fn()
                            self.forward_counter += 1
                        if self.STP == False:    # SZO-SCD
                            old_loss = new_loss
                        else:               # STP
                            if new_loss < old_loss:   # neg works
                                old_loss = new_loss 
                            else:                     # remain
                                p.data[idx] = old_value
                                # with torch.no_grad():
                                #     self.commit(layer_name, p_name, p)
        return y, old_loss

    # ============ apply gradients all together ============
    # def _apply_gradients(self, params, grad, lr):
    #     return {
    #         layer_name: {p_name: p.sub_(grad[layer_name][p_name] * lr) for p_name, p in layer_params.items()}
    #         for layer_name, layer_params in params.items()
    #     }

    # ============ apply gradients with momentum weight_decay dampening ============
    def _apply_gradients(self, params, grads, lr):
        if self.weight_decay != 0:
            grads_t = {
                layer_name: {
                    p_name: grads[layer_name][p_name] + self.weight_decay * params[layer_name][p_name]
                    for p_name, p in layer_params.items()
                }
                for layer_name, layer_params in params.items()
            }     
        else:
            grads_t = grads
        
        # t = 0, initialize bts (dict)
        if self.m_t is None:
            self.m_t = copy.deepcopy(grads_t)
        elif self.momentum != 0:
            for layer_name, layer_params in params.items():
                for p_name, p in layer_params.items():
                    self.m_t[layer_name][p_name] = self.momentum * self.m_t[layer_name][p_name] + (1-self.dampening)*grads_t[layer_name][p_name]
        else:
            self.m_t = grads_t
        
        return {
            layer_name: {p_name: p.sub_(self.m_t[layer_name][p_name] * lr) for p_name, p in layer_params.items()}
            for layer_name, layer_params in params.items()
        }
        
    def zo_coordinate_descent_batch(self, obj_fn, params):
        """
        description: stochastic coordinate-wise descent.
        A variation of 2020 DAC fine-tuning

        Update a batch of coordinates together 
        (save coordinate-wise gradients)
        """
        # evaluate objective on the current parameters
        with torch.no_grad():
            y, old_loss = obj_fn()
            self.forward_counter += 1
        lr = get_learning_rate(self)
        grads = dict()

        for layer_name, layer_params in params.items():
            layer_masks = self.mixedtrain_masks[layer_name]
            layer_grads = dict()

            for p_name, p in layer_params.items():
                selected_indices = layer_masks[p_name]
                # param_grad: same size of p, masked remained 0
                param_grad = torch.zeros_like(p)

                for idx in selected_indices:
                    # ============ SparseTune in FLOPS+ [Gu+, DAC 2020] ============
                    cur_seed = get_random_state()
                    set_torch_stochastic()
                    seed = np.random.rand()
                    set_torch_deterministic(cur_seed)
                    if seed < self.grad_sparsity:
                        continue
                    old_value = copy.deepcopy(p.data[idx])
                    pos_perturbed_value = old_value + lr
                    neg_perturbed_value = old_value - lr

                    with torch.no_grad():  # training=True to enable profiling, but do not save graph
                        p.data[idx] = pos_perturbed_value
                        # self.commit(layer_name, p_name, p)
                        y, pos_loss = obj_fn()
                        self.forward_counter += 1
                        
                        if self.STP == True:
                            p.data[idx] = neg_perturbed_value
                            # self.commit(layer_name, p_name, p)
                            y, neg_loss = obj_fn()
                            self.forward_counter += 1
                            loss_list = [old_loss, pos_loss, neg_loss]
                            grad_list = [0, -1, 1]
                        else:
                            loss_list = [old_loss, pos_loss]
                            grad_list = [1, -1]

                        param_grad[idx] = grad_list[loss_list.index(min(loss_list))]

                        p.data[idx] = old_value
                        # self.commit(layer_name, p_name, p)

                layer_grads[p_name] = param_grad

            grads[layer_name] = layer_grads

        self._apply_gradients(params, grads, lr)
        return y, old_loss, grads
    
    # ============ ZO-SCD-esti ============
    def zo_coordinate_descent_esti(self, obj_fn, params):
        """
        description: stochastic coordinate-wise descent.
            Coordinate Gradient Estimation
            Update all coordinate gradient at the end

        """
        # evaluate objective on the current parameters
        with torch.no_grad():
            y, old_loss = obj_fn()
            self.forward_counter += 1
        lr = get_learning_rate(self)
        grads = dict()

        for layer_name, layer_params in params.items():
            layer_masks = self.mixedtrain_masks[layer_name]
            layer_grads = dict()

            for p_name, p in layer_params.items():
                selected_indices = layer_masks[p_name]
                # param_grad: same size of p, masked remained 0
                param_grad = torch.zeros_like(p)

                for idx in selected_indices:
                    # ============ SparseTune in FLOPS+ [Gu+, DAC 2020] ============
                    cur_seed = get_random_state()
                    set_torch_stochastic()
                    seed = np.random.rand()
                    set_torch_deterministic(cur_seed)
                    if seed < self.grad_sparsity:
                        continue
                    old_value = copy.deepcopy(p.data[idx])
                    pos_perturbed_value = old_value + self.h_smooth
                    neg_perturbed_value = old_value - self.h_smooth

                    with torch.no_grad():  # training=True to enable profiling, but do not save graph
                        p.data[idx] = pos_perturbed_value
                        # self.commit(layer_name, p_name, p)
                        y, pos_loss = obj_fn()
                        self.forward_counter += 1

                        if self.STP == True:
                            p.data[idx] = neg_perturbed_value
                            # self.commit(layer_name, p_name, p)
                            y, neg_loss = obj_fn()
                            self.forward_counter += 1

                            param_grad[idx] = (pos_loss-neg_loss)/2/self.h_smooth
                        else:
                            param_grad[idx] = (pos_loss-old_loss)/self.h_smooth
                        
                        p.data[idx] = old_value
                        # self.commit(layer_name, p_name, p)

                layer_grads[p_name] = param_grad

            grads[layer_name] = layer_grads

        self._apply_gradients(params, grads, lr)
        return y, old_loss, grads
    
    def build_obj_fn(self, data, target, model, criterion):
        def _obj_fn():
            y = model(data)
            return y, criterion(y, target)

        return _obj_fn

    def build_obj_fn_ATIS(self, datas, targets, model, criterion):
        def _obj_fn():
            # optimizer.step((w1,attn,seg),(target,slot_label))
            w1 = datas[0]
            attn = datas[1]
            seg = datas[2]

            target = targets[0]
            slot_label = targets[1]

            pred,pred_slot = model(w1,attn=attn,seg=seg)

            pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
            slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)

            loss_MLM =  criterion(pred_slot, slot_label)
            loss = criterion(pred,target)  + loss_MLM
            
            return (pred, pred_slot), loss
        return _obj_fn
    
    def step(self, data, target, en_debug=False, ATIS=False):
        if ATIS == True:
            self.obj_fn = self.build_obj_fn_ATIS(data, target, self.model, self.criterion)
        else:
            self.obj_fn = self.build_obj_fn(data, target, self.model, self.criterion)
            
        if self.grad_estimator == 'sign':
            y, loss = self.zo_coordinate_descent_sign(self.obj_fn, self.trainable_params)
            grads_zo = None
        elif self.grad_estimator == 'batch':
            y, loss, grads_zo = self.zo_coordinate_descent_batch(self.obj_fn, self.trainable_params)
        elif self.grad_estimator == 'esti':
            y, loss, grads_zo = self.zo_coordinate_descent_esti(self.obj_fn, self.trainable_params)
        # update internal parameters
        self.global_step += 1
        # output gradients
        if en_debug == True:
            grads_fo = self.extract_grad_fo(self.model)
            grads_err = self.cal_grad_err(self.trainable_params, grads_zo, grads_fo)
            return y, loss, (grads_zo, grads_fo, grads_err)
        else:
            return y, loss, grads_zo
    
    # ==================== For PINNs ====================
    def build_obj_fn_pinn(self, model, dataset, loss_fn):
        inputs = dataset.train()
        def _obj_fn():
            train_loss = loss_fn(model=model, dataset=dataset, inputs=inputs)
            y = False
            return y, train_loss

        return _obj_fn
    
    def step_pinn(self, model, dataset, loss_fn, en_debug=False):
        self.obj_fn = self.build_obj_fn_pinn(model, dataset, loss_fn)
        
        if self.grad_estimator == 'sign':
            y, loss = self.zo_coordinate_descent_sign(self.obj_fn, self.trainable_params, STP=False)
            grads_zo = None
        elif self.grad_estimator == 'STP':
            y, loss = self.zo_coordinate_descent_sign(self.obj_fn, self.trainable_params, STP=True)
            grads_zo = None
        elif self.grad_estimator == 'batch':
            y, loss, grads_zo = self.zo_coordinate_descent_batch(self.obj_fn, self.trainable_params)
        elif self.grad_estimator == 'esti':
            y, loss, grads_zo = self.zo_coordinate_descent_esti(self.obj_fn, self.trainable_params)
        # update internal parameters
        self.global_step += 1
        # output gradients
        if en_debug == True:
            grads_fo = self.extract_grad_fo(self.model)
            grads_err = self.cal_grad_err(self.trainable_params, grads_zo, grads_fo)
            return y, loss, (grads_zo, grads_fo, grads_err)
        else:
            return y, loss, grads_zo
