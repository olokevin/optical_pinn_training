"""
Description:
Author:
Date: 2021-10-24 16:27:47
LastEditors:
LastEditTime: 2021-10-24 16:27:47
"""
from typing import Callable

import numpy as np
import torch
from core.models.layers.custom_conv2d import MZIBlockConv2d
from core.models.layers.custom_linear import MZIBlockLinear
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
from torchonn.op.mzi_op import RealUnitaryDecomposerBatch, checkerboard_to_vector, vector_to_checkerboard

import copy

__all__ = ["MixedTrainOptimizer"]


class MixedTrainOptimizer(Optimizer):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.1,
        param_sparsity: float = 0.0,
        grad_sparsity: float = 0.0,
        criterion: Callable = None,
        random_state: int = None,
        STP: bool = False,
        momentum: float = 0,
        weight_decay: float = 0,
        dampening: float = 0,
    ):
        defaults = dict(lr=lr)
        super().__init__(model.parameters(), defaults)
        self.param_sparsity = param_sparsity
        self.grad_sparsity = grad_sparsity
        self.forward_counter = 0
        self.global_step = 0
        self.model = model
        self.model.switch_mode_to("usv")
        self.random_state = random_state
        self.criterion = criterion

        self.STP = STP
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening

        # initialize
        self.m_t = None
        self.v_t = None

        self.init_state()

    def init_state(self):
        self.model.sync_parameters(src="usv")
        self.modules = self.extract_modules(self.model)
        self.trainable_params = self.extract_trainable_parameters(self.model)
        self.untrainable_params = self.extract_untrainable_parameters(self.model)
        self.quantizers = self.extract_quantizer(self.model)
        if self.param_sparsity > 1e-9:
            self.model.switch_mode_to("phase")
            masks = self.model.gen_mixedtraining_mask(self.param_sparsity, random_state=self.random_state)
            self.model.switch_mode_to("usv")
            self.enable_mixedtraining(masks)
        else:
            self.disable_mixedtraining()
        self.decomposer = RealUnitaryDecomposerBatch(alg="clements")
        self.m2v = checkerboard_to_vector
        self.v2m = vector_to_checkerboard

    def extract_modules(self, model):
        return {
            layer_name: layer
            for layer_name, layer in model.named_modules()
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))
        }

    def extract_trainable_parameters(self, model):
        # always flatten the parameters
        return {
            layer_name: {
                param_name: getattr(layer, param_name).view(-1)
                for param_name in ["phase_U", "phase_S", "phase_V"]
            }
            for layer_name, layer in model.named_modules()
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))
        }

    def extract_untrainable_parameters(self, model):
        return {
            layer_name: {
                param_name: getattr(layer, param_name)
                for param_name in ["phase_bias_U", "phase_bias_V", "delta_list_U", "delta_list_V"]
            }
            for layer_name, layer in model.named_modules()
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))
        }

    def extract_quantizer(self, model):
        return {
            layer_name: {
                param_name: getattr(layer, param_name)
                for param_name in ["phase_U_quantizer", "phase_V_quantizer"]
            }
            for layer_name, layer in model.named_modules()
            if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))
        }

    def enable_mixedtraining(self, masks):
        # need to change to index [0, 1, 2, 3][0, 1, 0, 1] => [1, 3]
        self.mixedtrain_masks = {
            layer_name: {
                p_name: torch.arange(p.numel(), device=p.device)[
                    masks[layer_name][p_name].to(p.device).bool().view(-1)
                ]
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

    def commit(self, layer_name: str, param_name: str, phases: Tensor) -> None:
        layer = self.modules[layer_name]
        if param_name == "phase_U":
            phase_bias = self.untrainable_params[layer_name]["phase_bias_U"]
            delta_list = self.untrainable_params[layer_name]["delta_list_U"]
            quantizer = self.quantizers[layer_name]["phase_U_quantizer"]
            layer.U.data.copy_(
                self.decomposer.reconstruct(
                    delta_list,
                    self.v2m(quantizer(phases.view(phase_bias.size(0), phase_bias.size(1), -1)) + phase_bias),
                )
            )
        elif param_name == "phase_V":
            phase_bias = self.untrainable_params[layer_name]["phase_bias_V"]
            delta_list = self.untrainable_params[layer_name]["delta_list_V"]
            quantizer = self.quantizers[layer_name]["phase_V_quantizer"]

            layer.V.data.copy_(
                self.decomposer.reconstruct(
                    delta_list,
                    self.v2m(quantizer(phases.view(phase_bias.size(0), phase_bias.size(1), -1)) + phase_bias),
                )
            )

        elif param_name == "phase_S":
            layer.S.data.copy_(phases.data.cos().view_as(layer.S).mul_(layer.S_scale))
        else:
            raise ValueError(f"Wrong param_name {param_name}")

    # ============ apply gradients with momentum weight_decay dampening ============
    def _apply_gradients(self, params, grads, lr):
         # ============ SGD ============
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
                        self.commit(layer_name, p_name, p)
                        y, pos_loss = obj_fn()
                        self.forward_counter += 1
                        
                        if self.STP == True:
                            p.data[idx] = neg_perturbed_value
                            self.commit(layer_name, p_name, p)
                            y, neg_loss = obj_fn()
                            self.forward_counter += 1
                            loss_list = [old_loss, pos_loss, neg_loss]
                            grad_list = [0, -1, 1]
                        else:
                            loss_list = [old_loss, pos_loss]
                            grad_list = [1, -1]

                        param_grad[idx] = grad_list[loss_list.index(min(loss_list))]

                        p.data[idx] = old_value
                        self.commit(layer_name, p_name, p)

                layer_grads[p_name] = param_grad

            grads[layer_name] = layer_grads

        self._apply_gradients(params, grads, lr)
        return y, old_loss, grads
    
    def zo_coordinate_descent(self, obj_fn, params):
        """
        description: stochastic coordinate-wise descent.
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
                    # SparseTune in FLOPS+ [Gu+, DAC 2020]
                    cur_seed = get_random_state()
                    set_torch_stochastic()
                    seed = np.random.rand()
                    set_torch_deterministic(cur_seed)
                    if seed < self.grad_sparsity:
                        continue
                    old_value = p.data[idx]
                    pos_perturbed_value = old_value + lr
                    neg_perturbed_value = old_value - lr

                    p.data[idx] = pos_perturbed_value
                    with torch.no_grad():  # training=True to enable profiling, but do not save graph
                        self.commit(layer_name, p_name, p)
                        y, new_loss = obj_fn()
                        self.forward_counter += 1

                    if new_loss < old_loss:
                        old_loss = new_loss
                    else:
                        p.data[idx] = neg_perturbed_value
                        with torch.no_grad():
                            self.commit(layer_name, p_name, p)
                            y, old_loss = obj_fn()
                            self.forward_counter += 1
        return y, old_loss

    def build_obj_fn(self, data, target, model, criterion):
        def _obj_fn():
            y = model(data)
            return y, criterion(y, target)

        return _obj_fn

    def step(self, data, target):
        self.obj_fn = self.build_obj_fn(data, target, self.model, self.criterion)
        y, loss = self.zo_coordinate_descent(self.obj_fn, self.trainable_params)
        # update internal parameters
        self.global_step += 1
        return y, loss
    
    def get_forward_cnt(self):
        return self.forward_counter

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
        y, loss = self.zo_coordinate_descent(self.obj_fn, self.trainable_params)
        # y, loss, grads = self.zo_coordinate_descent_batch(self.obj_fn, self.trainable_params)
        # update internal parameters
        self.global_step += 1
        return y, loss
    
        # advanced

        # if self.grad_estimator == 'sign':
        #     y, loss = self.zo_coordinate_descent_sign(self.obj_fn, self.trainable_params, STP=False)
        #     grads_zo = None
        # elif self.grad_estimator == 'STP':
        #     y, loss = self.zo_coordinate_descent_sign(self.obj_fn, self.trainable_params, STP=True)
        #     grads_zo = None
        # elif self.grad_estimator == 'batch':
        #     y, loss, grads_zo = self.zo_coordinate_descent_batch(self.obj_fn, self.trainable_params)
        # elif self.grad_estimator == 'esti':
        #     y, loss, grads_zo = self.zo_coordinate_descent_esti(self.obj_fn, self.trainable_params)
        # # update internal parameters
        # self.global_step += 1
        # # output gradients
        # if en_debug == True:
        #     grads_fo = self.extract_grad_fo(self.model)
        #     grads_err = self.cal_grad_err(self.trainable_params, grads_zo, grads_fo)
        #     return y, loss, (grads_zo, grads_fo, grads_err)
        # else:
        #     return y, loss, grads_zo
