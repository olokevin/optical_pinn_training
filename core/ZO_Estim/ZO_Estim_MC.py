from typing import Callable

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scipy.stats import qmc
from .ZO_utils import split_model, split_named_model, SplitedLayer, SplitedParam
from .QMC_sampler import sphere_n, coord_basis, block_mask_generator, layer_mask_generator

DEBUG=False
# DEBUG=True

def create_fwd_hook_add_perturbation(perturbation):
    def fwd_hook(module, input, output):
        # input is a tuple
        module.in_value = input[0].detach().clone()
        # output is a tensor
        output += perturbation
    return fwd_hook

class ZO_Estim_MC(nn.Module):
    """
    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
    """
    
    def __init__(
        self,
        model: nn.Module,
        obj_fn_type: str,
        # For param perturb ZO. A list of SplitedParam. Specifies what Tensors should be optimized.
        splited_param_list: list = None,
        # For actv  perturb ZO. A list of SplitedLayer. Specifies what layers' activations should be perturbed.
        splited_layer_list: list = None,

        sigma: float = 0.1,
        n_sample: int = 20,
        signsgd: bool = False,
        
        quantized: bool = False,

        estimate_method: str = 'forward',
        sample_method: str = 'gaussian',
        normalize_perturbation: bool = False,
        
        scale: str = None,

        en_layerwise_perturbation: bool = True,
        en_partial_forward: bool = True,
        en_param_commit: bool = False,

        get_iterable_block_name: Callable = None,
        pre_block_forward:  Callable = None,
        post_block_forward:  Callable = None
        ):
        super().__init__()

        self.model = model
        self.obj_fn_type = obj_fn_type

        self.splited_param_list = splited_param_list
        self.splited_layer_list = splited_layer_list
        
        self.splited_modules_list = split_model(model)
    
        self.sigma = sigma
        self.n_sample = n_sample
        self.signsgd = signsgd

        self.quantized = quantized
        self.estimate_method = estimate_method
        self.sample_method = sample_method
        self.normalize_perturbation = normalize_perturbation
        
        self.scale = scale
        
        self.en_layerwise_perturbation = en_layerwise_perturbation
        self.en_partial_forward = en_partial_forward
        self.en_param_commit = en_param_commit

        self.get_iterable_block_name = get_iterable_block_name
        self.pre_block_forward = pre_block_forward
        self.post_block_forward = post_block_forward

        self.device = next(model.parameters()).device
        
        self.forward_counter = 0
    
    def _init_sampler(self, dimension):
        if self.sample_method == 'sobol':
            sampler = qmc.Sobol(d=dimension, scramble=False)
        elif self.sample_method == 'halton':
            sampler = qmc.Halton(d=dimension, scramble=True)
        elif self.sample_method == 'sphere_n':
            sampler = sphere_n(n=dimension)
        elif self.sample_method == 'coord_basis':
            sampler = coord_basis(dimension=dimension)
        else:
            sampler = None
        return sampler
            
    ### Generate random vectors from a normal distribution
    def _generate_random_vector(self, shape, sample_method, device):
        
        dimension = np.prod(shape)

        if self.quantized == True:
            if sample_method == 'bernoulli':
                sample = torch.ones(shape, device=device) - 2*torch.bernoulli(0.5*torch.ones(shape, device=device))
            else:
                return NotImplementedError('Unlnown sample method', self.sample_method)
        else:
            if sample_method == 'uniform':
                sample = torch.randn(shape, device=device)
                sample = torch.nn.functional.normalize(sample, p=2, dim=0)
            elif sample_method == 'gaussian':
                sample = torch.randn(shape, device=device)
                # sample = torch.randn(shape, device=device) / dimension
            elif sample_method == 'bernoulli':
                ### Rademacher
                sample = torch.ones(shape, device=device) - 2*torch.bernoulli(0.5*torch.ones(shape, device=device))
            elif sample_method in ('sobol', 'halton'):
                if self.sampler == None:
                    raise ValueError('Need sampler input')
                else:
                    sample = torch.Tensor(self.sampler.random(1)).squeeze()
                    sample = 2*sample-torch.ones_like(sample)
                    sample = torch.nn.functional.normalize(sample, p=2, dim=0)
                    sample = sample.to(device)
            elif sample_method == 'sphere_n':
                sample = next(self.sampler)
                sample = sample.to(device)
            else:
                return NotImplementedError('Unlnown sample method', sample_method)
            
        return sample
    
    def get_single_param_ZO_gradient(self, splited_param, block_in, old_loss, sigma, estimate_method, sample_method):
        idx = splited_param.idx
        param = splited_param.param

        param_dim = param.numel()
        param_shape = param.shape
        param_vec = param.view(-1)
        old_param_vec = param_vec.clone().detach()

        param_grad_vec = torch.zeros_like(param_vec, device=self.device)
        
        if self.sample_method == 'coord_basis':
            for idx in range(param_dim):
                param_vec[idx].data.add_(sigma)
                if self.en_param_commit:
                    # if hasattr(splited_param.layer, 'commit_all'):
                    #     splited_param.layer.commit_all()
                    if hasattr(splited_param, 'commit_fn'):
                        splited_param.commit_fn()
                _, pos_loss = self.obj_fn()
                self.forward_counter += 1
                
                if estimate_method == 'forward':
                    param_vec[idx].data.copy_(old_param_vec[idx].data)
                    if self.en_param_commit:
                        # if hasattr(splited_param.layer, 'commit_all'):
                        #     splited_param.layer.commit_all()
                        if hasattr(splited_param, 'commit_fn'):
                            splited_param.commit_fn()
                    param_grad_vec[idx] = (pos_loss - old_loss) / sigma
                elif estimate_method == 'antithetic':
                    param_vec[idx].data.copy_(old_param_vec[idx].data)
                    param_vec[idx].data.sub_(sigma)
                    if self.en_param_commit:
                        # if hasattr(splited_param.layer, 'commit_all'):
                        #     splited_param.layer.commit_all()
                        if hasattr(splited_param, 'commit_fn'):
                            splited_param.commit_fn()
                    _, neg_loss = self.obj_fn()
                    self.forward_counter += 1
                    
                    param_vec[idx].data.copy_(old_param_vec[idx].data)
                    if self.en_param_commit:
                        # if hasattr(splited_param.layer, 'commit_all'):
                        #     splited_param.layer.commit_all()
                        if hasattr(splited_param, 'commit_fn'):
                            splited_param.commit_fn()
                    param_grad_vec[idx] = (pos_loss - neg_loss) / 2 / sigma
        else:
            splited_param.old_param = splited_param.param.clone().detach()
            
            for i in range(self.n_sample):
                ### Generate random perturbation with the same shape as the parameter
                if sample_method == 'coord_basis':
                    u = torch.zeros(param_dim, device=self.device)
                    u[i] = 1
                else:
                    u = self._generate_random_vector(param_vec.shape, sample_method, self.device)

                if self.normalize_perturbation:
                    p_sigma = sigma / torch.linalg.vector_norm(u)
                else:
                    p_sigma = sigma
                
                ### Add perturbation to the parameter
                # pos
                param_vec.add_(u * p_sigma)
                if self.en_param_commit:
                    # if hasattr(splited_param.layer, 'commit_all'):
                    #     splited_param.layer.commit_all()
                    if hasattr(splited_param, 'commit_fn'):
                        splited_param.commit_fn()
                if block_in is not None:
                    _, pos_loss = self.obj_fn(starting_idx=idx, input=block_in, return_loss_reduction='mean')
                else:
                    _, pos_loss = self.obj_fn()
                
                self.forward_counter += 1

                ### Estimate gradient
                if estimate_method == 'forward':
                    param_vec.data.copy_(old_param_vec)
                    if self.en_param_commit:
                      # if hasattr(splited_param.layer, 'commit_all'):
                      #     splited_param.layer.commit_all()
                      if hasattr(splited_param, 'commit_fn'):
                          splited_param.commit_fn()

                    param_grad_vec += (pos_loss - old_loss) / sigma * u
                elif estimate_method == 'antithetic':
                    param_vec.data.copy_(old_param_vec)
                    param_vec.sub_(u * p_sigma)
                    if self.en_param_commit:
                      # if hasattr(splited_param.layer, 'commit_all'):
                      #     splited_param.layer.commit_all()
                      if hasattr(splited_param, 'commit_fn'):
                          splited_param.commit_fn()
                        
                    if block_in is not None:
                        _, neg_loss = self.obj_fn(starting_idx=idx, input=block_in, return_loss_reduction='mean')
                    else:
                        _, neg_loss = self.obj_fn()
                    
                    self.forward_counter += 1
                    
                    param_vec.data.copy_(old_param_vec)
                    if self.en_param_commit:
                      # if hasattr(splited_param.layer, 'commit_all'):
                      #     splited_param.layer.commit_all()
                      if hasattr(splited_param, 'commit_fn'):
                          splited_param.commit_fn()

                    param_grad_vec += (pos_loss - neg_loss) / 2 / sigma * u
              
                param_grad_vec = param_grad_vec / self.n_sample
              
                if self.signsgd is True:
                    param_grad_vec = torch.sign(param_grad_vec)
                ### No scaling
                if self.scale is None:
                    pass
                elif self.scale == 'sqrt_dim':
                    param_grad_vec = param_grad_vec * math.sqrt(self.n_sample / (self.n_sample+param_dim-1))
                elif self.scale == 'dim':
                    param_grad_vec = param_grad_vec * (self.n_sample / (self.n_sample+param_dim-1))

        param_ZO_grad = param_grad_vec.view(param_shape)
        return param_ZO_grad
    
    def get_all_param_ZO_gradient(self, old_loss, sigma, estimate_method, sample_method):
        dimension = 0
        for splited_param in self.splited_param_list:
            dimension += splited_param.param.numel()
            splited_param.param.grad = torch.zeros_like(splited_param.param)
        
        if self.sample_method == 'coord_basis':
            for splited_param in self.splited_param_list:
                param_dim = splited_param.param.numel()
                param_vec = splited_param.param.view(-1)
                param_grad_vec = torch.zeros_like(param_vec)
                for idx in range(param_dim):
                    param_vec[idx].data.add_(sigma)
                    if self.en_param_commit:
                        if 'voltage' in splited_param.name:
                            splited_param.layer.commit_coordinate(idx)
                    _, pos_loss = self.obj_fn()
                    self.forward_counter += 1
                    
                    if estimate_method == 'forward':
                        param_vec[idx].data.sub_(sigma)
                        if self.en_param_commit:
                            if 'voltage' in splited_param.name:
                                splited_param.layer.commit_coordinate(idx)
                        param_grad_vec[idx] = (pos_loss - old_loss) / sigma
                    elif estimate_method == 'antithetic':
                        param_vec[idx].data.sub_(2*sigma)
                        if self.en_param_commit:
                            if 'voltage' in splited_param.name:
                                splited_param.layer.commit_coordinate(idx)
                        _, neg_loss = self.obj_fn()
                        self.forward_counter += 1
                        param_vec[idx].data.add_(sigma)
                        if self.en_param_commit:
                            if 'voltage' in splited_param.name:
                                splited_param.layer.commit_coordinate(idx)
                        param_grad_vec[idx] = (pos_loss - neg_loss) / 2 / sigma
                
                splited_param.param.grad = param_grad_vec.reshape(splited_param.param.shape)

        else:
            n_sample = self.n_sample
            for splited_param in self.splited_param_list:
                splited_param.old_param = splited_param.param.clone().detach()
            
            for i in range(n_sample):
                ### Generate random perturbation with the same shape as the parameter
                for splited_param in self.splited_param_list:
                    splited_param.u = self._generate_random_vector(splited_param.param.shape, sample_method, self.device)
                
                if self.normalize_perturbation:
                    p_sigma = sigma / torch.linalg.vector_norm(torch.cat([splited_param.u.view(-1) for splited_param in self.splited_param_list]))
                else:
                    p_sigma = sigma
                
                ### Add perturbation to the parameter
                # pos
                for splited_param in self.splited_param_list:
                    splited_param.param.add_(splited_param.u * p_sigma)
                    if self.en_param_commit:
                        if hasattr(splited_param.layer, 'commit_all'):
                            splited_param.layer.commit_all()  
                        # if 'voltage' in splited_param.name:
                        #     splited_param.layer.commit_all()
                    
                _, pos_loss = self.obj_fn()
                self.forward_counter += 1

                ### Estimate gradient
                if estimate_method == 'forward':
                    for splited_param in self.splited_param_list:
                        splited_param.param.copy_(splited_param.old_param)
                        if self.en_param_commit:
                            if hasattr(splited_param.layer, 'commit_all'):
                                splited_param.layer.commit_all()  
                            # if 'voltage' in splited_param.name:
                            #     splited_param.layer.commit_all()
                        
                        splited_param.param.grad += (pos_loss - old_loss) / sigma / n_sample * splited_param.u
                elif estimate_method == 'antithetic':
                    for splited_param in self.splited_param_list:
                        splited_param.param.copy_(splited_param.old_param)
                        splited_param.param.sub_(splited_param.u * p_sigma)

                        if self.en_param_commit:
                            if hasattr(splited_param.layer, 'commit_all'):
                                splited_param.layer.commit_all()  
                            # if 'voltage' in splited_param.name:
                            #     splited_param.layer.commit_all()
                                
                    _, neg_loss = self.obj_fn()
                    self.forward_counter += 1
                    
                    for splited_param in self.splited_param_list:
                        splited_param.param.copy_(splited_param.old_param)
                        if self.en_param_commit:
                            if hasattr(splited_param.layer, 'commit_all'):
                                splited_param.layer.commit_all()  
                            # if 'voltage' in splited_param.name:
                            #     splited_param.layer.commit_all()
                        
                        splited_param.param.grad += (pos_loss - neg_loss) / 2 / sigma / n_sample * splited_param.u
                        
            for splited_param in self.splited_param_list:
                if self.signsgd is True:
                    splited_param.param.grad = torch.sign(splited_param.param.grad)
                ### No scaling
                if self.scale is None:
                    pass
                elif self.scale == 'sqrt_dim':
                    splited_param.param.grad *= math.sqrt(n_sample / (n_sample+dimension-1))
                elif self.scale == 'dim':
                    splited_param.param.grad *= (n_sample / (n_sample+dimension-1))
        
        return None
    
    def get_param_ZO_gradient(self, old_loss):
        if self.en_layerwise_perturbation:
            
            for splited_param in self.splited_param_list:
                if self.en_partial_forward:
                    block_in = self.obj_fn(ending_idx=splited_param.idx, return_loss_reduction='no_loss')
                else:
                    block_in = None
                ### TODO: could further specify sigma, estimate_method, sample_method for different params
                param_ZO_grad = self.get_single_param_ZO_gradient(splited_param, block_in, old_loss, self.sigma, self.estimate_method, self.sample_method)
                    
                splited_param.param.grad = param_ZO_grad

        else:
            self.get_all_param_ZO_gradient(old_loss, self.sigma, self.estimate_method, self.sample_method)

        
    
    def get_actv_ZO_gradient(self):

        if self.estimate_method == 'forward':
            _, old_loss = self.obj_fn(return_loss_reduction='none')

        ### Generate random perturbation with the same shape as the parameter
        ### Add perturbation to the parameter
        ### Estimate gradient

        for splited_layer in self.splited_layer_list:
            block_in = self.obj_fn(ending_idx=splited_layer.idx, return_loss_reduction='no_loss')

            post_actv_shape = splited_layer.layer.output_shape
            batch_sz = post_actv_shape[0]
            mask = torch.ones(post_actv_shape, device=self.device)

            ZO_grad = torch.zeros(post_actv_shape, device=self.device)

            actv_dim = np.prod(post_actv_shape[1:])
            feature_shape = post_actv_shape[1:]
            if self.sample_method == 'coord_basis':
                n_sample = actv_dim
            else:
                n_sample = self.n_sample
            
            for i in range(n_sample):
                
                ### Generate random perturbation with the same shape as the parameter
                if self.sample_method == 'coord_basis':
                    # if hasattr(self, 'sync_batch_perturb') and self.sync_batch_perturb:
                    #     u = torch.zeros(actv_dim, device=self.device)
                    #     u[i] = 1
                    #     u.reshape(feature_shape)
                    #     u = mask * torch.tile(u, (batch_sz, 1))
                    # else:
                    #     raise NotImplementedError
                    
                    u = torch.zeros(actv_dim, device=self.device)
                    u[i] = 1
                    u.reshape(feature_shape)
                    u = mask * torch.tile(u, (batch_sz, 1))
                        
                else:
                    if hasattr(self, 'sync_batch_perturb') and self.sync_batch_perturb:
                        u = mask * torch.tile(self._generate_random_vector(feature_shape, self.sample_method, self.device).unsqueeze(0), (batch_sz, 1))
                    else:
                        u = mask * self._generate_random_vector(post_actv_shape, self.sample_method, self.device)    

                ### Add perturbation to the parameter
                # splited_layer.layer.en_perturb_forward(u * self.sigma)
                fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(u * self.sigma)
                fwd_hook_handle = splited_layer.layer.register_forward_hook(fwd_hook_add_perturbation)
  
                _, pos_loss = self.obj_fn(starting_idx=splited_layer.idx, input=block_in, return_loss_reduction='none')

                fwd_hook_handle.remove()
                self.forward_counter += 1

                ### Estimate gradient
                if self.estimate_method == 'forward':
                    if batch_sz == 1:
                        ZO_grad += (pos_loss - old_loss) / self.sigma * u
                    else:
                        ZO_grad += torch.einsum('i,i...->i...', ((pos_loss - old_loss) / self.sigma, u))

                elif self.estimate_method == 'antithetic':
                    ### Add perturbation to the parameter
                    # splited_layer.layer.en_perturb_forward( - u * self.sigma)
                    fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(- u * self.sigma)
                    fwd_hook_handle = splited_layer.layer.register_forward_hook(fwd_hook_add_perturbation)

                    _, neg_loss = self.obj_fn(starting_idx=splited_layer.idx, input=block_in, return_loss_reduction='none')

                    fwd_hook_handle.remove()
                    self.forward_counter += 1
                    
                    if batch_sz == 1:
                        ZO_grad += (pos_loss - neg_loss) / 2.0 / self.sigma * u
                    else:
                        ZO_grad += torch.einsum('i,i...->i...', ((pos_loss - neg_loss) / 2.0 / self.sigma, u))

            if self.sample_method == 'coord_basis':
                ZO_grad = (ZO_grad / batch_sz).view(post_actv_shape)
            else:
                ZO_grad = (ZO_grad / self.n_sample / batch_sz).view(post_actv_shape)

            ### Apply estimated gradient
            if type(splited_layer.layer) == nn.Linear:
                splited_layer.layer.weight.grad = torch.matmul(ZO_grad.T, splited_layer.layer.in_value)
                splited_layer.layer.bias.grad = torch.sum(ZO_grad, dim=0)
            elif type(splited_layer.layer) == nn.Conv2d:
                pass
                # splited_layer.layer.weight.grad = torch.nn.grad.conv2d_weight(splited_layer.layer.in_value, splited_layer.layer.weight.shape, ZO_grad,
                #                                     stride=splited_layer.layer.stride, padding=splited_layer.layer.padding,
                #                                     dilation=splited_layer.layer.dilation, groups=splited_layer.layer.groups)
                # splited_layer.layer.bias.grad = ZO_grad.sum([0, 2, 3])
            else:
                splited_layer.layer.local_backward(ZO_grad) 

            ### Apply true FO_grad_output
            # FO_grad = splited_layer.layer.adapter_down.out_grad[0]
            # splited_layer.layer.local_backward(FO_grad)
        
            ### Apply estimated gradient
            # splited_layer.layer.local_backward(ZO_grad)     
            # splited_layer.layer.disable_perturb_forward() 
        


    def update_obj_fn(self, obj_fn):
        self.obj_fn = obj_fn
    
    def get_forward_cnt(self):
        return self.forward_counter
    
    def estimate_grad(self, old_loss):
        
        # self.model.zero_grad()
        if self.splited_layer_list is not None:
            self.get_actv_ZO_gradient()
        
        if self.splited_param_list is not None:
            self.get_param_ZO_gradient(old_loss=old_loss)
        
        return None