from ctypes import Union
import math
from re import M
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .low_rank_tensors import CP,TensorTrain,TensorTrainMatrix,Tucker
from .emb_utils import get_cum_prod,tensorized_lookup
import tensorly as tl
from . import low_rank_tensors


# from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize
# from ..common_types import _size_1_t, _size_2_t, _size_3_t


class TensorizedLinear_linear(nn.Linear):
    def __init__(self,
                in_features,
                out_features,
                bias=False,
                init=None,
                shape=None,
                tensor_type='TensorTrainMatrix',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
    ):

        super(TensorizedLinear_linear,self).__init__(in_features,out_features,bias,device,dtype)

        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(2/self.in_features)

        #shape taken care of at input time
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

    # def forward(self, input, rank_update=True):
    def forward(self, input, rank_update=False):

        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        
        return F.linear(input,self.tensor.get_full().reshape([self.out_features,self.in_features]),self.bias)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()



class TensorizedLinear_module(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                tensor_type='TensorTrainMatrix',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
    ):

        super(TensorizedLinear_module,self).__init__()

        self.tensor_type = tensor_type

        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(1/self.in_features)

        #shape taken care of at input time
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

        if bias == False:
            self.bias = 0
        else:
            stdv = 1. / math.sqrt(out_features)
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
            # self.bias.data.uniform_(-stdv, stdv)
            # self.bias = torch.nn.Parameter(torch.randn(out_features))
            # self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, ranks = None,scale = None, reconstruct = False,rank_update=False):

        if self.training and rank_update:
            self.tensor.update_rank_parameters()

        self.x = input
        # print(self.tensor.get_full().shape)
        if reconstruct:
            # print(input.device)
            # print(self.tensor.get_full().device)
            return F.linear(input,self.tensor.get_full().reshape([self.out_features,self.in_features]).to(input.device),bias=self.bias)
        else:
            if self.tensor_type == 'TensorTrainMatrix':
                output = self.forward_ttm(input) + self.bias
            elif self.tensor_type == 'TensorTrain':
                output = self.forward_tt(input,target_rank=ranks,scale=scale) 
                if self.bias!=None:
                    output += self.bias
            return output
        

    def forward_tt(self,input,target_rank=None, scale = None):
        if target_rank==None:
            ranks = [U.shape[0] for U in self.tensor.factors] + [1]
        elif isinstance(target_rank, list):
            ranks = target_rank
        else:
            ranks = [1] + [target_rank]*(len(self.tensor.factors)-1) + [1]

        if scale==None:
            scale = 1.0



        m = len(self.tensor.factors)//2
        
 
        if len(input.shape)==2:
            mat_shape = [input.shape[0]] + [U.shape[1] for U in self.tensor.factors[0:m]]
            N=2
        elif len(input.shape)==3:
            N=3
            mat_shape = [input.shape[0]]+[input.shape[1]] + [U.shape[1] for U in self.tensor.factors[0:m]]
        input = torch.reshape(input, [1] + mat_shape)

        out = scale*self.tensor.factors[0]
        out[:ranks[0],:,:ranks[1]] = self.tensor.factors[0][:ranks[0],:,:ranks[1]]
        out = torch.squeeze(out)

        for i in range(1,m):
            U = scale*self.tensor.factors[i]
            U[:ranks[i],:,:ranks[i+1]] = self.tensor.factors[i][:ranks[i],:,:ranks[i+1]]
            out = torch.tensordot(out, U, [[-1],[0]])

        out = torch.tensordot(input, out, [list(range(N,N+m)), list(range(0,m))])
        out = [out] + self.tensor.factors[m:]


        N = len(out[0].shape)
        output = scale*self.tensor.factors[m]
        output[:ranks[m],:,:ranks[m+1]] = self.tensor.factors[m][:ranks[m],:,:ranks[m+1]]
        # output = self.tensor.factors[m][:ranks[m],:,:ranks[m+1]]

        for i in range(m+1,2*m):
            U = scale*self.tensor.factors[i]
            U[:ranks[i],:,:ranks[i+1]] = self.tensor.factors[i][:ranks[i],:,:ranks[i+1]]
            output = torch.tensordot(output,U,[[-1],[0]])
        
        output = torch.tensordot(out[0],output,[[-1],[0]])

        output = torch.flatten(output, start_dim = N-1, end_dim = -1)
        output = torch.squeeze(output)

        return output

            

    # def forward(self, input, reconstruct = False,rank_update=False):

    #     if self.training and rank_update:
    #         self.tensor.update_rank_parameters()

    #     self.x = input
    #     # print(self.tensor.get_full().shape)
    #     if reconstruct:
    #         # print(input.device)
    #         # print(self.tensor.get_full().device)
    #         return F.linear(input,self.tensor.get_full().reshape([self.out_features,self.in_features]).to(input.device),bias=self.bias)
    #     else:
    #         if self.tensor_type == 'TensorTrainMatrix':
    #             output = self.forward_ttm(input) + self.bias
    #         elif self.tensor_type == 'TensorTrain':
    #             output = self.forward_tt(input) 
    #             if self.bias!=None:
    #                 output += self.bias
    #         return output
        

    # def forward_tt(self,input):
    #     m = len(self.tensor.factors)//2
        
 
    #     if len(input.shape)==2:
    #         mat_shape = [input.shape[0]] + [U.shape[1] for U in self.tensor.factors[0:m]]
    #         N=2
    #     elif len(input.shape)==3:
    #         N=3
    #         mat_shape = [input.shape[0]]+[input.shape[1]] + [U.shape[1] for U in self.tensor.factors[0:m]]
    #     input = torch.reshape(input, [1] + mat_shape)

    #     out = torch.squeeze(self.tensor.factors[0])

    #     for U in self.tensor.factors[1:m]:
    #         out = torch.tensordot(out, U, [[-1],[0]])

    #     out = torch.tensordot(input, out, [list(range(N,N+m)), list(range(0,m))])
    #     out = [out] + self.tensor.factors[m:]


    #     N = len(out[0].shape)
    #     output = out[1]

    #     for U in out[2:]:
    #         output = torch.tensordot(output,U,[[-1],[0]])
        
    #     output = torch.tensordot(out[0],output,[[-1],[0]])

    #     output = torch.flatten(output, start_dim = N-1, end_dim = -1)
    #     output = torch.squeeze(output)

    #     return output



    def forward_ttm(self,matrix):
        # Prepare tensor shape
        mat_shape = list(matrix.shape)
        if len(mat_shape) == 3:
            N = 2
        else:
            N = 1

        tensor = self.tensor

        tensorized_shape_x, tensorized_shape_y = tensor.dims1, tensor.dims2
        # print(tensorized_shape_x,tensorized_shape_y)
        output = matrix.reshape(mat_shape[0:N] + tensorized_shape_x + [1])
        
        order = len(tensorized_shape_x)

        for i in range(order):
            output = torch.tensordot(output,tensor.factors[i],[[N,-1],[1,0]])

        output = torch.flatten(output,start_dim = N, end_dim = -1)

        return output


    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()

# ================== TONN TensorizedLinear_module ==================
class TensorizedLinear_module_tonn(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                tensor_type='TensorTrainMatrix',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
    ):

        super(TensorizedLinear_module_tonn,self).__init__()

        if tensor_type != 'TensorTrainMatrix':
            raise (ValueError("Only support TensorTrainMatrix now"))
        self.tensor_type = tensor_type

        self.in_features = in_features
        self.out_features = out_features
        self.in_shape = shape[0]
        self.out_shape = shape[1]
        self.order = len(shape[0])

        if type(max_rank)==int:
            self.tt_rank = [1]+(len(self.in_shape)-1)*[max_rank]+[1]
        elif type(max_rank)==list:
            if len(max_rank) == self.order+1:
                self.tt_rank = max_rank
            else:
                raise (ValueError("Rank is a list, but length does not match"))
        else:
            raise (ValueError("Rank is neither a list nor int"))

        if len(self.in_shape) != len(self.out_shape):
            raise (ValueError("the input dim and output dim should be the same"))
        
        if math.prod(self.in_shape) != in_features or math.prod(self.out_shape) != out_features:
            raise (ValueError("The decomposition shape does not match input/output channel"))

        if bias == False:
            self.bias = 0
        else:
            stdv = 1. / math.sqrt(out_features)
            # self.bias = torch.nn.Parameter(torch.zeros(out_features))
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)
        
        self.tt_cores = nn.ModuleList()

        for i in range(self.order):
            # tt_core[i]: r[i]*m[i]*n[i]*r[i+1]
            
            out_tt_core = self.tt_rank[i]  * self.out_shape[i]
            in_tt_core  = self.in_shape[i] * self.tt_rank[i+1]
            self.tt_cores.append(
                nn.Linear(in_tt_core, out_tt_core, bias=False)
            )
    
    def forward(self, x, ranks = None,scale = None,reconstruct = False,rank_update=False,return_factors = False,prune=False):
    # def forward(self, x):
        x_shape = tuple(x.shape)
        batch_sz = x_shape[0:-1]
        new_shape = x_shape[0:-1] + tuple(self.in_shape)

        # x: decomposed as [bz,N_1,N_2,...,N_d]
        output = torch.reshape(x,new_shape).contiguous()
        i_tp = -3

        for i in reversed(range(self.order)):
            # [bz,N1,N2,...,Nd] -> [bzN1N2...Nd-1,Nd]
            temp_shape = output.shape[:-1]
            # output = torch.flatten(output, 0, -2)
            # print('1',output.shape)
            # [bzN1N2...Nd-1,Nd] * [NdRd+1 * RdMd]
            # output = torch.matmul(output, self.tt_cores[i].T)
            output = self.tt_cores[i](output)
            # print('2',output.shape)
            # [bzN1N2...Nd-1,RdMd] -> [bz,N1,N2,...Nd-1,Rd,Md]
            # output = torch.unflatten(output, 0, temp_shape)
            output = torch.unflatten(output, -1, (self.tt_rank[i],self.out_shape[i]))
            # print('3',output.shape)
            # [bz,N1,N2,...Nd-1,Rd,Md] -> [bz,N1,N2,...,Md,Nd-1,Rd]
            output = torch.transpose(torch.transpose(output,-1,-2), -2, i_tp)
            # print('4',output.shape)
            # [bz,N1,N2,...,Md,Nd-1,Rd] -> [bz,N1,N2,...,Md,Nd-1Rd]
            output = torch.flatten(output, -2, -1)
            # print('5',output.shape)
            i_tp = i_tp - 1

        # permute batch_sz back (batch is not only at dim 0)
        t_dim = len(batch_sz)-1
        output = torch.unsqueeze(output, dim=t_dim)
        output = torch.transpose(output, t_dim, -1)
        output = torch.reshape(output, batch_sz+(-1,))
        
        # # permute batch_sz to the dim 0
        # output = torch.permute(output, (-1,)+tuple(range(self.order))).contiguous()
        # # print('end',output.shape)
        # output = torch.reshape(output, (batch_sz, -1))
            
        if self.bias != None:
            output = output + self.bias

        return output


class TensorizedLinear_module_prune(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                tensor_type='TensorTrainMatrix',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
    ):

        super(TensorizedLinear_module_prune,self).__init__()

        self.tensor_type = tensor_type

        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(1/self.in_features)

        #shape taken care of at input time
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

        if bias == False:
            self.bias = 0
        else:
            stdv = 1. / math.sqrt(out_features)
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
            # self.bias.data.uniform_(-stdv, stdv)
            # self.bias = torch.nn.Parameter(torch.randn(out_features))
            # self.bias.data.uniform_(-stdv, stdv)
        

    def forward(self, input, target_rank = None,scale = None, reconstruct = False,rank_update=False):

        if self.training and rank_update:
            self.tensor.update_rank_parameters()

        self.x = input
        # print(self.tensor.get_full().shape)
        if reconstruct:
            # print(input.device)
            # print(self.tensor.get_full().device)
            return F.linear(input,self.tensor.get_full().reshape([self.out_features,self.in_features]).to(input.device),bias=self.bias)
        else:
            if self.tensor_type == 'TensorTrainMatrix':
                output = self.forward_ttm(input) + self.bias
            elif self.tensor_type == 'TensorTrain':
                output = self.forward_tt(input,target_rank=target_rank,scale=scale) 
                if self.bias!=None:
                    output += self.bias
            return output
        

    def forward_tt(self,input,target_rank=None, scale = None):
        if target_rank==None:
            ranks = [U.shape[0] for U in self.tensor.factors] + [1]
        elif isinstance(target_rank, list):
            ranks = target_rank
        else:
            ranks = [1] + [target_rank]*(len(self.tensor.factors)-1) + [1]

        if scale==None:
            scale = 1.0



        m = len(self.tensor.factors)//2
        
 
        if len(input.shape)==2:
            mat_shape = [input.shape[0]] + [U.shape[1] for U in self.tensor.factors[0:m]]
            N=2
        elif len(input.shape)==3:
            N=3
            mat_shape = [input.shape[0]]+[input.shape[1]] + [U.shape[1] for U in self.tensor.factors[0:m]]
        input = torch.reshape(input, [1] + mat_shape)

        out = scale*self.tensor.factors[0]
        out[:ranks[0],:,:ranks[1]] = self.tensor.factors[0][:ranks[0],:,:ranks[1]]
        out = torch.squeeze(out)

        for i in range(1,m):
            U = scale*self.tensor.factors[i]
            U[:ranks[i],:,:ranks[i+1]] = self.tensor.factors[i][:ranks[i],:,:ranks[i+1]]
            out = torch.tensordot(out, U, [[-1],[0]])

        out = torch.tensordot(input, out, [list(range(N,N+m)), list(range(0,m))])
        out = [out] + self.tensor.factors[m:]


        N = len(out[0].shape)
        output = scale*self.tensor.factors[m]
        output[:ranks[m],:,:ranks[m+1]] = self.tensor.factors[m][:ranks[m],:,:ranks[m+1]]
        # output = self.tensor.factors[m][:ranks[m],:,:ranks[m+1]]

        for i in range(m+1,2*m):
            U = scale*self.tensor.factors[i]
            U[:ranks[i],:,:ranks[i+1]] = self.tensor.factors[i][:ranks[i],:,:ranks[i+1]]
            output = torch.tensordot(output,U,[[-1],[0]])
        
        output = torch.tensordot(out[0],output,[[-1],[0]])

        output = torch.flatten(output, start_dim = N-1, end_dim = -1)
        output = torch.squeeze(output)

        return output



    def forward_ttm(self,matrix):
        # Prepare tensor shape
        mat_shape = list(matrix.shape)
        if len(mat_shape) == 3:
            N = 2
        else:
            N = 1

        tensor = self.tensor

        tensorized_shape_x, tensorized_shape_y = tensor.dims1, tensor.dims2
        # print(tensorized_shape_x,tensorized_shape_y)
        output = matrix.reshape(mat_shape[0:N] + tensorized_shape_x + [1])
        
        order = len(tensorized_shape_x)

        for i in range(order):
            output = torch.tensordot(output,tensor.factors[i],[[N,-1],[1,0]])

        output = torch.flatten(output,start_dim = N, end_dim = -1)

        return output


    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()

class TensorizedEmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 tensor_type='TensorTrainMatrix',
                 max_rank=16,
                 em_stepsize=1.0,
                 prior_type='log_uniform',
                 eta = None,
                 batch_dim_last=None,
                 padding_idx=None,
                 naive=False):

        super(TensorizedEmbedding,self).__init__()

        self.shape = shape
        self.tensor_type=tensor_type

        target_stddev = np.sqrt(1/(np.prod(self.shape[0])+np.prod(self.shape[1])))

        if self.tensor_type=='TensorTrainMatrix':
            tensor_shape = shape
        else:
            # tensor_shape = shape
            tensor_shape = self.shape[0]+self.shape[1]
        

        self.tensor = getattr(low_rank_tensors,self.tensor_type)(tensor_shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

        self.parameters = self.tensor.parameters

        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))

        self.voc_quant = np.prod(self.shape[0])
        self.emb_quant = np.prod(self.shape[1])

        self.padding_idx = padding_idx
        self.naive = naive

        self.cum_prod = get_cum_prod(shape)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()


    def forward(self, x,rank_update=True):

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        # x = x.view(-1)
        x = torch.flatten(x)
        #x_ind = self.ind2sub(x)

#        full = self.tensor.get_full()
#        full = torch.reshape(full,[self.voc_quant,self.emb_quant])
#        rows = full[x]
        if hasattr(self.tensor,"masks"):
            rows = tensorized_lookup(x,self.tensor.get_masked_factors(),self.cum_prod,self.shape,self.tensor_type)
        else:
            rows = tensorized_lookup(x,self.tensor.factors,self.cum_prod,self.shape,self.tensor_type)
#        rows = gather_rows(self.tensor, x_ind)

        

        rows = rows.view(x.shape[0], -1)

        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))

        rows = rows.view(*xshape_new)

        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        
        # print(rows[:,0:2,0:2])
        return rows.to(x.device)

# #---------------------Define Scale Layer-------------------------------------------------------
# class scale(torch.autograd.Function):
#     """
#     We can implement our own custom autograd Functions by subclassing
#     torch.autograd.Function and implementing the forward and backward passes
#     which operate on Tensors.
#     """

#     @staticmethod
#     def forward(ctx, input, scale, bit, half = False):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
#         if half:
#             max_q = 2.0**(bit)-1.0
#             min_q = 0
#             quant = lambda x : fixed_point_quantize(x, wl=bit+1, fl=0, rounding="nearest")
#             # quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="stochastic")

#         else:
#             max_q = 2.0**(bit-1)-1.0
#             min_q = -2.0**(bit-1)
#             quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")
#             # quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="stochastic")


#         ctx.save_for_backward(input, scale)
#         ctx.quant = quant
#         ctx.input_div_scale = input/scale
#         ctx.q_input = quant(ctx.input_div_scale)
#         ctx.min_q = torch.tensor(min_q)
#         ctx.max_q = torch.tensor(max_q)
#         return scale * ctx.q_input

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         """
#         input, scale= ctx.saved_tensors
#         grad_input = grad_output*torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), 1.0, 0.0)
        
#         grad_scale = (torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), ctx.q_input - ctx.input_div_scale, ctx.input_div_scale))

#         grad_scale = grad_output*torch.clamp(grad_scale, min = ctx.min_q, max = ctx.max_q)

#         return grad_input, grad_scale, None, None, None




# class ScaleLayer(nn.Module):

#    def __init__(self, scale=2**(-5), bit = 8, half = True):
#        super().__init__()
#        self.scale = nn.Parameter(torch.FloatTensor([scale]))

#        self.bit = bit
#        self.half = half

#     #    max_q = 2.0**(bit-1)-1.0
#     #    min_q = -2.0**(bit-1)
#     #    quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

#     #    self.quant = quant
#     #    self.min_q = min_q
#     #    self.max_q = max_q

#    def forward(self, input):
#        return scale.apply(input,self.scale,self.bit,self.half)


# class Quantized_Linear(nn.Linear):
#     def __init__(self,
#                 in_features,
#                 out_features,
#                 bias=True,
#                 init=None,
#                 shape=None,
#                 eta = None,
#                 device=None,
#                 dtype=None,
#                 bit = 8,
#                 scale_w = 2**(-5),
#                 scale_b = 2**(-5)
#     ):

#         super(Quantized_Linear,self).__init__(in_features,out_features,bias,device,dtype)

#         self.in_features = in_features
#         self.out_features = out_features

#         self.bit = bit

#         # self.max_q = 2.0**(bit-1)-1.0
#         # self.min_q = -2.0**(bit-1)
#         # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

#         self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
#         self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))
       

#     def forward(self, input):
#         self.weight = scale.apply(self.weight,self.scale_w,self.bit)
#         self.bias = scale.apply(self.bias,self.scale_b,self.bit)
        
#         return F.linear(input,self.weight,self.bias)



# class Q_TensorizedLinear(nn.Linear):
#     def __init__(self,
#                 in_features,
#                 out_features,
#                 bias=True,
#                 init=None,
#                 shape=None,
#                 tensor_type='TensorTrainMatrix',
#                 max_rank=20,
#                 em_stepsize=1.0,
#                 prior_type='log_uniform',
#                 eta = None,
#                 device=None,
#                 dtype=None,
#                 bit_w = 8,
#                 bit_b = 8,
#                 scale_w = 2**(-5),
#                 scale_b = 2**(-5),
#                 Q_back = False,
#                 recur = False,
#     ):

#         super(Q_TensorizedLinear,self).__init__(in_features,out_features,bias,device,dtype)

#         self.Q_back = Q_back
#         self.recur = recur

#         self.in_features = in_features
#         self.out_features = out_features
#         target_stddev = np.sqrt(2/self.in_features)

#         self.bit_w = bit_w
#         self.bit_b = bit_b

#         #shape taken care of at input time
#         self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

#         # self.max_q = 2.0**(bit-1)-1.0
#         # self.min_q = -2.0**(bit-1)
#         # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

#         self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
#         self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))


#     def forward(self, input, rank_update=True):
        

#         if self.training and rank_update:
#             self.tensor.update_rank_parameters()
        
#         Q_factors = []        
#         for U in self.tensor.factors:
#             Q_factors.append(scale.apply(U,self.scale_w,self.bit_w, False))
#         if self.Q_back:
#             quant = lambda x : fixed_point_quantize(x, wl=8, fl=6, rounding="nearest")
#             Q_factors = [torch.swapaxes(U,0,-1) for U in Q_factors[::-1]]
#             output = ttm_times_mat.apply(quant,input.T, *Q_factors)
#         elif self.recur:
#             Q = lambda x: scale.apply(x,self.scale_w,self.bit_w, False)
#             output = input @ self.tensor.full_from_factors(Q_factors,quant=Q).reshape([self.out_features,self.in_features]).T
#         else:
#             output = input @ self.tensor.full_from_factors(Q_factors).reshape([self.out_features,self.in_features]).T

#         if self.bias is not None:
#             Q_bias = (scale.apply(self.bias,self.scale_b,self.bit_b, False))
#             output = scale.apply(output,self.scale_b,self.bit_b, False) + Q_bias

#         ### Code for test purpose only ##############################################
#         # Q_factors_int = []
#         # for U in Q_factors:
#         #     Q_factors_int.append(U/self.scale_w)

#         # self.Q_tensor = self.tensor.full_from_factors(Q_factors_int).reshape([self.out_features,self.in_features])
#         # self.n_tensor = self.tensor.full_from_factors(Q_factors).reshape([self.out_features,self.in_features])
#         self.Q_factors = Q_factors
#         self.output = output
#         ### Code for test purpose only ##############################################
        
#         return output
        
#         # return F.linear(input,self.tensor.full_from_factors(Q_factors).reshape([self.out_features,self.in_features]),Q_bias)

#     def update_rank_parameters(self):
#         self.tensor.update_rank_parameters()



# # class Q_conv2d_old(nn.Conv2d):
# #     def __init__(self,
# #                  in_channels,
# #                  out_channels,
# #                  kernel_size,
# #                  stride= (1,1),
# #                  padding=(0,0),
# #                  dilation=(1,1),
# #                  groups=1,
# #                  bias = True,
# #                  padding_mode = 'zeros',
# #                  device=None,
# #                  dtype=None,
# #                  bit_w = 8,
# #                  bit_b = 8,
# #                  scale_w = 2**(-5),
# #                  scale_b = 2**(-5)
# #     ):
# #         super(Q_conv2d_old,self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode,device,dtype)

# #         self.stride = stride
# #         self.padding = padding 
# #         self.dilation = dilation
# #         self.groups = groups

# #         self.bit_w = bit_w
# #         self.bit_b = bit_b
# #         # self.max_q = 2.0**(bit-1)-1.0
# #         # self.min_q = -2.0**(bit-1)
# #         # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

# #         self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
# #         self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))
       

# #     def forward(self, input):
# #         Q_weight = scale.apply(self.weight,self.scale_w,self.bit_w)
# #         Q_bias = scale.apply(self.bias,self.scale_b,self.bit_b)
        
# #         output = F.conv2d(input,Q_weight,bias = None, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
# #         output = scale.apply(output,self.scale_b,self.bit_b)
# #         # print(output.shape)
# #         # print(Q_bias.shape)
# #         output = output.transpose(1,3)
# #         output = output + Q_bias
# #         output = output.transpose(1,3)

# #         return output


# class Q_conv2d(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride= (1,1),
#                  padding=(0,0),
#                  dilation=(1,1),
#                  groups=1,
#                  bias = True,
#                  padding_mode = 'zeros',
#                  device=None,
#                  dtype=None,
#                  bit_w = 8,
#                  bit_b = 8,
#                  scale_w = 2**(-5),
#                  scale_b = 2**(-5)
#     ):
#         super(Q_conv2d,self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         # self.transposed = transposed
#         # self.output_padding = output_padding
#         self.groups = groups

#         self.bit_w = bit_w
#         self.bit_b = bit_b
#         # self.max_q = 2.0**(bit-1)-1.0
#         # self.min_q = -2.0**(bit-1)
#         # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

#         self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
#         self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))

#         self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#         self.init()
       
#     def init(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        

#     def forward(self, input):
        

#         # output = F.conv2d(input,self.weight,bias = self.bias, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

#         Q_weight = scale.apply(self.weight,self.scale_w,self.bit_w, False)
        
#         output = F.conv2d(input,Q_weight,bias = None, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        

#         self.output = output
#         if self.bias is not None:
#             Q_bias = scale.apply(self.bias,self.scale_b,self.bit_b, False)
#             output = scale.apply(output,self.scale_b,self.bit_b,False)
#             output = output.transpose(1,3)
#             output = output + Q_bias
#             output = output.transpose(1,3)

#         self.Q_weight = Q_weight

#         return output

# class Q_Tensorizedconv2d(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size = (3,3),
#                  stride= (1,1),
#                  padding=(0,0),
#                  dilation=(1,1),
#                  groups=1,
#                  bias = True,
#                  padding_mode = 'zeros',
#                  device=None,
#                  dtype=None,
#                  init=None,
#                  shape=None,
#                  tensor_type='TensorTrain',
#                  max_rank=20,
#                  em_stepsize=1.0,
#                  prior_type='log_uniform',
#                  eta = None,
#                  bit_w = 8,
#                  bit_b = 8,
#                  scale_w = 2**(-5),
#                  scale_b = 2**(-5)
#     ):
#         super(Q_Tensorizedconv2d,self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         # self.transposed = transposed
#         # self.output_padding = output_padding
#         self.groups = groups

#         self.bit_w = bit_w
#         self.bit_b = bit_b
#         # self.max_q = 2.0**(bit-1)-1.0
#         # self.min_q = -2.0**(bit-1)
#         # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

#         self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
#         self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))

#         # self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#         self.init()

#         if shape == None:
#             shape = self.get_tensor_shape(out_channels)
#             shape = shape + self.get_tensor_shape(in_channels)
#             shape = shape + list(kernel_size)


#         target_stddev = 2/np.sqrt(self.in_channels*kernel_size[0]*kernel_size[1])
#         self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

       
#     def init(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         # self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
    
#     def get_tensor_shape(self,n):
#         if n==64:
#             return [8,8]
#         if n==128:
#             return [8,16]
#         if n==256:
#             return [16,16]
#         if n==512:
#             return [16,32]

#     def forward(self, input, rank_update = True):
        
#         if self.training and rank_update:
#             self.tensor.update_rank_parameters()
        
#         Q_factors = []        
#         for U in self.tensor.factors:
#             Q_factors.append(scale.apply(U,self.scale_w,self.bit_w, False))
#         self.Q_factors = Q_factors
        
#         # output = F.conv2d(input,self.weight,bias = self.bias, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

#         w = self.tensor.get_full_factors(Q_factors).reshape(self.out_channels,self.in_channels,*self.kernel_size)
#         output = F.conv2d(input,w,bias = None, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

#         if self.bias is not None:
#             Q_bias = (scale.apply(self.bias,self.scale_b,self.bit_b, False))
#             output = scale.apply(output,self.scale_b,self.bit_b,False)

#             self.output = output

#             output = output.transpose(1,3)
#             output = output + Q_bias
#             output = output.transpose(1,3)

#         self.Q_weight = w
#         self.output = output

#         return output


# class Tensorizedconv2d(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size = (3,3),
#                  stride= (1,1),
#                  padding=(0,0),
#                  dilation=(1,1),
#                  groups=1,
#                  bias = True,
#                  padding_mode = 'zeros',
#                  device=None,
#                  dtype=None,
#                  init=None,
#                  shape=None,
#                  tensor_type='TensorTrain',
#                  max_rank=20,
#                  em_stepsize=1.0,
#                  prior_type='log_uniform',
#                  eta = None,
#     ):
#         super(Tensorizedconv2d,self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         # self.transposed = transposed
#         # self.output_padding = output_padding
#         self.groups = groups

    

#         # self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#         self.init()

#         if shape == None:
#             shape = self.get_tensor_shape(out_channels)
#             shape = shape + self.get_tensor_shape(in_channels)
#             shape = shape + list(kernel_size)


#         target_stddev = 2/np.sqrt(self.in_channels*kernel_size[0]*kernel_size[1])
#         self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

       
#     def init(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         # self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
    
#     def get_tensor_shape(self,n):
#         if n==64:
#             return [8,8]
#         if n==128:
#             return [8,16]
#         if n==256:
#             return [16,16]
#         if n==512:
#             return [16,32]

#     def forward(self, input, rank_update = True):
        
#         if self.training and rank_update:
#             self.tensor.update_rank_parameters()
        
       
#         # output = F.conv2d(input,self.weight,bias = self.bias, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

#         w = self.tensor.get_full().reshape(self.out_channels,self.in_channels,*self.kernel_size)
#         output = F.conv2d(input,w,bias = self.bias, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
#         self.output = output
#         return output


################# Layers that have tensor format input and output ##############################################################

class T_linear(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                bias=False,
                init=None,
                shape=None,
                tensor_type='TensorTrain',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
                out_tensor = True,
                input_order = None,
    ):

        # super(T_linear,self).__init__(in_features,out_features,bias,device,dtype)
        super(T_linear,self).__init__()

        self.out_tensor = out_tensor
        if input_order == None:
            self.input_order = int(len(shape)/2)
        self.input_order = input_order


        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(2/self.in_features)

        #shape taken care of at input time
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

    def forward(self, input, rank_update=True):
        #input is in TT tensor format 
        #output is in TT tensor format
        if self.training and rank_update:
            self.tensor.update_rank_parameters()

        m = self.input_order
        
        if isinstance(input, list):
            out = torch.tensordot(input[1],self.tensor.factors[0],[[1],[1]])
            out = torch.moveaxis(out, 2, 1)
            for i in range(1,m):
                tmp = torch.tensordot(input[i+1],self.tensor.factors[i],[[1],[1]])
                out = torch.tensordot(out, tmp, [[2,3],[0,2]])
            out = torch.squeeze(out)
            out = torch.tensordot(input[0],out,[[-1],[0]])
            # out = torch.tensordot(out,self.tensor.factors[m],[[2],[0]])

            out = [out] + self.tensor.factors[m:]

        else:
            # print(input.shape)
            if len(input.shape)==2:
                mat_shape = [input.shape[0]] + [U.shape[1] for U in self.tensor.factors[0:m]]
                N=2
            elif len(input.shape)==3:
                N=3
                mat_shape = [input.shape[0]]+[input.shape[1]] + [U.shape[1] for U in self.tensor.factors[0:m]]
            input = torch.reshape(input, [1] + mat_shape)

            out = torch.squeeze(self.tensor.factors[0])

            for U in self.tensor.factors[1:m]:
                out = torch.tensordot(out, U, [[-1],[0]])
  
            out = torch.tensordot(input, out, [list(range(N,N+m)), list(range(0,m))])
            out = [out] + self.tensor.factors[m:]

            # out = torch.tensordot(input,torch.squeeze(self.tensor.factors[0]),[[N],[0]])
            # for U in self.tensor.factors[1:m]:
            #     out = torch.tensordot(out, U, [[N,-1],[1,0]])
            # out = [out] + self.tensor.factors[m:]
            


        if self.out_tensor:
            return out
        else:
            N = len(out[0].shape)
            output = out[1]

            for U in out[2:]:
                output = torch.tensordot(output,U,[[-1],[0]])
            
            output = torch.tensordot(out[0],output,[[-1],[0]])

            output = torch.flatten(output, start_dim = N-1, end_dim = -1)
            output = torch.squeeze(output)

            return output
            # out = tl.tt_to_tensor(out)
            # return out.reshape(out.shape[0],-1)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()


class T_relu(nn.Module):
    def __init__(self):
        super(T_relu,self).__init__()
    def forward(self, input):
        if isinstance(input, list):
            # return [F.relu(input[0])] + input[1:]
            # return [input[0]] + [F.relu(U) for U in input[1:]]
            return [F.relu(U) for U in input]
        else:
            return F.relu(input)

class T_dropout(nn.Module):
    def __init__(self,p):
        super(T_dropout,self).__init__()
        self.p = p
    def forward(self, input):
        if isinstance(input, list):
            return [F.dropout(input[0],self.p)] + input[1:]
            # return [F.dropout(U,self.p) for U in input]
        else:
            return F.dropout(input,self.p)

class T_layernorm(nn.Module):
    def __init__(self,shape,eps = 1e-5):
        super(T_layernorm,self).__init__()
        self.shape = shape
        self.eps = eps
        self.layer_norm = torch.nn.LayerNorm(self.shape, eps = self.eps)
    def forward(self, input):
        if isinstance(input, list):
            return [self.layer_norm(input[0])] + input[1:]  #only normalize the first core
            # return [self.layer_norm(input[0])] + [F.layer_norm(U, U.shape, eps = self.eps) for U in input[1:]]    #normalize all cores
        else:
            return self.layer_norm(input)



class TTM_linear(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                bias=False,
                init=None,
                shape=None,
                tensor_type='TensorTrainMatrix',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
                out_tensor = True,
                input_order = None,
    ):

        # super(T_linear,self).__init__(in_features,out_features,bias,device,dtype)
        super(TTM_linear,self).__init__()

        self.out_tensor = out_tensor
        if input_order == None:
            self.input_order = int(len(shape[0]))
        self.input_order = input_order


        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(2/self.in_features)

        # m = self.input_order
        # if isinstance(max_rank, list):
        #     pass
        # else:
        #     max_rank = [max_rank]*(m+1)

        self.factors = []
        for i in range(self.input_order):
            tmp = torch.randn(max_rank,shape[0][i],shape[1][i],max_rank,device=device)*torch.sqrt(torch.tensor(2/(shape[0][i]*max_rank)))
            self.factors.append(nn.Parameter(tmp))
        tmp = torch.randn(max_rank,max_rank,device=device)*torch.sqrt(torch.tensor(2/(max_rank)))
        self.factors = [nn.Parameter(tmp)] + self.factors 
        
        # i = self.input_order-1
        # tmp = torch.randn(shape[0][i],shape[1][i],device=device)*torch.sqrt(torch.tensor(2/(shape[1][i])))
        # self.factors.append(nn.Parameter(tmp))
        #shape taken care of at input time
        # self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

    def forward(self, input, rank_update=True):

        m = self.input_order

        out = []
        
        if isinstance(input, list):
            for i in range(m-1):
                out = out + [torch.tensordot(input[i+1],self.factors[i+1],[[0,1],[0,1]])]
         
            out = out + [torch.tensordot(input[-1],self.factors[-1],[[0,1],[0,1]]).swapaxes(0,2)]
            out = [torch.tensordot(input[0],self.factors[0],[[-1],[0]])] + out


      
        if self.out_tensor:
            return out
        else:
            N = len(out[0].shape)
            output = out[1]

            for U in out[2:]:
                output = torch.tensordot(output,U,[[-1],[0]])
            
            output = torch.tensordot(out[0],output,[[-1],[0]])

            output = torch.flatten(output, start_dim = N-1, end_dim = -1)
            output = torch.squeeze(output)

            return output


class Factor_linear(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                bias=False,
                rank=20,
                device=None
    ):

        # super(T_linear,self).__init__(in_features,out_features,bias,device,dtype)
        super(Factor_linear,self).__init__()
        A = nn.Parameter(torch.randn(in_features,rank,device=device)*torch.sqrt(torch.tensor(2/in_features)))
        B = nn.Parameter(torch.randn(rank,out_features,device=device)*torch.sqrt(torch.tensor(2/out_features)))
        self.A = A
        self.B = B
        
    def forward(self, input):
            self.x = input
            return input@self.A@self.B


class TT_embedding(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                shape=None,
                max_rank=20,
                device=None,
                input_order = None,
                out_tensor = None
    ):

        # super(T_linear,self).__init__(in_features,out_features,bias,device,dtype)
        super(TTM_linear,self).__init__()

        self.out_tensor = out_tensor
        if input_order == None:
            self.input_order = int(len(shape[0]))
        self.input_order = input_order


        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(2/self.in_features)

        # m = self.input_order
        # if isinstance(max_rank, list):
        #     pass
        # else:
        #     max_rank = [max_rank]*(m+1)

        self.factors = []
        for i in range(self.input_order):
            tmp = torch.randn(max_rank,shape[0][i],shape[1][i],max_rank,device=device)*torch.sqrt(torch.tensor(2/(shape[0][i]*max_rank)))
            self.factors.append(nn.Parameter(tmp))
        tmp = torch.randn(max_rank,max_rank,device=device)*torch.sqrt(torch.tensor(2/(max_rank)))
        self.factors = [nn.Parameter(tmp)] + self.factors 
        
        # i = self.input_order-1
        # tmp = torch.randn(shape[0][i],shape[1][i],device=device)*torch.sqrt(torch.tensor(2/(shape[1][i])))
        # self.factors.append(nn.Parameter(tmp))
        #shape taken care of at input time
        # self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

    def forward(self, input, rank_update=True):

        m = self.input_order

        out = []
        
        if isinstance(input, list):
            for i in range(m-1):
                out = out + [torch.tensordot(input[i+1],self.factors[i+1],[[0,1],[0,1]])]
         
            out = out + [torch.tensordot(input[-1],self.factors[-1],[[0,1],[0,1]]).swapaxes(0,2)]
            out = [torch.tensordot(input[0],self.factors[0],[[-1],[0]])] + out


      
        if self.out_tensor:
            return out
        else:
            N = len(out[0].shape)
            output = out[1]

            for U in out[2:]:
                output = torch.tensordot(output,U,[[-1],[0]])
            
            output = torch.tensordot(out[0],output,[[-1],[0]])

            output = torch.flatten(output, start_dim = N-1, end_dim = -1)
            output = torch.squeeze(output)

            return output
