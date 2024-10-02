from collections import OrderedDict
import torch
import torch.nn as nn
import tltorch
import math
from bisect import insort_left
from core.tensor_fwd_bwd.tensorized_linear import TensorizedLinear
from core.tensor_layers.layers import TensorizedLinear_module
from core.models.sparse_bp_ttm_mlp import TTM_Linear_module, TTM_LinearBlock
from .fourier_layers import FourierMappingLayer
from .nl_init import nl_init_dict
from core.models.sparse_bp_base import SparseBP_Base
from core.models.sparse_bp_mlp import LinearBlock
from core.models.layers.custom_linear import MZIBlockLinear
from core.fsr_mrr.fsr_mrr_linear import FSR_MRRLinear

# ============= self-defined factorization search for tt =============
def factorize(value, min_value=2, remaining=-1):
    """Factorize an integer input value into it's smallest divisors
    
    Parameters
    ----------
    value : int
        integer to factorize
    min_value : int, default is 2
        smallest divisors to use
    remaining : int, default is -1
        DO NOT SPECIFY THIS VALUE, IT IS USED FOR TAIL RECURSION

    Returns
    -------
    factorization : int tuple
        ints such that prod(factorization) == value
    """
    if value <= min_value or remaining == 0:
        return (value, )
    lim = math.isqrt(value)
    for i in range(min_value, lim+1):
        if value == i:
            return (i, )
        if not (value % i):
            return (i, *factorize(value//i, min_value=min_value, remaining=remaining-1))
    return (value, )

def merge_ints(values, size):
    """Utility function to merge the smallest values in a given tuple until it's length is the given size
    
    Parameters
    ----------
    values : int list
        list of values to merge
    size : int
        target len of the list
        stop merging when len(values) <= size
    
    Returns
    -------
    merge_values : list of size ``size``
    """
    if len(values) <= 1:
        return values

    values = sorted(list(values))
    while (len(values) > size):
        a, b, *values = values
        insort_left(values, a*b)
    
    return tuple(values)
    
def get_tensorized_shape(in_features, out_features, order=None, min_dim=2, factorization=None, verbose=True):
    """ Factorizes in_features and out_features such that:
    * they should both be factorized into `order` integers
    * If not satisfied, remain its main length (only for TT)
    * each of the factors should be at least min_dim
    
    This is used to tensorize a matrix of size (in_features, out_features) into a higher order tensor
    
    Parameters
    ----------
    in_features, out_features : int
    order : int
        the number of integers that each input should be factorized into
    min_dim : int
        smallest acceptable integer value for the factors
        
    Returns
    -------
    in_tensorized, out_tensorized : tuple[int]
        tuples of ints used to tensorize each dimension
        
    Notes
    -----
    This is a bruteforce solution but is enough for the dimensions we encounter in DNNs
    """
    in_ten = factorize(in_features, min_value=min_dim)
    out_ten = factorize(out_features, min_value=min_dim)
    if order is not None:
        if factorization == 'tt':
            merge_size = order
        else:
            merge_size = min(order, len(in_ten), len(out_ten))
    else:
        merge_size = min(len(in_ten), len(out_ten))

    if len(in_ten) > merge_size:
        in_ten = merge_ints(in_ten, size=merge_size)
    if len(out_ten) > merge_size:
        out_ten = merge_ints(out_ten, size=merge_size)

    if verbose:
        print(f'Tensorizing (in, out)=({in_features, out_features}) -> ({in_ten, out_ten})')
    return in_ten, out_ten

class FCBlock(nn.Module):

    def __init__(self, in_features=3, out_features=1, hidden_features=20, num_layers=3, nonlinearity='sine',
                 nl_last_layer=False, fourier_mapping=False, freq=30, freq_trainable=False,
                 bias=True, device=None, dtype=None):
        super(FCBlock, self).__init__()
        nl, init = nl_init_dict[nonlinearity]

        self.net = OrderedDict()

        if fourier_mapping:
            self.net['fourier_mapping'] = FourierMappingLayer(in_features=in_features, out_features=hidden_features,
                                                              freq=freq, freq_trainable=freq_trainable, device=device)
            first_layer_in_features = hidden_features

        else:
            first_layer_in_features = in_features

        for i in range(1, num_layers + 1):
            if i == 1:
                self.net['fc1'] = nn.Linear(in_features=first_layer_in_features, out_features=hidden_features,
                                            bias=bias, device=device, dtype=dtype)

                if num_layers > 1:
                    self.net['nl1'] = nl
            elif i != num_layers:
                self.net['fc%d' % i] = nn.Linear(in_features=hidden_features, out_features=hidden_features, bias=bias,
                                                 device=device, dtype=dtype)
                self.net['nl%d' % i] = nl
            else:
                self.net['fc%d' % i] = nn.Linear(in_features=hidden_features, out_features=out_features, bias=bias,
                                                 device=device, dtype=dtype)

            init(self.net['fc%d' % i])

        if nl_last_layer:
            self.net['nl%d' % num_layers] = nl

        self.net = nn.Sequential(self.net)

    def forward(self, x):
        return self.net(x)


# class TensorizedFCBlock(nn.Module):

#     def __init__(self, in_features=3, out_features=1, hidden_features=20, num_layers=3, nonlinearity='sine',
#                  order=4, min_dim=4, factorization='cp', rank='same', tensorize_first=False, tensorize_last=False,
#                  bias=True, yml_config=None, device=None, dtype=None):
#         super(TensorizedFCBlock, self).__init__()

#         nl, init = nl_init_dict[nonlinearity]

#         self.net = OrderedDict()

#         if not tensorize_first:
#             self.net['fc1'] = nn.Linear(in_features=in_features, out_features=hidden_features,
#                                         bias=bias, device=device, dtype=dtype)
#             init(self.net['fc1'])
#         else:
#             if yml_config is not None and yml_config.model.set_shape == True:
#                 tensorized_shape = (tuple(yml_config.model.in_shape), tuple(yml_config.model.hidden_shape))
#             else:
#                 tensorized_shape = tltorch.utils.get_tensorized_shape(in_features=in_features, out_features=hidden_features,
#                                                                   order=order, verbose=False, min_dim=min_dim)
            
#             if factorization == 'ttm':
#                 self.net['tensorized_fc1'] = TensorizedLinear_module(in_features=in_features, out_features=hidden_features, 
#                                                       bias=bias, shape=tensorized_shape, tensor_type='TensorTrainMatrix', max_rank=rank,
#                                                       device=device, dtype=dtype)
#             else:
#                 self.net['tensorized_fc1'] = TensorizedLinear(in_tensorized_features=tensorized_shape[0],
#                                                           out_tensorized_features=tensorized_shape[1],
#                                                           factorization=factorization, rank=rank,
#                                                           bias=bias, device=device, dtype=dtype)

#         self.net['nl1'] = nl

#         for i in range(2, num_layers):
#             if yml_config is not None and yml_config.model.set_shape == True:
#                 tensorized_shape = (tuple(yml_config.model.hidden_shape), tuple(yml_config.model.hidden_shape))
#             else:
#                 tensorized_shape = tltorch.utils.get_tensorized_shape(in_features=hidden_features,
#                                                                   out_features=hidden_features,
#                                                                   order=order, verbose=False, min_dim=min_dim)
            
#             if factorization == 'ttm':
#                 self.net['tensorized_fc%d' % i] = TensorizedLinear_module(in_features=hidden_features, out_features=hidden_features, 
#                                                       bias=bias, shape=tensorized_shape, tensor_type='TensorTrainMatrix', max_rank=rank,
#                                                       device=device, dtype=dtype)
#             else:
#                 self.net['tensorized_fc%d' % i] = TensorizedLinear(in_tensorized_features=tensorized_shape[0],
#                                                                out_tensorized_features=tensorized_shape[1],
#                                                                factorization=factorization, rank=rank,
#                                                                bias=bias, device=device, dtype=dtype)
#             self.net['nl%d' % i] = nl

#         if not tensorize_last:
#             self.net['fc%d' % num_layers] = nn.Linear(in_features=hidden_features, out_features=out_features,
#                                                       bias=bias, device=device, dtype=dtype)
#             init(self.net['fc%d' % num_layers])

#         else:
#             if yml_config is not None and yml_config.model.set_shape == True:
#                 tensorized_shape = (tuple(yml_config.model.hidden_shape), tuple(yml_config.model.out_shape))
#             else:
#                 tensorized_shape = tltorch.utils.get_tensorized_shape(in_features=hidden_features,
#                                                                   out_features=out_features,
#                                                                   order=order, verbose=False, min_dim=min_dim)

#             if factorization == 'ttm':
#                 self.net['tensorized_fc%d' % i] = TensorizedLinear_module(in_features=hidden_features, out_features=out_features, 
#                                                       bias=bias, shape=tensorized_shape, tensor_type='TensorTrainMatrix', max_rank=rank,
#                                                       device=device, dtype=dtype)
#             else:
#                 self.net['fc%d' % num_layers] = TensorizedLinear(in_tensorized_features=tensorized_shape[0],
#                                                           out_tensorized_features=tensorized_shape[1],
#                                                           factorization=factorization, rank=rank,
#                                                           bias=bias, device=device, dtype=dtype)

#         self.net = nn.Sequential(self.net)

#     def forward(self, x):
#         return self.net(x)

class TensorizedFCBlock(nn.Module):

    def __init__(self, 
                 in_features=3, 
                 out_features=1, 
                 hidden_features=20, 
                 num_layers=3, 
                 nonlinearity='sine',
                 shape_list=None, 
                 order=4, 
                 min_dim=4, 
                 factorization='cp', 
                 rank='same', 
                 tensorize_first=False, 
                 tensorize_last=False,
                 bias=True, 
                 device=None, 
                 dtype=None):
        # shape_list: [in_shape, hidden_shape, out_shape]
        super(TensorizedFCBlock, self).__init__()

        if shape_list is not None:
            # in_shape_list = tuple(shape_list[0])
            # hidden_shape_list = tuple(shape_list[1])
            # out_shape_list = tuple(shape_list[2])
            in_shape_list = shape_list[0]
            hidden_shape_list = shape_list[1]
            out_shape_list = shape_list[2]
        else:
            in_shape, hidden_shape = get_tensorized_shape(
                in_features=in_features, 
                out_features=hidden_features,
                order=order, 
                verbose=False, 
                min_dim=min_dim,
                factorization = factorization)
            _, out_shape = get_tensorized_shape(
                in_features=hidden_features, 
                out_features=out_features,
                order=order, 
                verbose=False, 
                min_dim=min_dim,
                factorization = factorization)
            in_shape_list=[in_shape, hidden_shape]
            hidden_shape_list=[hidden_shape, hidden_shape]
            out_shape_list=[out_shape, out_shape]

        nl, init = nl_init_dict[nonlinearity]

        self.net = OrderedDict()

        if not tensorize_first:
            self.net['fc1'] = nn.Linear(in_features=in_features, out_features=hidden_features,
                                        bias=bias, device=device, dtype=dtype)
            init(self.net['fc1'])
        else:
            # tensorized_shape = get_tensorized_shape(
            #     in_features=in_features, 
            #     out_features=hidden_features,
            #     order=order, 
            #     verbose=False, 
            #     min_dim=min_dim,
            #     factorization = factorization)

            if factorization == 'ttm':
                self.net['tensorized_fc1'] = TensorizedLinear_module(in_features=in_features, out_features=hidden_features, 
                                                      bias=bias, shape=[tuple(in_shape_list[0]),tuple(in_shape_list[1])], tensor_type='TensorTrainMatrix', max_rank=rank,
                                                      device=device, dtype=dtype)
            else:
                self.net['tensorized_fc1'] = TensorizedLinear(in_tensorized_features=tuple(in_shape_list[0]),
                                                          out_tensorized_features=tuple(in_shape_list[1]),
                                                          factorization=factorization, rank=rank,
                                                          bias=bias, device=device, dtype=dtype)
                # self.net['tensorized_fc1'] = TensorizedLinear(in_tensorized_features=in_shape,
                #                                           out_tensorized_features=hidden_shape,
                #                                           factorization=factorization, rank=rank,
                #                                           bias=bias, device=device, dtype=dtype)

        self.net['nl1'] = nl

        for i in range(2, num_layers):
            # tensorized_shape = get_tensorized_shape(
            #     in_features=hidden_features, 
            #     out_features=hidden_features,
            #     order=order, 
            #     verbose=False, 
            #     min_dim=min_dim,
            #     factorization = factorization)
            
            if factorization == 'ttm':
                self.net['tensorized_fc%d' % i] = TensorizedLinear_module(in_features=hidden_features, out_features=hidden_features, 
                                                      bias=bias, shape=[tuple(hidden_shape_list[0]),tuple(hidden_shape_list[1])], tensor_type='TensorTrainMatrix', max_rank=rank,
                                                      device=device, dtype=dtype)
            else:
                self.net['tensorized_fc%d' % i] = TensorizedLinear(in_tensorized_features=tuple(hidden_shape_list[0]),
                                                               out_tensorized_features=tuple(hidden_shape_list[1]),
                                                               factorization=factorization, rank=rank,
                                                               bias=bias, device=device, dtype=dtype)
                # self.net['tensorized_fc%d' % i] = TensorizedLinear(in_tensorized_features=hidden_shape,
                #                                                out_tensorized_features=hidden_shape,
                #                                                factorization=factorization, rank=rank,
                #                                                bias=bias, device=device, dtype=dtype)
            self.net['nl%d' % i] = nl

        if not tensorize_last:
            self.net['fc%d' % num_layers] = nn.Linear(in_features=hidden_features, out_features=out_features,
                                                      bias=bias, device=device, dtype=dtype)
            init(self.net['fc%d' % num_layers])

        else:
            # tensorized_shape = get_tensorized_shape(
            #     in_features=hidden_features, 
            #     out_features=out_features,
            #     order=order, 
            #     verbose=False, 
            #     min_dim=min_dim,
            #     factorization = factorization)

            if factorization == 'ttm':
                self.net['tensorized_fc%d' % num_layers] = TensorizedLinear_module(in_features=hidden_features, out_features=out_features, 
                                                      bias=bias, shape=[hidden_shape, out_shape], tensor_type='TensorTrainMatrix', max_rank=rank,
                                                      device=device, dtype=dtype)
            else:
                self.net['tensorized_fc%d' % num_layers] = TensorizedLinear(in_tensorized_features=hidden_shape,
                                                          out_tensorized_features=out_shape,
                                                          factorization=factorization, rank=rank,
                                                          bias=bias, device=device, dtype=dtype)
        self.net = nn.Sequential(self.net)

    def forward(self, x):
        x = torch.flatten(x,1)
        return self.net(x)

class ONN_FCBlock(SparseBP_Base):

    def __init__(
        self, 
        in_features=3, 
        out_features=1, 
        hidden_features=20, 
        num_layers=3, 
        nonlinearity='sine',
        bias=False, 
        device=None, 
        dtype=None,
        in_bit: int = 32,
        w_bit: int = 32,
        mode: str = "usv",
        v_max: float = 10.8,
        v_pi: float = 4.36,
        act_thres: float = 6.0,
        photodetect: bool = True
        ) -> None:
        super(ONN_FCBlock, self).__init__()

        nl, init = nl_init_dict[nonlinearity]

        # ====================== BUILD LAYERS ======================
        self.net = OrderedDict()
        
        # input layer
        self.net['fc1'] = MZIBlockLinear(
            in_channel = in_features,
            out_channel = hidden_features,
            # miniblock=miniblock, # init as 8
            bias=bias,
            mode=mode,
            v_max=v_max,
            v_pi=v_pi,
            in_bit=in_bit,
            w_bit=w_bit,
            photodetect=photodetect,
            device=device
        )

        self.net['nl1'] = nl

        # hidden layers
        for i in range(2, num_layers):
            self.net['fc%d' % i] = MZIBlockLinear(
                in_channel = hidden_features,
                out_channel = hidden_features,
                # miniblock=miniblock, # init as 8
                bias=bias,
                mode=mode,
                v_max=v_max,
                v_pi=v_pi,
                in_bit=in_bit,
                w_bit=w_bit,
                photodetect=photodetect,
                device=device
            )
            
            self.net['nl%d' % i] = nl

        # output layer
        self.net['fc%d' % num_layers] = MZIBlockLinear(
            in_channel = hidden_features,
            out_channel = out_features,
            # miniblock=miniblock, # init as 8
            bias=bias,
            mode=mode,
            v_max=v_max,
            v_pi=v_pi,
            in_bit=in_bit,
            w_bit=w_bit,
            photodetect=photodetect,
            device=device
        )
        
        # self.net['fc%d' % num_layers] = nn.Linear(in_features=hidden_features, out_features=out_features,
        #                                               bias=bias, device=device, dtype=dtype)
        # init(self.net['fc%d' % num_layers])

        self.net = nn.Sequential(self.net)

        # ====================== INIT ======================
        # Init every MZIBlockLinear & MZIBlockConv2d, so no init in creating these blocks
        # self.reset_parameters()
        self.gamma_noise_std = 0
        self.crosstalk_factor = 0

    def forward(self, x):
        return self.net(x)
    
"""
  ########## MLNCP Workshop ##########
"""
# class TensorizedONN_FCBlock(SparseBP_Base):

#     def __init__(
#         self, 
#         in_features=3, 
#         out_features=1, 
#         hidden_features=20, 
#         num_layers=3, 
#         nonlinearity='sine',
#         order=4, 
#         min_dim=4, 
#         factorization='ttm', 
#         rank: list=2, 
#         tensorize_first=False, 
#         tensorize_last=False,
#         bias=False, 
#         device=None, 
#         dtype=None,
#         in_bit: int = 32,
#         w_bit: int = 32,
#         mode: str = "usv",
#         v_max: float = 10.8,
#         v_pi: float = 4.36,
#         act_thres: float = 6.0,
#         photodetect: bool = True,
#         yml_config=None
#         ) -> None:
#         super(TensorizedONN_FCBlock, self).__init__()

#         nl, init = nl_init_dict[nonlinearity]

#         # ====================== BUILD LAYERS ======================
#         self.net = OrderedDict()

#         if hasattr(yml_config.model, 'set_shape'):
#             set_shape = yml_config.model.set_shape
#         else:
#             set_shape = False

#         if not tensorize_first:
#             self.net['fc1'] = LinearBlock(
#                 in_channel = in_features,
#                 out_channel = hidden_features,
#                 # miniblock=miniblock, # init as 8
#                 bias=bias,
#                 mode=mode,
#                 v_max=v_max,
#                 v_pi=v_pi,
#                 in_bit=in_bit,
#                 w_bit=w_bit,
#                 photodetect=photodetect,
#                 activation=False,
#                 device=device
#             )
#         else:
#             if yml_config is not None and set_shape == True:
#                 # tensorized_shape = (tuple(yml_config.model.in_shape), tuple(yml_config.model.hidden_shape))
#                 tensorized_shape = (tuple(yml_config.model.in_shape[0]), tuple(yml_config.model.in_shape[1]))
#             else:
#                 tensorized_shape = tltorch.utils.get_tensorized_shape(in_features=in_features, out_features=hidden_features,
#                                                                   order=order, verbose=False, min_dim=min_dim)

#             if factorization == 'ttm':
#                 self.net['tensorized_fc1'] = TTM_Linear_module(
#                     in_features=in_features, 
#                     out_features=hidden_features, 
#                     bias=bias, 
#                     shape=tensorized_shape, 
#                     tensor_type='TensorTrainMatrix', 
#                     max_rank=rank,
#                     device=device, 
#                     dtype=dtype,
#                     # miniblock=miniblock,
#                     mode=mode,
#                     v_max=v_max,
#                     v_pi=v_pi,
#                     in_bit=in_bit,
#                     w_bit=w_bit,
#                     photodetect=photodetect,
#                     activation=False,
#                     act_thres=act_thres
#                 )
#             else:
#                 raise NotImplementedError

#         self.net['nl1'] = nl

#         for i in range(2, num_layers):
#             if yml_config is not None and set_shape == True:
#                 # tensorized_shape = (tuple(yml_config.model.hidden_shape), tuple(yml_config.model.hidden_shape))
#                 tensorized_shape = (tuple(yml_config.model.hidden_shape[0]), tuple(yml_config.model.hidden_shape[1]))
#             else:
#                 tensorized_shape = tltorch.utils.get_tensorized_shape(in_features=hidden_features,
#                                                                   out_features=hidden_features,
#                                                                   order=order, verbose=False, min_dim=min_dim)
            
#             if factorization == 'ttm':
#                 self.net['tensorized_fc%d' % i] = TTM_Linear_module(
#                     in_features=hidden_features, 
#                     out_features=hidden_features, 
#                     bias=bias, 
#                     shape=tensorized_shape, 
#                     tensor_type='TensorTrainMatrix', 
#                     max_rank=rank,
#                     device=device, 
#                     dtype=dtype,
#                     # miniblock=miniblock,
#                     mode=mode,
#                     v_max=v_max,
#                     v_pi=v_pi,
#                     in_bit=in_bit,
#                     w_bit=w_bit,
#                     photodetect=photodetect,
#                     activation=False,
#                     act_thres=act_thres
#                 )
#             else:
#                 raise NotImplementedError
            
#             self.net['nl%d' % i] = nl

#         if not tensorize_last:
#             self.net['fc%d' % num_layers] = LinearBlock(
#                 in_channel = hidden_features,
#                 out_channel = out_features,
#                 # miniblock=miniblock, # set as init 8
#                 bias=bias,
#                 mode=mode,
#                 v_max=v_max,
#                 v_pi=v_pi,
#                 in_bit=in_bit,
#                 w_bit=w_bit,
#                 photodetect=photodetect,
#                 activation=False,
#                 device=device
#             )

#         else:
#             if yml_config is not None and set_shape == True:
#                 tensorized_shape = (tuple(yml_config.model.hidden_shape), tuple(yml_config.model.out_shape))
#             else:
#                 tensorized_shape = tltorch.utils.get_tensorized_shape(in_features=hidden_features,
#                                                                   out_features=out_features,
#                                                                   order=order, verbose=False, min_dim=min_dim)

#             if factorization == 'ttm':
#                 self.net['tensorized_fc%d' % i] = TTM_Linear_module(
#                     in_features=hidden_features, 
#                     out_features=out_features, 
#                     bias=bias, 
#                     shape=tensorized_shape, 
#                     tensor_type='TensorTrainMatrix', 
#                     max_rank=rank,
#                     device=device, 
#                     dtype=dtype,
#                     # miniblock=miniblock,
#                     mode=mode,
#                     v_max=v_max,
#                     v_pi=v_pi,
#                     in_bit=in_bit,
#                     w_bit=w_bit,
#                     photodetect=photodetect,
#                     activation=False,
#                     act_thres=act_thres
#                 )
#             else:
#                 raise NotImplementedError

#         self.net = nn.Sequential(self.net)

#         # ====================== INIT ======================
#         # Init every MZIBlockLinear & MZIBlockConv2d, so no init in creating these blocks
#         # self.reset_parameters()
#         self.gamma_noise_std = 0
#         self.crosstalk_factor = 0

#     def forward(self, x):
#         return self.net(x)

class TensorizedONN_FCBlock(SparseBP_Base):

    def __init__(self, 
                in_features=3, 
                out_features=1, 
                hidden_features=20, 
                num_layers=3, 
                nonlinearity='sine',
                shape_list=None, 
                order=4, 
                min_dim=4, 
                factorization='ttm', 
                rank=4, 
                tensorize_first=False, 
                tensorize_last=False,
                bias=True, 
                device=None, 
                dtype=None,
                
                in_bit: int = 32,
                w_bit: int = 32,
                mode: str = "usv",
                v_max: float = 10.8,
                v_pi: float = 4.36,
                act_thres: float = 6.0,
                photodetect: bool = True
                ):
        # shape_list: [in_shape, hidden_shape, out_shape]
        super(TensorizedONN_FCBlock, self).__init__()

        if shape_list is not None:
            # in_shape_list = tuple(shape_list[0])
            # hidden_shape_list = tuple(shape_list[1])
            # out_shape_list = tuple(shape_list[2])
            in_shape_list = shape_list[0]
            hidden_shape_list = shape_list[1]
            out_shape_list = shape_list[2]
        else:
            in_shape, hidden_shape = get_tensorized_shape(
                in_features=in_features, 
                out_features=hidden_features,
                order=order, 
                verbose=False, 
                min_dim=min_dim,
                factorization = factorization)
            _, out_shape = get_tensorized_shape(
                in_features=hidden_features, 
                out_features=out_features,
                order=order, 
                verbose=False, 
                min_dim=min_dim,
                factorization = factorization)
            in_shape_list=[in_shape, hidden_shape]
            hidden_shape_list=[hidden_shape, hidden_shape]
            out_shape_list=[out_shape, out_shape]

        nl, init = nl_init_dict[nonlinearity]

        self.net = OrderedDict()

        if not tensorize_first:
            self.net['fc1'] = LinearBlock(
                in_channel = in_features,
                out_channel = hidden_features,
                # miniblock=miniblock, # init as 8
                bias=bias,
                mode=mode,
                v_max=v_max,
                v_pi=v_pi,
                in_bit=in_bit,
                w_bit=w_bit,
                photodetect=photodetect,
                activation=False,
                device=device
            )
        else:
            if factorization == 'ttm':
                self.net['tensorized_fc1'] = TTM_LinearBlock(
                    in_channel=in_features,
                    out_channel=hidden_features,
                    miniblock=8,
                    bias=bias,
                    mode=mode,
                    v_max=v_max,
                    v_pi=v_pi,
                    in_bit=in_bit,
                    w_bit=w_bit,
                    photodetect=photodetect,
                    
                    device=device,
                    activation=False,
                    act_thres=act_thres,
  
                    in_shape=tuple(in_shape_list[0]),
                    out_shape=tuple(in_shape_list[1]),
                    tt_rank=rank,
                ) 
            else:
                raise NotImplementedError

        self.net['nl1'] = nl

        for i in range(2, num_layers):
            if factorization == 'ttm':
                self.net['tensorized_fc%d' % i] = TTM_LinearBlock(
                    in_channel=hidden_features,
                    out_channel=hidden_features,
                    miniblock=8,
                    bias=bias,
                    mode=mode,
                    v_max=v_max,
                    v_pi=v_pi,
                    in_bit=in_bit,
                    w_bit=w_bit,
                    photodetect=photodetect,
                    
                    device=device,
                    activation=False,
                    act_thres=act_thres,
  
                    in_shape=tuple(hidden_shape_list[0]),
                    out_shape=tuple(hidden_shape_list[1]),
                    tt_rank=rank,
                ) 
            else:
                raise NotImplementedError
            self.net['nl%d' % i] = nl

        if not tensorize_last:
            # self.net['fc%d' % num_layers] = LinearBlock(
            #     in_channel = hidden_features,
            #     out_channel = out_features,
            #     # miniblock=miniblock, # set as init 8
            #     bias=bias,
            #     mode=mode,
            #     v_max=v_max,
            #     v_pi=v_pi,
            #     in_bit=in_bit,
            #     w_bit=w_bit,
            #     photodetect=photodetect,
            #     activation=False,
            #     device=device
            # )
            
            self.net['fc%d' % num_layers] = nn.Linear(in_features=hidden_features, out_features=out_features,
                                                      bias=bias, device=device, dtype=dtype)
            init(self.net['fc%d' % num_layers])

        else:
            if factorization == 'ttm':
                self.net['tensorized_fc%d' % num_layers] = TTM_LinearBlock(
                    in_channel=hidden_features,
                    out_channel=out_features,
                    miniblock=8,
                    bias=bias,
                    mode=mode,
                    v_max=v_max,
                    v_pi=v_pi,
                    in_bit=in_bit,
                    w_bit=w_bit,
                    photodetect=photodetect,
                    
                    device=device,
                    activation=False,
                    act_thres=act_thres,
  
                    in_shape=tuple(out_shape_list[0]),
                    out_shape=tuple(out_shape_list[1]),
                    tt_rank=rank,
                ) 
            else:
                raise NotImplementedError
        
        self.net = nn.Sequential(self.net)

    def forward(self, x):
        x = torch.flatten(x,1)
        return self.net(x)