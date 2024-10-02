import torch
import numpy as np
import os

from core.TFONet import to_numpy
from loss_func import gaussian_augment


def gaussian_smoothed_model(X,f,sigma,N_sample):
    batch_size = X.shape[0]
    sample_x, e_x = gaussian_augment(X, sigma, N_sample)
    sample_x_plus = sample_x
    sample_u_plus = f(sample_x_plus).reshape(N_sample,batch_size,1)
    u_pred = torch.mean(sample_u_plus,0)
    return to_numpy(u_pred)

def sparse_grids_model(X,f,sigma,level,rule):

    if rule == 'GQN':
        data_path = os.path.join(os.getcwd(), 'GQN_d2', 'GQN_d2_l' + str(level) + '.asc')
        data = np.loadtxt(data_path,delimiter=',')
        delta = data[:,0:-1]
        weight = data[:,-1:]
 
    elif rule == 'KPN':
        data_path = os.path.join(os.getcwd(), 'KPN_d2', 'KPN_d2_l' + str(level) + '.asc')
        data = np.loadtxt(data_path,delimiter=',')
        delta = data[:,0:-1]
        weight = data[:,-1:]

    delta = torch.tensor(delta,device=X.device,dtype=X.dtype) * sigma
    weight = torch.tensor(weight,device=X.device,dtype=X.dtype)
    
    n_node = delta.shape[0]
    batch_size = X.shape[0]

    delta_expand = delta.view(n_node,1,2).repeat(1,batch_size,1)
    weight_expand = weight.view(n_node,1,1).repeat(1,batch_size,1)
    X_expand = X.view(1,batch_size,2).repeat(n_node,1,1)
    X_plus = (X_expand + delta_expand).reshape(n_node*batch_size,2)
    u_plus = f(X_plus).reshape(n_node,batch_size,1)
    u_pred = torch.sum(weight_expand*u_plus,dim=0)
    return to_numpy(u_pred)


def val_func_init_from_config(config):

    if config.getboolean('loss_func', 'val_every') == True:
        if config['loss_func'].get('gradient_func') == 'gradients_cd2':
            val_func = val_func_init_from_config_ADFD(config)
        elif config['loss_func'].get('gradient_func') == 'gradients_stein':
            val_func = val_func_init_from_config_stein(config)
        elif config['loss_func'].get('gradient_func') == 'gradients_sparse_grid':
            val_func = val_func_init_from_config_sparse_grids(config)
        elif config['loss_func'].get('gradient_func') == 'gradients_auto_diff':
            val_func = val_func_init_from_config_ADFD(config)
        else:
            raise NotImplementedError

        return val_func
    else:
        # use this to save time during training
        return None


def val_func_init_from_config_ADFD(config):
    # use this for eval when using auto diff or finite difference

    def val_func(model, dataset, *args):
        x, u = dataset.eval()
        u_pred = to_numpy(model(x))
        # mse_error = ((u_pred - u) ** 2).mean()
        # return mse_error
        
        relative_l2_error = np.linalg.norm(u_pred - u) / np.linalg.norm(u)
        return relative_l2_error

    return val_func



def val_func_init_from_config_stein(config):
    # use this for eval when using MC stein

    def val_func(model, dataset, *args):
        x, u = dataset.eval()
        f_model = lambda X: model(X)
        u_pred = gaussian_smoothed_model(x,f_model,sigma=1e-3,N_sample=256)
        # mse_error = ((u_pred - u) ** 2).mean()
        # return mse_error
        
        relative_l2_error = np.linalg.norm(u_pred - u) / np.linalg.norm(u)
        return relative_l2_error

    return val_func



def val_func_init_from_config_sparse_grids(config):
    # use this for eval when using sparse grids


    def val_func(model, dataset, *args):
        x, u = dataset.eval()
        f_model = lambda X: model(X)
        u_pred = sparse_grids_model(x,f_model,sigma=1e-3,level=3,rule='GQN')
      
        # mse_error = ((u_pred - u) ** 2).mean()
        # return mse_error
        
        relative_l2_error = np.linalg.norm(u_pred - u) / np.linalg.norm(u)
        return relative_l2_error

    return val_func