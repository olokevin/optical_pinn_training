import torch
from core.TFONet import gradients
import numpy as np
import os

def gaussian_augment(x:torch.Tensor, std, N_sample):

    # sample_x.shape: (N_sample*batch_size, 2)
    sample_x = torch.cat(N_sample*[x])
    e = torch.normal(mean=0, std=std, size=sample_x.shape, device=sample_x.device)
    return sample_x+e, e


def direction_vector(X:torch.Tensor,h,ith_dim):
    '''
    h: hyperparameter in FD method
    '''
    e = torch.zeros(X.shape,device=X.device)
    e[:,ith_dim:ith_dim+1] = 1
    return h*e

def gradients_auto_diff(X,f):

    u = f(X)
    u_g = gradients(u, X)[0]
    u_x,u_t = u_g[:, 0:1],u_g[:, 1:2]
    u_xx = gradients(u_x,X)[0][:, 0:1]

    return u_x,u_t,u_xx,u

def gradients_cd2(X,f,hx, ht):
    
    u = f(X)
    e_x = direction_vector(X,hx,0)
    u_plus = f(X+e_x)
    u_minus = f(X-e_x)
    u_x = (u_plus - u_minus) / (2*hx)
    u_xx = (u_plus+u_minus-2*u) / (hx**2)
    e_t = direction_vector(X,ht,1)
    u_t = (f(X+e_t) - f(X-e_t)) / (2*ht)
  
    return u_x,u_t,u_xx,u


def gradients_stein(X,f,sigma,N_sample):

    batch_size = X.shape[0]
    x,t = X[:, 0:1], X[:, 1:2]
    sample_x, e_x = gaussian_augment(x, sigma, N_sample)
    sample_x_plus = sample_x
    sample_x_minus = sample_x - 2*e_x

    sample_t, e_t = gaussian_augment(t, sigma, N_sample)
    sample_t_plus = sample_t
    sample_t_minus = sample_t - 2*e_t

    sample_X_plus = torch.cat([sample_x_plus, sample_t_plus], dim=1)
    sample_X_minus = torch.cat([sample_x_minus, sample_t_minus], dim=1)
    sample_u_plus = f(sample_X_plus).reshape(N_sample,batch_size,1)
    sample_u_minus = f(sample_X_minus).reshape(N_sample,batch_size,1)

    e_x = e_x.reshape(N_sample,batch_size,1)
    e_t = e_t.reshape(N_sample,batch_size,1)

    u_t = ((sample_u_plus-sample_u_minus)*e_t)/(2*sigma*sigma)
    u_t = torch.mean(u_t, dim=0)

    u_x = ((sample_u_plus-sample_u_minus)*e_x)/(2*sigma*sigma)
    u_x = torch.mean(u_x, dim=0)
    u_xx = (e_x**2-sigma**2)*(sample_u_plus+sample_u_minus-2*f(X).reshape(1,batch_size,1))/(2*(sigma**4))
    u_xx = torch.mean(u_xx, dim=0)
    u = torch.mean(sample_u_plus,0)
    return u_x,u_t,u_xx,u


def gradients_sparse_gird(X,f,sigma,level,rule):
    
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
    dim = X.shape[1]

    delta_expand = delta.view(n_node,1,dim).repeat(1,batch_size,1)
    weight_expand = weight.view(n_node,1,1).repeat(1,batch_size,1)

    X_expand = X.view(1,batch_size,dim).repeat(n_node,1,1)
    X_plus = (X_expand + delta_expand).reshape(n_node*batch_size,dim)
    X_minus = (X_expand - delta_expand).reshape(n_node*batch_size,dim)
    u_plus = f(X_plus).reshape(n_node,batch_size,1)
    u_minus = f(X_minus).reshape(n_node,batch_size,1)
    
    u_X = delta_expand*(u_plus-u_minus)/(2*(sigma**2))
    u_X = torch.sum(weight_expand*u_X,dim=0)
    u_x,u_t = u_X[:,0:dim-1],u_X[:,dim-1:dim]

    u_XX = (delta_expand**2 - sigma*sigma)*(u_plus + u_minus - 2*f(X).reshape(1,batch_size,1))/(2*(sigma**4))
    u_XX = torch.sum(weight_expand*u_XX,dim=0)
    u_xx = u_XX[:,0:dim-1]
    u = torch.sum(weight_expand*u_plus,dim=0)
    
    return u_x,u_t,u_xx,u


def loss_func_init_from_config(config):


    def loss_func(model, dataset, inputs=None, return_loss_reduction='mean'):
        if inputs is not None:
            x = inputs
        else:
            x = dataset.train()
        S_max = dataset.S_max
        T = dataset.T
        K = dataset.K
        r = dataset.r
        sigma = dataset.sigma
        num_interior = dataset.num_interior
        num_boundary = dataset.num_boundary
        f_model = lambda X: model(X)
        x_interior = x[0:num_interior,...]
        x_terminal = x[num_interior:num_interior+num_boundary,...]
        x_boundary_0 = x[num_interior+num_boundary:num_interior+2*num_boundary,...]
        x_boundary_max = x[num_interior+2*num_boundary:num_interior+3*num_boundary,...]
        
        if config['loss_func'].get('gradient_func') == 'gradients_cd2':
            u_x,u_t,u_xx,u_interior = gradients_cd2(x_interior,f_model,hx=1e-2,ht=1e-2)
        elif config['loss_func'].get('gradient_func') == 'gradients_stein':
            u_x,u_t,u_xx,u_interior = gradients_stein(x_interior,f_model,sigma=1e-3,N_sample=2048)
        elif config['loss_func'].get('gradient_func') == 'gradients_sparse_grid':
            u_x,u_t,u_xx,u_interior = gradients_sparse_gird(x_interior,f_model,sigma=1e-3,level=3,rule='GQN')
        elif config['loss_func'].get('gradient_func') == 'gradients_auto_diff':
            u_x,u_t,u_xx,u_interior = gradients_auto_diff(x_interior,f_model)
        else:
            raise NotImplementedError
        
        pde_loss = (u_t + 0.5 * sigma**2 * x_interior[:,0:1]**2 * u_xx + r * x_interior[:,0:1] * u_x - r * u_interior).pow(2).mean()
        u_terminal = model(x_terminal)
        payoff = torch.max(x_terminal[:,0:1]-K/200,torch.tensor(0.0))
        terminal_loss = (u_terminal - payoff).pow(2).mean()
        u_boundary_0 = model(x_boundary_0)
        boundary_loss_0 = u_boundary_0.pow(2).mean()
        u_boundary_max = model(x_boundary_max)
        asymptotic_payoff = S_max/200 - K/200 * torch.exp(-r * (T - x_boundary_max[:,1:2]))
        boundary_loss_max = (u_boundary_max - asymptotic_payoff).pow(2).mean()
        loss = pde_loss + terminal_loss + boundary_loss_0 + boundary_loss_max
        
        return loss

    return loss_func