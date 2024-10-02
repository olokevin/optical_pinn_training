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

    dim = X.shape[1]
    u = f(X)
    u_g = gradients(u, X)[0]
    u_x,u_t = u_g[:, 0:dim-1],u_g[:, dim-1:dim]
    laplacian = 0
    for i in range(dim-1):
        laplacian = laplacian + gradients(u_x[:,i:i+1],X)[0][:,i:i+1]

    return u_x,u_t,laplacian

def gradients_cd2(X,f,h):

    batch_size = X.shape[0]
    dim = X.shape[1]
    u = f(X)
    u_x = torch.zeros((batch_size,dim-1),device=X.device)
    u_xx = torch.zeros((batch_size,dim-1),device=X.device)

    for i in range(dim-1):
        e_xi = direction_vector(X,h,i)
        u_plus = f(X+e_xi)
        u_minus = f(X-e_xi)
        u_xi = (u_plus - u_minus) / (2*h)
        u_xxi = (u_plus+u_minus-2*u) / (h**2)
        u_x[:,i:i+1] = u_xi
        u_xx[:,i:i+1] = u_xxi
    
    e_t = direction_vector(X,h,dim-1)
    u_t = (f(X+e_t) - f(X-e_t)) / (2*h)
    laplacian = torch.sum(u_xx,1,keepdim=True)

    return u_x,u_t,laplacian


def gradients_stein(X,f,sigma,N_sample):

    batch_size = X.shape[0]
    dim = X.shape[1]
    x,t = X[:, 0:dim-1], X[:, dim-1:dim]
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

    e_x = e_x.reshape(N_sample,batch_size,-1)
    e_t = e_t.reshape(N_sample,batch_size,1)

    u_t = ((sample_u_plus-sample_u_minus)*e_t)/(2*sigma*sigma)
    u_t = torch.mean(u_t, dim=0)

    u_x = ((sample_u_plus-sample_u_minus)*e_x)/(2*sigma*sigma)
    u_x = torch.mean(u_x, dim=0)
    u_xx = (e_x**2-sigma**2)*(sample_u_plus+sample_u_minus-2*f(X).reshape(1,batch_size,1))/(2*(sigma**4))
    u_xx = torch.mean(u_xx, dim=0)
    laplacian = torch.sum(u_xx,dim=1,keepdim=True)
    return u_x,u_t,laplacian


def gradients_sparse_gird(X,f,sigma):

    data_path = os.path.join(os.getcwd(), 'GQN_d21_l3.asc')
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
    laplacian = torch.sum(u_xx,dim=1,keepdim=True)
    
    return u_x,u_t,laplacian



# def loss_func_init_from_config(config):

#     def loss_func(model, dataset, inputs=None, return_loss_reduction='mean'):
#         if inputs is not None:
#             x = inputs
#         else:
#             x = dataset.train()
#         f_model = lambda X: torch.sum(X[:,0:20],1,keepdim=True)+(1-X[:,20:21])*model(X)

#         if config.get('loss_func', 'gradient_func') == 'gradients_auto_diff':
#             u_x,u_t,laplacian = gradients_auto_diff(x,f_model)
#         elif config.get('loss_func', 'gradient_func') == 'gradients_cd2':
#             u_x,u_t,laplacian = gradients_cd2(x,f_model,h=1e-2)
#         elif config.get('loss_func', 'gradient_func') == 'gradients_stein':
#             u_x,u_t,laplacian = gradients_stein(x,f_model,sigma=1e-1,N_sample=1024)
#         elif config.get('loss_func', 'gradient_func') == 'gradients_sparse_grid':
#             u_x,u_t,laplacian = gradients_sparse_gird(x,f_model,sigma=1e-1)
#         else:
#             raise ValueError('wrong gradient_func setting in .ini')


        
#         loss = u_t + laplacian - 0.05*torch.sum(u_x**2,1,keepdim=True) + 2
#         return (loss ** 2).mean()

#     return loss_func

def loss_func_init_from_config(config):

    sigma = config['loss_func'].getfloat('sigma')

    def loss_func(model, dataset, inputs=None, return_loss_reduction='mean'):
        if inputs is not None:
            x = inputs
        else:
            x = dataset.train()
        f_model = lambda X: torch.sum(X[:,0:20],1,keepdim=True)+(1-X[:,20:21])*model(X)

        if config.get('loss_func', 'gradient_func') == 'gradients_auto_diff':
            u_x,u_t,laplacian = gradients_auto_diff(x,f_model)
        elif config.get('loss_func', 'gradient_func') == 'gradients_cd2':
            u_x,u_t,laplacian = gradients_cd2(x,f_model,h=sigma)
        elif config.get('loss_func', 'gradient_func') == 'gradients_stein':
            u_x,u_t,laplacian = gradients_stein(x,f_model,sigma=sigma,N_sample=1024)
        elif config.get('loss_func', 'gradient_func') == 'gradients_sparse_grid':
            u_x,u_t,laplacian = gradients_sparse_gird(x,f_model,sigma=sigma)
        else:
            raise ValueError('wrong gradient_func setting in .ini')


        
        loss = u_t + laplacian - 0.05*torch.sum(u_x**2,1,keepdim=True) + 2
        return (loss ** 2).mean()

    return loss_func

