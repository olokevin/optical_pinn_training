import numpy as np
import copy
import torch
from torch.utils.data import Dataset


def dataio_init_from_config(config, mode):
    device = config['basic'].get('device')

    return BlackScholesDataIO(device=device)


def to_device(data, device, dtype=torch.float32):
    if dtype == torch.float32:
        return torch.tensor(data, requires_grad=True).float().to(device)
    elif dtype == torch.float16:
        return torch.tensor(data, requires_grad=True).half().to(device)
    
def black_scholes_call(S, t, K, r, sigma, T):
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * torch.sqrt(T - t))
    d2 = d1 - sigma * torch.sqrt(T - t)
    N = lambda x: 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    return S * N(d1) - K * torch.exp(-r * (T - t)) * N(d2)


class BlackScholesDataIO(Dataset):
    # def __init__(self, num_interior=1000, num_boundary=100, device=None, dtype=torch.float32):
    def __init__(self, num_interior=100, num_boundary=10, device=None, dtype=torch.float32):
        super(BlackScholesDataIO, self).__init__()
        self.S_max = 200
        self.T = 1
        self.K = 100
        self.r = 0.05
        self.sigma = 0.2
        self.num_interior = num_interior
        self.num_boundary = num_boundary
        self.device = device
        self.dtype = dtype
        self.mode = 'train'

    def __len__(self):
        return 1

    def __getitem__(self, item):
        if self.mode == 'train':
            # Interior points
            S_interior = np.random.uniform(0, self.S_max/200, (self.num_interior, 1))
            t_interior = np.random.uniform(0, self.T, (self.num_interior, 1))
            interior_points = np.hstack((S_interior, t_interior))

            # Terminal condition points
            S_terminal = np.random.uniform(0, self.S_max/200, (self.num_boundary, 1))
            t_terminal = np.full((self.num_boundary, 1), self.T)
            terminal_points = np.hstack((S_terminal, t_terminal))

            # Boundary condition at S = 0
            S_boundary_0 = np.zeros((self.num_boundary, 1))
            t_boundary_0 = np.random.uniform(0, self.T, (self.num_boundary, 1))
            boundary_0_points = np.hstack((S_boundary_0, t_boundary_0))

            # Boundary condition at S = S_max
            S_boundary_max = np.full((self.num_boundary, 1), self.S_max/200)
            t_boundary_max = np.random.uniform(0, self.T, (self.num_boundary, 1))
            boundary_max_points = np.hstack((S_boundary_max, t_boundary_max))

            all_points = np.vstack((interior_points, terminal_points, boundary_0_points, boundary_max_points))
            
            return to_device(copy.deepcopy(all_points), self.device, dtype=self.dtype)
        
        elif self.mode == 'eval':
            # Create a grid for evaluation
            S = np.linspace(0, self.S_max, 100)
            t = np.linspace(0, self.T, 100)
            S, t = np.meshgrid(S, t)
            ST_eval = np.stack((S.flatten(), t.flatten()), axis=-1)
            ST_eval_tensor = to_device(copy.deepcopy(ST_eval), self.device, dtype=self.dtype)
            
            # Calculate analytical solution
            u_test = black_scholes_call(
                ST_eval_tensor[:, 0:1],
                ST_eval_tensor[:, 1:2],
                self.K,
                self.r,
                self.sigma,
                self.T
            ).cpu().detach().numpy()
            
            return ST_eval_tensor/200, u_test/200

    def train(self):
        self.mode = 'train'
        return next(iter(self))

    def eval(self):
        self.mode = 'eval'
        return next(iter(self))
