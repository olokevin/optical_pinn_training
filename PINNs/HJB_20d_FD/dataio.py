import numpy as np
import copy
import torch
from torch.utils.data import Dataset


def dataio_init_from_config(config, mode):
    batch_size = config['dataio'].getint('batch_size')
    device = config['basic'].get('device')

    return HJBDataIO(batch_size=batch_size, device=device)


def to_device(data, device, dtype=torch.float32):
    if dtype == torch.float32:
        return torch.tensor(data, requires_grad=True).float().to(device)
    elif dtype == torch.float16:
        return torch.tensor(data, requires_grad=True).half().to(device)


class HJBDataIO(Dataset):

    def __init__(self, batch_size=100, device=None, dtype=torch.float32):
        super(HJBDataIO, self).__init__()

        self.batch_size = batch_size    
        self.x_test = np.random.uniform(0, 1, size=(10000, 21))
        self.u_test = np.sum(self.x_test[:,0:20],axis=1, keepdims=True) + 1 - self.x_test[:,20:21]

        self.mode = 'train'
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return 1

    def __getitem__(self, item):
        if self.mode == 'train':
            return to_device(copy.deepcopy(np.random.uniform(0, 1, size=(self.batch_size, 21))), self.device, dtype=self.dtype)
        elif self.mode == 'eval':
            return to_device(copy.deepcopy(self.x_test), self.device, dtype=self.dtype), self.u_test

    def train(self):
        self.mode = 'train'
        return next(iter(self))

    def eval(self):
        self.mode = 'eval'
        return next(iter(self))