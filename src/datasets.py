import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

"""
Torch datasets for simulation data
"""

class DiscreteDynamicsDataset(Dataset):
    """ 
    Dataset containing one-step prediction pairs ((x[k], u[k]), (x[k+1])) 
    """
    def __init__(self, x, u, transform=None, drop_nans=False):
        """
        Input: 
            x: (N+1,nx) or (n_batches,N+1,nx) array with sequence of states
            u: (N,nu) or (n_batches,N,nu) array with sequence of inputs
        """
        
        assert x.ndim == u.ndim
        assert x.shape[-2] == u.shape[-2] + 1
        
        if x.ndim == 2:
            x = x.reshape(1,-1,nx)
            u = u.reshape(1,-1,nu)
        
        nx = x.shape[-1]
        nu = u.shape[-1]
        
        if drop_nans:
            valid_batches = ~np.isnan(x).any(axis=-1).any(axis=-1)
            dx = (x[valid_batches,1:] - x[valid_batches,:-1]).reshape(-1,nx)
            x = x[valid_batches,:-1].reshape(-1,nx)
            u = u[valid_batches].reshape(-1,nu)
            valid = ~np.isnan(dx).any(axis=-1)
            print('Number of valid batches: ', np.sum(valid_batches))
        else:
            dx = (x[:,1:] - x[:,:-1]).reshape(-1,nx)
            x = x[:,:-1].reshape(-1,nx)
            u = u.reshape(-1,nu)

            valid = ~np.isnan(dx).any(axis=-1)
        
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        if not torch.is_tensor(u):
            u = torch.from_numpy(u).float()
        if not torch.is_tensor(dx):
            dx = torch.from_numpy(dx).float()
        

        self.nx = nx
        self.nu = nu
        self.x = x[valid]
        self.u = u[valid]
        self.dx = dx[valid]
        self.transform = transform

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x': self.x[idx], 'u': self.u[idx], 'dx': self.dx[idx]}

        if self.transform:
            return self.transform(sample)

        return sample


