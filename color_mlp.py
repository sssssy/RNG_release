
import torch
from torch import nn
import numpy as np

class ColorMLP(nn.Module):
    
    def __init__(self, in_channels, checkpoint=None):
        super().__init__()
        self.in_channels = in_channels
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        if checkpoint:
            print(f"Restoring ColorMLP from checkpoint: {checkpoint}")
            self.restore_from_checkpoint(checkpoint)
        
    def forward(self, x):
        return self.layers(x) * x[..., -1:] ## light intensity
    
    def capture(self,):
        return {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
    
    def restore_from_checkpoint(self, ckpt_path):
        ckpt, iteration = torch.load(ckpt_path)
        self.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        
class Residual(nn.Module):
    
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    
    def forward(self, x):
        return x + self.layers(x)
        
def gaussian_1d(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def normalize(xyz):
    return xyz / (torch.norm(xyz, dim=-1, keepdim=True) + 1e-6)

def positional_encoding(x, levels=4):
    if levels == 0:
        return x
    encoding = torch.cat([
        x,
        *[torch.sin(x * 2 ** i) for i in range(levels)],
        *[torch.cos(x * 2 ** i) for i in range(levels)],
    ], dim=-1)
    return encoding

def positional_encoding_2d(xyz, levels=4):
    if levels == 0:
        return xyz
    xy = xyz[..., :2]
    encoding = torch.cat([
        xy,
        *[torch.sin(xy * 2 ** i) for i in range(levels)],
        *[torch.cos(xy * 2 ** i) for i in range(levels)],
    ], dim=-1)
    return encoding

def positional_encoding_3d(xyz, levels=4):
    if levels == 0:
        return xyz
    encoding = torch.cat([
        xyz,
        *[torch.sin(xyz * 2 ** i) for i in range(levels)],
        *[torch.cos(xyz * 2 ** i) for i in range(levels)],
    ], dim=-1)
    return encoding
        
def one_blob_encoding(xyz, dims=16):
    if dims == 0:
        return xyz
    ## [N, 3] -> [N, dims]
    N = xyz.shape[0]
    dims = dims // 2
    theta = torch.acos(xyz[..., 2:3])
    phi = torch.atan2(xyz[..., 1:2], xyz[..., 0:1])
    ## one-blob encoding
    theta_blob = torch.arange(0, np.pi / 2, np.pi / 2 / dims, dtype=torch.float32, device=xyz.device).reshape(1, -1).expand(N, -1)
    phi_blob = torch.arange(-np.pi, np.pi, 2*np.pi/dims, dtype=torch.float32, device=xyz.device).reshape(1, -1).expand(N, -1)
    phi_blob = torch.where(phi_blob < phi - np.pi, phi_blob + 2 * np.pi, phi_blob) ## make phi_blob loop around
    theta_blob = gaussian_1d(theta_blob, theta, np.pi / 2 / 24)
    phi_blob = gaussian_1d(phi_blob, phi, 2*np.pi / 24)
    return torch.cat([theta_blob, phi_blob], dim=-1)