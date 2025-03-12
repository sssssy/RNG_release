
import torch
from torch import nn
import numpy as np

class DepthMLP(nn.Module):
    
    def __init__(self, in_channels, checkpoint=None, depth_mlp_modifier=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 32), ## view direction (3) => shading point depth offset (1)
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.depth_mlp_modifier = depth_mlp_modifier
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        if checkpoint:
            print(f"Restoring DepthMLP from checkpoint: {checkpoint}")
            self.restore_from_checkpoint(checkpoint)
        
    def forward(self, x):
        return self.layers(x) * self.depth_mlp_modifier
    
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