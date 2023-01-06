import torch 
from torch import nn 
import torch.nn.functional as F 
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim,64)
        self.layer2 = nn.Linear(64,64)
        self.layer3 = nn.Linear(64,out_dim)
    
    def forward(self,obs):
        if isinstance(obs,np.ndarray):
            obs = torch.from_numpy(obs)
        
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = F.relu(self.layer3(activation2))
        return output




