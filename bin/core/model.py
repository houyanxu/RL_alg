import torch.nn as nn
import torch
#TODO: custom model

class FCModel(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super(FCModel,self).__init__()
        self.fc1 = nn.Linear(obs_dim,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,act_dim)

    def forward(self,obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        out = torch.tanh(self.fc3(x))
        return out

