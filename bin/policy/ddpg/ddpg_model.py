import torch.nn as nn
import torch
#TODO: custom model

class Actor(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(obs_dim,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,act_dim)

    def forward(self,obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        out = torch.tanh(self.fc3(x))
        return out

    def get_gradient(self):
        grads = []
        for p in self.parameters():
            grads.append(None if p.grad is None else p.grad.data.numpy())
        return grads

    def set_gradients(self,gradients):
        for g,p in zip(gradients,self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

class Critic(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(obs_dim+act_dim,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,1)

    def forward(self,obs,act):
        x = torch.relu(self.fc1(torch.cat([obs,act],dim=-1)))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return torch.squeeze(out,-1)

    def get_gradient(self):
        grads = []
        for p in self.parameters():
            grads.append(None if p.grad is None else p.grad.data.numpy())
        return grads

    def set_gradients(self,gradients):
        for g,p in zip(gradients,self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)