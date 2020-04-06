import torch.nn as nn
import torch
#TODO: custom model

class FCModel(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super(FCModel,self).__init__()
        self.fc1 = nn.Linear(obs_dim,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,act_dim)

    def forward(self,obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        out = self.fc3(x)
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


