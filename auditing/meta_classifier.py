import torch 
import torch.nn as nn
import torch.nn.functional as F


class MLP_B(nn.Module):
    def __init__(self, dim_in):
        super(MLP_B, self).__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(self.dim_in, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1,self.dim_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP_O(nn.Module):
    def __init__(self, input_size, dim_in):
        super(MLP_O, self).__init__()
        
        self.input = nn.Parameter(torch.zeros(*input_size).normal_()*1e-3)
        self.dim_in = dim_in
        self.fc1 = nn.Linear(self.dim_in, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1,self.dim_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
