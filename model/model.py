import torch
import torch.nn as nn
import torch.nn.functional as F

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc1 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        return x
