import torch.nn as nn
import torch

class HypertrophyClassifier(nn.Module):
    def __init__(self):
        super(HypertrophyClassifier, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)     # (c=3, (200,200)) => (c=1, (200, 200))
        
        self.linear = nn.Linear(40000, 1)  # in:40000 out:1

        self.relu = nn.ReLU()

    def forward(self, x):
        temp = self.conv(x.float())
        temp = self.relu(temp)
        temp = temp.view(-1, 40000)
        temp = self.linear(temp)
        return temp