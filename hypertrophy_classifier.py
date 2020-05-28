import torch.nn as nn
import torch

class HypertrophyClassifier(nn.Module):
    def __init__(self):
        super(HypertrophyClassifier, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(3,3), stride=1)     # (c=3, (224,224)) => (c=9, (222, 222))
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=(3,3), stride=1)     # (c=9, (222,222)) => (c=18, (220, 220))
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=(2,2), stride=2)  # (c=18, (220,220)) => (c=18, (110, 110))
        self.conv4 = nn.Conv2d(in_channels=18, out_channels=9, kernel_size=(2,2), stride=1, padding=1)     # (c=9, (110,110)) => (c=9, (110, 110))
        self.conv5 = nn.Conv2d(in_channels=9, out_channels=6, kernel_size=(2,2), stride=1, padding=1) # (c=6, (110,110)) => (c=9, (110, 110))
        self.linear1 = nn.Linear(75264, 200)
        self.linear2 = nn.Linear(200, 3)

        self.relu = nn.ReLU()

    def forward(self, x):
        temp = self.relu(self.conv(x))
        temp = self.relu(self.conv2(temp))
        temp = self.relu(self.conv3(temp))
        temp = self.relu(self.conv4(temp))
        temp = self.relu(self.conv5(temp))
        temp = temp.view(-1, 75264)
        temp = self.relu(self.linear1(temp))
        temp = self.linear2(temp)
        return temp