import torch.nn as nn


class HypertrophyClassifier(nn.Module):
    def __init__(self):
        super(HypertrophyClassifier, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(3, 3), stride=1)
        self.maxpool_2_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=15, kernel_size=(2, 2), stride=1)
        self.conv3 = nn.Conv2d(in_channels=15, out_channels=21, kernel_size=(3, 3), stride=1)
        self.conv4 = nn.Conv2d(in_channels=21, out_channels=30, kernel_size=(3, 3), stride=1, padding=1)

        self.linear1 = nn.Linear(4320, 200)
        self.linear2 = nn.Linear(200, 3)

        # bn
        self.conv1_bn = nn.BatchNorm2d(9)
        self.conv2_bn = nn.BatchNorm2d(15)
        self.conv3_bn = nn.BatchNorm2d(21)
        self.conv4_bn = nn.BatchNorm2d(30)

        self.linear1_bn = nn.BatchNorm1d(200)

        self.relu = nn.ReLU()

    def forward(self, x):
        temp = self.relu(self.conv1_bn(self.conv(x)))  # (108, 108)
        temp = self.maxpool_2_2(temp)  # (54,54)
        temp = self.relu(self.conv2_bn(self.conv2(temp)))  # (53, 53)
        temp = self.maxpool_2_2(temp)  # (26,26)
        temp = self.relu(self.conv3_bn(self.conv3(temp)))  # (24, 24)
        temp = self.maxpool_2_2(temp)  # (12, 12)
        temp = self.relu(self.conv4_bn(self.conv4(temp)))  # (12, 12)
        temp = temp.view(-1, 4320)
        temp = self.relu(self.linear1_bn(self.linear1(temp)))
        temp = self.linear2(temp)
        return temp
