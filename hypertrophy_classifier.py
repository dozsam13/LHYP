import torch.nn as nn


class HypertrophyClassifier(nn.Module):
    def __init__(self):
        super(HypertrophyClassifier, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(2, 2), stride=1)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(3, 3), stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=44, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=44, out_channels=75, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool_2_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear1 = nn.Linear(2700, 200)
        self.linear2 = nn.Linear(200, 3)

        # bn
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4_bn = nn.BatchNorm2d(44)
        self.conv5_bn = nn.BatchNorm2d(75)

        # dropout
        self.dropout_02 = nn.Dropout(0.2)
        self.dropout_05 = nn.Dropout(0.5)

        self.linear1_bn = nn.BatchNorm1d(200)

        self.relu = nn.ReLU()

    def forward(self, x):
        temp = self.relu(self.conv1_bn(self.conv(x)))  # (108, 108)
        temp = self.dropout_02(temp)
        temp = self.maxpool_2_2(temp)  # (54,54)
        temp = self.relu(self.conv2_bn(self.conv2(temp)))  # (53, 53)
        temp = self.dropout_05(temp)
        temp = self.maxpool_2_2(temp)  # (26,26)
        temp = self.relu(self.conv3_bn(self.conv3(temp)))  # (24, 24)
        temp = self.dropout_05(temp)
        temp = self.maxpool_2_2(temp)  # (12, 12)
        temp = self.relu(self.conv4_bn(self.conv4(temp)))  # (12, 12)
        temp = self.dropout_05(temp)
        temp = self.maxpool_2_2(temp)  # (6, 6)
        temp = self.relu(self.conv5_bn(self.conv5(temp)))
        temp = self.dropout_05(temp)
        temp = temp.view(-1, 2700)
        temp = self.relu(self.linear1_bn(self.linear1(temp)))
        temp = self.dropout_05(temp)
        temp = self.linear2(temp)

        return temp
