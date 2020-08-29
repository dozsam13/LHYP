import torch.nn as nn


class HypertrophyClassifier(nn.Module):
    def __init__(self, c1c2, c2c3, c3c4, c4c5, c5c6):
        super(HypertrophyClassifier, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=c1c2, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=c1c2, out_channels=c2c3, kernel_size=(2, 2), stride=1)
        self.conv3 = nn.Conv2d(in_channels=c2c3, out_channels=c3c4, kernel_size=(3, 3), stride=1)
        self.conv4 = nn.Conv2d(in_channels=c3c4, out_channels=c4c5, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=c4c5, out_channels=c5c6, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=c5c6, out_channels=80, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool_2_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear1 = nn.Linear(720, 3)
        # self.linear2 = nn.Linear(200, 3)

        # bn
        self.conv1_bn = nn.BatchNorm2d(c1c2)
        self.conv2_bn = nn.BatchNorm2d(c2c3)
        self.conv3_bn = nn.BatchNorm2d(c3c4)
        self.conv4_bn = nn.BatchNorm2d(c4c5)
        self.conv5_bn = nn.BatchNorm2d(c5c6)
        self.conv6_bn = nn.BatchNorm2d(80)
        self.linear1_bn = nn.BatchNorm1d(200)

        # dropout
        self.dropout_02 = nn.Dropout(0.2)
        self.dropout_05 = nn.Dropout(0.5)

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
        temp = self.maxpool_2_2(temp)  # (3, 3)
        temp = self.relu(self.conv6_bn(self.conv6(temp)))
        temp = self.dropout_02(temp)
        temp = temp.view(-1, 720)
        temp = self.linear1(temp)
        #temp = self.dropout_02(temp)
        #temp = self.linear2(temp)

        return temp
