import torch.nn as nn


class SegmentOrderModel(nn.Module):
    def __init__(self):
        super(SegmentOrderModel, self).__init__()
        c1c2 = 8
        c2c3 = 15
        c3c4 = 18
        c4c5 = 35
        c5c6 = 30
        c6l1 = 50
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=c1c2, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c1c2, out_channels=c2c3, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=c2c3, out_channels=c3c4, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=c3c4, out_channels=c4c5, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=c4c5, out_channels=c5c6, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=c5c6, out_channels=c6l1, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool_2_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear1 = nn.Linear(50*9, 4)

        # bn
        self.conv1_bn = nn.BatchNorm2d(c1c2)
        self.conv2_bn = nn.BatchNorm2d(c2c3)
        self.conv3_bn = nn.BatchNorm2d(c3c4)
        self.conv4_bn = nn.BatchNorm2d(c4c5)
        self.conv5_bn = nn.BatchNorm2d(c5c6)
        self.conv6_bn = nn.BatchNorm2d(c6l1)
        self.linear1_bn = nn.BatchNorm1d(4)

        # dropout
        self.dropout_02 = nn.Dropout(0.2)
        self.dropout_05 = nn.Dropout(0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        temp = self.relu(self.conv1(x))  # (110, 110)
        temp = self.maxpool_2_2(temp)  # (55, 55)
        temp = self.relu(self.conv2(temp))  # (53, 53)
        temp = self.maxpool_2_2(temp)  # (23, 23)
        temp = self.relu(self.conv3(temp))  # (46, 46)
        temp = self.maxpool_2_2(temp)
        temp = self.relu(self.conv4(temp))  # (21, 21)
        temp = self.maxpool_2_2(temp)  # (8, 8)
        temp = self.relu(self.conv5(temp))  # (17, 17)
        temp = self.maxpool_2_2(temp)  # (19,19)
        temp = self.relu(self.conv6(temp))  # (10, 10)
        temp = temp.view(-1, 50*9)
        temp = self.linear1(temp)

        return temp