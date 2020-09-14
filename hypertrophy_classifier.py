import torch.nn as nn


class HypertrophyClassifier(nn.Module):
    def __init__(self, c1c2, c2c3, c3c4, c4c5, c5c6, c6c7, c7l1):
        super(HypertrophyClassifier, self).__init__()
        self.c7l1 = c7l1

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=c1c2, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c1c2, out_channels=c2c3, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=c2c3, out_channels=c3c4, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=c3c4, out_channels=c4c5, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=c4c5, out_channels=c5c6, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=c5c6, out_channels=c6c7, kernel_size=(3, 3), stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=c6c7, out_channels=c7l1, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool_2_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.maxpool_3_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1)
        self.avgpool_3_1 = nn.AvgPool2d(kernel_size=(3, 3), stride=1)

        self.linear1 = nn.Linear(9 * c7l1, 3)

        # bn
        self.conv1_bn = nn.BatchNorm2d(c1c2)
        self.conv2_bn = nn.BatchNorm2d(c2c3)
        self.conv3_bn = nn.BatchNorm2d(c3c4)
        self.conv4_bn = nn.BatchNorm2d(c4c5)
        self.conv5_bn = nn.BatchNorm2d(c5c6)
        self.conv6_bn = nn.BatchNorm2d(c6c7)
        self.conv7_bn = nn.BatchNorm2d(c7l1)

        # dropout
        self.dropout_02 = nn.Dropout(0.1)
        self.dropout_05 = nn.Dropout(0.3)

        self.relu = nn.ReLU()

    def max_avg_pool(self, temp):
        temp_max = self.maxpool_3_1(temp)
        temp_avg = self.avgpool_3_1(temp)

        return temp_max + temp_avg

    def forward(self, x):
        temp = self.relu(self.conv1_bn(self.conv1(x)))  # (108, 108)

        temp = self.maxpool_2_2(temp)  # (54, 54)

        temp = self.dropout_02(temp)
        temp = self.relu(self.conv2_bn(self.conv2(temp)))  # (53, 53)

        temp = self.maxpool_2_2(temp)  # (23, 23)

        temp = self.dropout_02(temp)
        temp = self.relu(self.conv3_bn(self.conv3(temp)))  # (46, 46)

        temp = self.max_avg_pool(temp)

        temp = self.dropout_02(temp)
        temp = self.relu(self.conv4_bn(self.conv4(temp)))  # (21, 21)

        temp = self.maxpool_2_2(temp)  # (8, 8)

        temp = self.dropout_02(temp)
        temp = self.relu(self.conv5_bn(self.conv5(temp)))  # (17, 17)

        temp = self.max_avg_pool(temp)  # (19,19)

        temp = self.dropout_02(temp)
        temp = self.relu(self.conv6_bn(self.conv6(temp)))  # (8, 8)

        temp = self.maxpool_2_2(temp)  # (4, 4)

        temp = self.dropout_02(temp)
        temp = self.relu(self.conv7_bn(self.conv7(temp)))  # (4, 4)
        temp = self.max_avg_pool(temp)

        temp = self.dropout_05(temp)

        temp = temp.view(-1, 9 * self.c7l1)
        temp = self.linear1(temp)

        return temp

# 'c1c2': 12, 'c2c3': 14, 'c3c4': 34, 'c4c5': 34, 'c5c6': 42, 'c6c7': 58, 'c7l1': 64})