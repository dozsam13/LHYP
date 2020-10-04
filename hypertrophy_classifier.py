import torch.nn as nn
import torch

class HypertrophyClassifier(nn.Module):
    def __init__(self, device):
        super(HypertrophyClassifier, self).__init__()
        self.device = device

        self.conv = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(2, 2), stride=1)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), stride=1)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=30, out_channels=35, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=35, out_channels=20, kernel_size=(1, 1), stride=1)
        self.maxpool_2_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.LSTM = nn.LSTM(input_size=180, hidden_size=200, num_layers=1, dropout=0.3)

        self.linear = nn.Linear(200, 3)

        # bn
        self.conv1_bn = nn.BatchNorm2d(5)
        self.conv2_bn = nn.BatchNorm2d(10)
        self.conv3_bn = nn.BatchNorm2d(20)
        self.conv4_bn = nn.BatchNorm2d(30)
        self.conv5_bn = nn.BatchNorm2d(35)
        self.conv6_bn = nn.BatchNorm2d(20)

        # dropout
        self.dropout_02 = nn.Dropout(0.2)
        self.dropout_05 = nn.Dropout(0.5)

        self.relu = nn.ReLU()

    def cnn(self, x):
        temp = self.relu(self.conv1_bn(self.conv(x)))  # (108, 108)
        temp = self.maxpool_2_2(temp)  # (54,54)
        temp = self.relu(self.conv2_bn(self.conv2(temp)))  # (53, 53)
        temp = self.maxpool_2_2(temp)  # (26,26)
        temp = self.relu(self.conv3_bn(self.conv3(temp)))  # (24, 24)
        temp = self.maxpool_2_2(temp)  # (12, 12)
        temp = self.relu(self.conv4_bn(self.conv4(temp)))  # (12, 12)
        temp = self.maxpool_2_2(temp)  # (6, 6)
        temp = self.relu(self.conv5_bn(self.conv5(temp)))
        temp = self.maxpool_2_2(temp)  # (3, 3)
        temp = self.relu(self.conv6_bn(self.conv6(temp)))

        return temp.view(-1, 20 * 9)

    def forward(self, x):
        sequence = []
        for i in range(x.shape[1]):
            sequence.append(self.cnn(x[:, i, :, :, :]))
        sequence = torch.cat(sequence).view(x.shape[0], x.shape[1], 180)
        hidden = (torch.zeros(1, x.shape[1], 200, device = self.device), torch.zeros(1, x.shape[1], 200, device = self.device))
        out, _ = self.LSTM(sequence, hidden)
        out = out[:, out.shape[1]-1, :]
        out = self.linear(out)

        return out
