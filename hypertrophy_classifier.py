import torch.nn as nn


class HypertrophyClassifier(nn.Module):
    def __init__(self, segment_order_model):
        super(HypertrophyClassifier, self).__init__()
        self.segment_order_model = segment_order_model
        self.linear1 = nn.Linear(9 * 80, 3)

        # bn
        self.linear1_bn = nn.BatchNorm1d(3)

        self.dropout_02 = nn.Dropout(0.2)
        self.dropout_05 = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        temp = self.segment_order_model(x)
        temp = self.dropout_05(temp)
        temp = self.linear1(temp)

        return temp