import torch
import torch.nn as nn
from hypertrophy_classifier import HypertrophyClassifier
from data_reader import DataReader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from hypertrophy_dataset import HypertrophyDataset
import torch.optim as optim
from datetime import datetime
import sys
import matplotlib.pyplot as plt

def split_data(ratio1, ratio2, data_x, data_y):
	n = len(data_x)
	x_1 = data_x[:int(n*ratio1)]
	y_1 = data_y[:int(n*ratio1)]
	x_2 = data_x[int(n*ratio1):int(n*ratio2)]
	y_2 = data_y[int(n*ratio1):int(n*ratio2)]
	x_3 = data_x[int(n*ratio2):]
	y_3 = data_y[int(n*ratio2):]
	return ((x_1, y_1), (x_2, y_2), (x_3, y_3))

def validation():
    loss_sum = 0.0
    counter = 0
    for sample in loader_validation:
        counter += 1
        image = sample['image']
        target = sample['target']
        predicted = model(image)
        loss = criterion(predicted.reshape(-1), target)
        loss_sum += loss.detach().numpy() / len(sample)
        del loss

    return loss_sum / counter

batch_size = 1000
device = torch.device("cuda")
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
model = nn.Sequential(
    model,
    nn.Linear(1000, 1),
    nn.Sigmoid()
)
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

in_dir = sys.argv[1]
data_reader = DataReader(in_dir)

(train_data, validation_data, test_data) = split_data(0.66, 0.83, data_reader.x, data_reader.y)

dataset = HypertrophyDataset(train_data[0], train_data[1], device)
loader_train = DataLoader(dataset, batch_size)
dataset = HypertrophyDataset(validation_data[0], validation_data[1], device)
loader_validation = DataLoader(dataset, batch_size)
dataset = HypertrophyDataset(test_data[0], test_data[1], device)
loader_test = DataLoader(dataset, batch_size)

epochs = 10
train_losses = []
validation_losses = []
print("Training has started at {}".format(datetime.now()))
for epoch in range(epochs):
    trainloss_for_epoch = 0.0
    counter = 0
    for index, sample in enumerate(loader_train):
        counter += 1
        image = sample['image']
        target = sample['target']

        predicted = model(image)
        loss = criterion(predicted.reshape(-1), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trainloss_for_epoch += loss.detach().numpy() / len(sample)
    trainloss_for_epoch /= counter
    validationloss_for_epoch = validation()
    train_losses.append(trainloss_for_epoch)
    validation_losses.append(validationloss_for_epoch)

    print("Epoch {} has finished (train loss: {}, validation loss: {}".format(epoch, trainloss_for_epoch, validationloss_for_epoch))
print("Training has finished.")
plt.plot(train_losses, label='train_loss')
plt.plot(validation_losses, label='validation_loss')
plt.legend()
plt.savefig("train_valid.png")