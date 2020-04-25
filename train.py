import torch
import torch.nn as nn
from hypertrophy_classifier import HypertrophyClassifier
from data_reader import DataReader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from hypertrophy_dataset import HypertrophyDataset
import torch.optim as optim
from datetime import datetime

def split_data(ratio1, ratio2, data_x, data_y):
	n = len(data_x)
	x_1 = data_x[:int(n*ratio1)]
	y_1 = data_y[:int(n*ratio1)]
	x_2 = data_x[int(n*ratio1):int(n*ratio2)]
	y_2 = data_y[int(n*ratio1):int(n*ratio2)]
	x_3 = data_x[int(n*ratio2):]
	y_3 = data_y[int(n*ratio2):]
	return ((x_1, y_1), (x_2, y_2), (x_3, y_3))

def validation(epoch, index):
    counter = 0
    loss_sum = 0.0
    for sample in loader_validation:
        counter += 1
        image = sample['image']
        target = sample['target']
        predicted = model(image)
        print("-------------------------------------------------------------------------")
        print(predicted.size())
        print(target.long().size())
        print(predicted.dtype)
        print(target.dtype)
        loss = criterion(predicted.reshape(-1), target.float())
        loss_sum += loss
        del loss

    loss_mean = loss_sum / counter
    print("Current status at: (epoch: %d, i: %d ) with validation loss: %f"%(epoch, index, loss_mean))

batch_size = 1000
data_reader = DataReader("C:\\dev\\LHYP\\out")
(train_data, validation_data, test_data) = split_data(0.66, 0.83, data_reader.x, data_reader.y)

dataset = HypertrophyDataset(train_data[0], train_data[1])
loader_train = DataLoader(dataset, batch_size)
dataset = HypertrophyDataset(validation_data[0], validation_data[1])
loader_validation = DataLoader(dataset, batch_size)
dataset = HypertrophyDataset(test_data[0], test_data[1])
loader_test = DataLoader(dataset, batch_size)

#device = torch.device("cuda")
model = HypertrophyClassifier()
#model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

epochs = 6
global_counter = 0
validation(0,0)
print("Started training at {}".format(datetime.now()))
for epoch in range(epochs):
    for index, sample in enumerate(loader_train):
        global_counter += 1

        image = sample['image']
        target = sample['target']

        predicted = model(image)
        print(predicted.size())
        print(target.long().size())
        print(predicted.dtype)
        print(target.dtype)
        loss = criterion(predicted.reshape(-1), target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("TRAINLOSS: ", loss/batch_size)

    validation(epoch, index)
    print("\r Current epoch {} and global counter {}".format(epoch, global_counter), end="")
print("Training has finished.")