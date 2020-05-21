import torch
import torch.nn as nn
from data_reader import DataReader
from torch.utils.data import DataLoader
from hypertrophy_dataset import HypertrophyDataset
import torch.optim as optim
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np

def calc_accuracy(loader):
    counter = 0
    correctly_labeled = 0
    topk_counter = 0
    for sample in loader:
        counter += 1
        image = sample['image']
        target = sample['target']
        res = model(image)
        predicted = torch.argmax(res)
        print("-----------------------------------")
        print(target)
        print(torch.topk(res, 3)[1])
        top2 = torch.topk(res, 2)[1]
        if target == predicted:
            correctly_labeled += 1
            topk_counter += 1
        elif predicted in top2:
            topk_counter += 1

    print("Accuracy: {}".format(correctly_labeled/ counter))
    print("TopK: {}".format(topk_counter/ counter))
    print(topk_counter, counter)


def split_data(ratio1, ratio2, data_x, data_y):
    n = len(data_x)
    x_1 = data_x[:int(n * ratio1)]
    y_1 = data_y[:int(n * ratio1)]
    x_2 = data_x[int(n * ratio1):int(n * ratio2)]
    y_2 = data_y[int(n * ratio1):int(n * ratio2)]
    x_3 = data_x[int(n * ratio2):]
    y_3 = data_y[int(n * ratio2):]
    return (x_1, y_1), (x_2, y_2), (x_3, y_3)


def calculate_loss(loader):
    loss_sum = 0.0
    counter = 0
    for sample in loader:
        counter += 1
        image = sample['image']
        target = sample['target']
        predicted = model(image)
        loss = criterion(predicted, target)
        loss_sum += loss.cpu().detach().numpy() / len(sample)
        del loss

    return loss_sum / counter


batch_size = 50
device = torch.device("cuda")
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
model = nn.Sequential(
    model,
    nn.ReLU(),
    nn.Linear(1000, len(DataReader.possible_pathologies) + 1),
)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0)

in_dir = sys.argv[1]
data_reader = DataReader(in_dir)

(train_data, validation_data, test_data) = split_data(0.66, 0.83, data_reader.x, data_reader.y)

dataset = HypertrophyDataset(train_data[0], train_data[1], device)
loader_train = DataLoader(dataset, batch_size)
dataset = HypertrophyDataset(validation_data[0], validation_data[1], device)
loader_validation = DataLoader(dataset, batch_size)
dataset = HypertrophyDataset(test_data[0], test_data[1], device)
loader_test = DataLoader(dataset, 1)

epochs = 1
train_losses = []
validation_losses = []
scheduler = StepLR(optimizer, step_size=6, gamma=0.8)
print("Training has started at {}".format(datetime.now()))
c = 0
for epoch in range(epochs):
    trainloss_for_epoch = 0.0
    counter = 0
    c = 0
    for index, sample in enumerate(loader_train):
        counter += 1
        image = sample['image']
        target = sample['target']
        predicted = model(image)
        loss = criterion(predicted, target)
        if c == 0:
          c += 1
          npimg = image.cpu().detach().numpy()
          print("EEEZ-------------------------------------")
          print(np.count_nonzero(npimg != 0.0))
          print(npimg.shape)
          print("-------------------------------------")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trainloss_for_epoch += loss.cpu().detach().numpy() / len(sample)
    scheduler.step()
    trainloss_for_epoch /= counter
    validationloss_for_epoch = calculate_loss(loader_validation)
    train_losses.append(trainloss_for_epoch)
    validation_losses.append(validationloss_for_epoch)

    if epoch % 5 == 0:
      print("Epoch {} has finished (train loss: {}, validation loss: {}".format(epoch, trainloss_for_epoch,
                                                                              validationloss_for_epoch))

plt.clf()
calc_accuracy(loader_test)
print("Training has finished.")
plt.plot(train_losses, label='train_loss')
plt.plot(validation_losses, label='validation_loss')
plt.legend()
plt.savefig("train_dev_loss.png")
