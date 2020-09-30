import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import util.plot_util as plot_util
from data_reader import DataReader
from hypertrophy_classifier import HypertrophyClassifier
from hypertrophy_dataset import HypertrophyDataset
from datetime import datetime


def calculate_accuracy(loader, model):
    correctly_labeled = 0
    for sample in loader:
        sequence = sample['sequence']
        target = sample['target']
        res = model(sequence)
        predicted = torch.argmax(res, dim=1)
        correctly_labeled += torch.eq(target, predicted).sum().cpu().detach().numpy()

    return correctly_labeled / len(loader.dataset)


def split_data(ratio1, ratio2, data_x, data_y):
    n = len(data_x)
    test_x = data_x[int(n * ratio2):]
    test_y = data_y[int(n * ratio2):]
    data_dev_train_x = data_x[:int(n * ratio2)]
    data_dev_train_y = data_y[:int(n * ratio2)]
    n = len(data_dev_train_x)
    indices = list(range(len(data_dev_train_x)))
    np.random.shuffle(indices)
    train_indices = indices[:int(n * ratio1)]
    dev_indices = indices[int(n * ratio1):int(n * ratio2)]
    train_x = [data_dev_train_x[idx] for idx in train_indices]
    train_y = [data_dev_train_y[idx] for idx in train_indices]
    dev_x = [data_dev_train_x[idx] for idx in dev_indices]
    dev_y = [data_dev_train_y[idx] for idx in dev_indices]

    return (train_x, train_y), (dev_x, dev_y), (test_x, test_y)


def calculate_loss(loader, model, criterion):
    loss_sum = 0.0
    counter = 0
    for sample in loader:
        counter += 1
        sequence = sample['sequence']
        target = sample['target']
        predicted = model(sequence)
        loss = criterion(predicted, target)
        loss_sum += loss.cpu().detach().numpy()
        del loss

    return loss_sum / counter


def train_model():
    device = torch.device("cuda")
    model = HypertrophyClassifier()

    model.to(device)

    in_dir = sys.argv[1]
    data_reader = DataReader(in_dir)

    batch_size = 20
    (train_data, validation_data, test_data) = split_data(0.66, 0.83, data_reader.x, data_reader.y)
    dataset = HypertrophyDataset(train_data[0], train_data[1], device)
    loader_train = DataLoader(dataset, batch_size)
    dataset = HypertrophyDataset(train_data[0], train_data[1], device)
    loader_train_accuracy = DataLoader(dataset, batch_size)
    dataset = HypertrophyDataset(validation_data[0], validation_data[1], device)
    loader_validation = DataLoader(dataset, batch_size)

    epochs = 3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_losses = []
    train_accuracies = []
    dev_losses = []
    dev_accuracies = []
    print("Training has started at {}".format(datetime.now()))
    for epoch in range(epochs):
        trainloss_for_epoch = 0.0
        counter = 0
        for sample in loader_train:
            sequence = sample['sequence']
            target = sample['target']
            predicted = model(sequence)
            loss = criterion(predicted, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainloss_for_epoch += loss.cpu().detach().numpy()
            counter += 1
        trainloss_for_epoch /= counter
        train_losses.append(trainloss_for_epoch)
        train_accuracies.append(calculate_accuracy(loader_train_accuracy, model))
        dev_losses.append(calculate_loss(loader_validation, model, criterion))
        dev_accuracies.append(calculate_accuracy(loader_validation, model))
    print("Training has finished at {}".format(datetime.now()))

    plot_util.plot_data(train_losses, 'train_loss', dev_losses, 'dev_loss', "loss.png")
    plot_util.plot_data(train_accuracies, 'train accuracy', dev_accuracies, 'dev accuracy', "accuracy.png")

if __name__ == '__main__':
    train_model()

