import torch
import torch.nn as nn
from data_reader import DataReader
from torch.utils.data import DataLoader
from hypertrophy_dataset import HypertrophyDataset
from hypertrophy_classifier import HypertrophyClassifier
import torch.optim as optim
from datetime import datetime
import sys
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torchvision import transforms
import util.plot_util as plot_util


def calc_accuracy(loader, model):
    correctly_labeled = 0
    for sample in loader:
        image = sample['image']
        target = sample['target']
        res = model(image)
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
        image = sample['image']
        target = sample['target']
        predicted = model(image)
        loss = criterion(predicted, target)
        loss_sum += loss.cpu().detach().numpy()
        del loss

    return loss_sum / counter


def manage_batchnorm(model, state):
    for child in model.children():
        if type(child) is nn.BatchNorm2d or type(child) is nn.BatchNorm1d:
            child.track_running_stats = state


def train_model(config):
    batch_size = 70
    device = torch.device("cuda")
    model = HypertrophyClassifier(config["c1c2"], config["c2c3"], config["c3c4"], config["c4c5"], config["c5c6"], config["c6c7"], config["c7l1"])

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=config["weight_decay"], lr=config["lr"])

    in_dir = sys.argv[1]
    data_reader = DataReader(in_dir)

    (train_data, validation_data, test_data) = split_data(0.66, 0.83, data_reader.x, data_reader.y)

    augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine([-45, 45]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = HypertrophyDataset(train_data[0], train_data[1], device, augmenter)
    loader_train = DataLoader(dataset, batch_size)
    loader_train_accuracy = DataLoader(dataset, batch_size)
    dataset = HypertrophyDataset(validation_data[0], validation_data[1], device)
    loader_validation = DataLoader(dataset, 1)
    # dataset = HypertrophyDataset(test_data[0], test_data[1], device)
    # loader_test = DataLoader(dataset, batch_size)

    epochs = 250
    train_losses = []
    dev_losses = []
    train_accuracies = []
    dev_accuracies = []
    scheduler = StepLR(optimizer, step_size=60, gamma=0.85)
    print("Training has started at {}".format(datetime.now()))
    for epoch in range(epochs):
        trainloss_for_epoch = 0.0
        counter = 0
        for index, sample in enumerate(loader_train):
            counter += 1
            image = sample['image']
            target = sample['target']

            predicted = model(image)
            loss = criterion(predicted, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainloss_for_epoch += loss.cpu().detach().numpy()
        scheduler.step()

        trainloss_for_epoch /= counter
        validationloss_for_epoch = calculate_loss(loader_validation, model, criterion)
        train_losses.append(trainloss_for_epoch)
        dev_losses.append(validationloss_for_epoch)
        train_accuracies.append(calc_accuracy(loader_train_accuracy, model))
        dev_accuracies.append(calc_accuracy(loader_validation, model))

    manage_batchnorm(model, False)
    model.eval()
    print("Training has finished.")
    print("Dev accuracy: ", calc_accuracy(loader_validation, model))
    plot_util.plot_confusion_matrix(loader_validation, model)
    plot_util.plot_data(train_losses, 'train_loss', dev_losses, 'dev_loss', "loss.png")
    plot_util.plot_data(train_accuracies, 'train accuracy', dev_accuracies, 'dev accuracy', "accuracy.png")

    return calculate_loss(loader_validation, model, criterion)

def train_multiple(config):
    dev_losses = []
    for i in range(15):
        dev_losses.append(train_model(config))
    return min(dev_losses)

if __name__ == '__main__':
    print(train_model({'weight_decay': 0.0, 'lr': 0.10045252445237711, 'c1c2': 13, 'c2c3': 20, 'c3c4': 42, 'c4c5': 46, 'c5c6': 46, 'c6c7': 60, 'c7l1': 86}))