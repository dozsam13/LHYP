import torch
import torch.nn as nn
from data_reader import DataReader
from torch.utils.data import DataLoader
from ssl.segment_order_model import SegmentOrderModel
import torch.optim as optim
from datetime import datetime
import sys
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torchvision import transforms
import util.plot_util as plot_util
from ssl.puzzle_shuffle import PuzzleShuffle
from ssl.puzzle_dataset import PuzzleDataset
import pathlib
import os

def calc_accuracy(loader, model):
    correctly_labeled = 0
    for sample in loader:
        image = sample['image']
        target = sample['target']
        res = model(image)
        torch.sort(res, dim=1)
        predicted = torch.round(res)
        correctly_labeled += torch.eq(target, predicted).sum().cpu().detach().numpy()

    return correctly_labeled / (len(loader.dataset) * next(iter(loader))["target"].size()[1])


def split_data(ratio1, ratio2, data_x):
    n = len(data_x)
    test_x = data_x[int(n * ratio2):]
    data_dev_train_x = data_x[:int(n * ratio2)]
    n = len(data_dev_train_x)
    indices = list(range(len(data_dev_train_x)))
    np.random.shuffle(indices)
    train_indices = indices[:int(n * ratio1)]
    dev_indices = indices[int(n * ratio1):int(n * ratio2)]
    train_x = [data_dev_train_x[idx] for idx in train_indices]
    dev_x = [data_dev_train_x[idx] for idx in dev_indices]

    return train_x, dev_x, test_x


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


def train_model():
    batch_size = 70
    device = torch.device("cuda")
    model = SegmentOrderModel()

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    in_dir = sys.argv[1]
    data_reader = DataReader(in_dir)

    (train_data, validation_data, test_data) = split_data(0.66, 0.83, data_reader.x)

    augmenter = transforms.Compose([
        PuzzleShuffle(2, 110)
    ])

    dataset = PuzzleDataset(train_data, device, augmenter)
    loader_train = DataLoader(dataset, batch_size)
    loader_train_accuracy = DataLoader(dataset, batch_size)
    dataset = PuzzleDataset(validation_data, device)
    loader_validation = DataLoader(dataset, batch_size)
    # dataset = PuzzleDataset(test_data[0], test_data[1], device)
    # loader_test = DataLoader(dataset, batch_size)

    epochs = 100
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
#    plot_util.plot_confusion_matrix(loader_validation, model)
    plot_util.plot_data(train_losses, 'train_loss', dev_losses, 'dev_loss', "loss.png")
    plot_util.plot_data(train_accuracies, 'train accuracy', dev_accuracies, 'dev accuracy', "accuracy.png")

    model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "segment_oder_model.pth")
    torch.save(model, model_path)

if __name__ == '__main__':
    train_model()