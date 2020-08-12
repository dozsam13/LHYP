import torch
import torch.nn as nn
from data_reader import DataReader
from torch.utils.data import DataLoader
from hypertrophy_dataset import HypertrophyDataset
from hypertrophy_classifier import HypertrophyClassifier
import torch.optim as optim
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
import itertools
from torchvision import transforms


def calc_accuracy(loader, do_log):
    counter = 0
    correctly_labeled = 0
    topk_counter = 0
    for sample in loader:
        counter += 1
        image = sample['image']
        target = sample['target']
        res = model(image)
        predicted = torch.argmax(res)
        if do_log:
            print("------------------------------------------------------")
            print(target)
            print(torch.topk(res, 3)[1])
            print(res)
        top2 = torch.topk(res, 2)[1]
        if target == predicted:
            correctly_labeled += 1
            topk_counter += 1
        elif target in top2.tolist()[0]:
            topk_counter += 1

    # print("Accuracy: {}".format(correctly_labeled/ counter))
    # print("TopK: {}".format(topk_counter/ counter))
    # print(topk_counter, counter)
    return correctly_labeled / counter


def create_data_for_confusion_mx(loader):
    counters = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    counter = 0
    for sample in loader:
        counter += 1
        image = sample['image']
        target = sample['target']
        res = model(image)
        predicted = torch.argmax(res)
        counters[target][predicted] += 1
    return counters, counter


def plot_confusion_matrix(loader):
    cmap = plt.cm.Blues

    cm = np.array(create_data_for_confusion_mx(loader)[0]).astype('float')
    classes = ["Normal", "HCM", "Other"]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion.png")


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


def calculate_loss(loader):
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


batch_size = 30
device = torch.device("cuda")
model = HypertrophyClassifier()
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
# model = nn.Sequential(
#    model,
#    nn.ReLU(),
#    nn.Linear(1000, len(DataReader.possible_pathologies) + 1)
# )

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay=0.7)

in_dir = sys.argv[1]
data_reader = DataReader(in_dir)

(train_data, validation_data, test_data) = split_data(0.66, 0.83, data_reader.x, data_reader.y)

augmenter = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine([-90, 90]),
            transforms.ToTensor()
        ])
dataset = HypertrophyDataset(train_data[0], train_data[1], device, augmenter)
loader_train = DataLoader(dataset, batch_size)
loader_train_accuracy = DataLoader(dataset, 1)
dataset = HypertrophyDataset(validation_data[0], validation_data[1], device)
loader_validation = DataLoader(dataset, 1)
dataset = HypertrophyDataset(test_data[0], test_data[1], device)
loader_test = DataLoader(dataset, 1)

epochs = 100
train_losses = []
validation_losses = []
train_accuracies = []
dev_accuracies = []
scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
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
    # scheduler.step()
    trainloss_for_epoch /= counter
    validationloss_for_epoch = calculate_loss(loader_validation)
    train_losses.append(trainloss_for_epoch)
    validation_losses.append(validationloss_for_epoch)
    train_accuracies.append(calc_accuracy(loader_train_accuracy, False))
    dev_accuracies.append(calc_accuracy(loader_validation, False))
    if epoch % 10 == 0:
        print("Epoch {} has finished (train loss: {}, validation loss: {}".format(epoch, trainloss_for_epoch,
                                                                            validationloss_for_epoch))

print("Test accuracy: ", calc_accuracy(loader_test))
plt.clf()
print("Training has finished.")
plt.plot(train_losses, label='train_loss')
plt.plot(validation_losses, label='validation_loss')
plt.legend()
plt.savefig("train_dev_loss.png")
plt.clf()
plt.plot(dev_accuracies, label='dev accuracy')
plt.plot(train_accuracies, label='train accuracy')
plt.legend()
plt.savefig("accuracy.png")
plt.clf()
plot_confusion_matrix(loader_test)
