import torch
import matplotlib.pyplot as plt
import numpy as np

def create_data_for_confusion_mx(loader, model):
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

def plot_data(data1, label1, data2, label2, filename):
    plt.clf()
    plt.plot(data1, label=label1)
    plt.plot(data2, label=label2)
    plt.legend()
    plt.savefig(filename)
