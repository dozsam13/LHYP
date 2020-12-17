import torch.nn as nn
import torch
import sys
import matplotlib.pyplot
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from cdcgan.data_reader import DataReader
from cdcgan.hypertrophy_dataset import HypertrophyDataset
import util.plot_util as plot_util


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + 3, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 3, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, 1, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, attr):
        attr = attr.view(-1, 3, 1, 1)
        x = torch.cat([x, attr], 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, channel_n, target_n, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.feature_input = nn.Linear(target_n, img_size * img_size)
        self.net = nn.Sequential(
            self._block(4, channel_n, 4, 2, 1),
            self._block(channel_n, channel_n * 2, 4, 2, 1),
            self._block(channel_n * 2, channel_n * 4, 4, 2, 1),
            self._block(channel_n * 4, channel_n * 8, 4, 2, 1),
            nn.Conv2d(channel_n * 8, 1, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, attr):
        attr = self.feature_input(attr).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([x, attr], 1)
        s = self.net(x)
        return s.view(-1, 1)

device = torch.device("cuda")

def train():
    batch_size = 8
    dim_z = 3
    in_dir = sys.argv[1]
    data_reader = DataReader(in_dir)

    dataset = HypertrophyDataset(data_reader.x, data_reader.y, device)
    data_loader = DataLoader(dataset, batch_size)
    discriminator = Discriminator(8, 3, 82).to(device)
    generator = Generator(dim_z).to(device)

    optimizer_g = torch.optim.AdamW(generator.parameters())
    optimizer_d = torch.optim.AdamW(discriminator.parameters())
    g_lr_dec = StepLR(optimizer_g, step_size=100, gamma=0.8)
    d_lr_dec = StepLR(optimizer_d, step_size=100, gamma=0.8)

    criterion = nn.BCELoss(reduction='mean')

    epochs = 1000
    g_losses = []
    d_losses = []
    for epoch in range(epochs):
        g_loss_epoch = 0
        d_loss_epoch = 0
        for index, sample in enumerate(data_loader):
            label = sample['target']
            real = sample['image']
            current_batch_size = len(real)
            discriminator.zero_grad()

            noise = torch.FloatTensor(current_batch_size, dim_z, 1, 1).normal_(0, 1).to(device)
            label_real = torch.FloatTensor(current_batch_size, 1).fill_(1).to(device)
            label_fake = torch.FloatTensor(current_batch_size, 1).fill_(0).to(device)

            p_real = discriminator(real, label)

            fake = generator(noise, label)
            p_fake = discriminator(fake.detach(), label)

            d_loss = criterion(p_real, label_real) + criterion(p_fake, label_fake)
            if epoch % 2 == 0:
                d_loss.backward()
                optimizer_d.step()

            # train generator
            generator.zero_grad()
            p_fake = discriminator(fake, label)
            g_loss = criterion(p_fake, label_real)
            g_loss.backward()
            optimizer_g.step()

            g_loss_epoch += g_loss.cpu().detach().numpy()
            d_loss_epoch += d_loss.cpu().detach().numpy()
        g_lr_dec.step()
        d_lr_dec.step()
        g_losses.append(g_loss_epoch)
        d_losses.append(d_loss_epoch)

    plot_util.plot_data(g_losses, 'generator', d_losses, 'discriminator', "loss.png")
    for i in range(10):
        noise = torch.FloatTensor(1, dim_z, 1, 1).normal_(0, 1).to(device)
        fake = generator(noise, torch.tensor([[1, 0, 0]],  dtype=torch.float, device=device)).cpu().detach().numpy()
        matplotlib.pyplot.imsave("generated" + str(i) + ".png", (fake[0][1])*255, cmap='gray')


train()