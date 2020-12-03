import torch.nn as nn
import torch
import sys

from torch.utils.data import DataLoader

from data_reader import DataReader
from hypertrophy_dataset import HypertrophyDataset


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

def train():
    batch_size = 8
    dim_z = 10
    in_dir = sys.argv[1]
    data_reader = DataReader(in_dir)
    device = torch.device("cpu")
    dataset = HypertrophyDataset(data_reader.x, data_reader.y, device)
    data_loader = DataLoader(dataset, batch_size)
    discriminator = Discriminator(8, 3, 82).to(device)
    generator = Generator(10).to(device)

    loss = nn.MSELoss()
    optimizer_g = torch.optim.Adam(generator.parameters())
    optimizer_d = torch.optim.Adam(discriminator.parameters())


    noise = torch.FloatTensor(batch_size, dim_z, 1, 1).to(device)
    label_real = torch.FloatTensor(batch_size, 1).fill_(1).to(device)
    label_fake = torch.FloatTensor(batch_size, 1).fill_(0).to(device)
    criterion = nn.BCELoss()
    for index, sample in enumerate(data_loader):
        discriminator.zero_grad()

        label_real.data.resize(batch_size, 1).fill_(1)
        label_fake.data.resize(batch_size, 1).fill_(0)
        noise.data.resize_(batch_size, dim_z, 1, 1).normal_(0, 1)

        attr = sample['target']
        real = sample['image']
        p_real = discriminator(real, attr)

        fake = generator(noise, attr)
        p_fake = discriminator(fake.detach(), attr)

        d_loss = criterion(p_real, label_real) + criterion(p_fake, label_fake)
        d_loss.backward()
        optimizer_d.step()

        # train generator
        generator.zero_grad()
        p_fake = discriminator(fake, attr)
        g_loss = criterion(p_fake, label_real)
        g_loss.backward()
        optimizer_g.step()


train()