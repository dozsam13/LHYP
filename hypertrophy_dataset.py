from torch.utils.data import Dataset
from torchvision import transforms
import torch


class HypertrophyDataset(Dataset):
    def __init__(self, images, targets, device):
        self.images = images
        self.targets = targets
        self.device = device
        self.augmenter = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine([-90, 90]),
            transforms.ToTensor()
        ])
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_ = self.images[index]
     #   image_ = self.augmenter(image_)
        sample = {
            'image': torch.tensor(image_, dtype=torch.float, device=self.device),
            'target': torch.tensor(self.targets[index], dtype=torch.long, device=self.device)
        }
        return sample