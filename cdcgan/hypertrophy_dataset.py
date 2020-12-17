from torch.utils.data import Dataset
import torch
from torchvision import transforms


class HypertrophyDataset(Dataset):
    default_augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    def __init__(self, images, targets, device, augmenter=default_augmenter):
        self.images = images
        self.targets = targets
        self.device = device
        self.augmenter = augmenter

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_ = self.images[index]
        image_ = self.augmenter(image_)
        sample = {
            'image': torch.tensor(image_, dtype=torch.float, device=self.device),
            'target': torch.tensor(self.targets[index], dtype=torch.float, device=self.device)
        }
        return sample
