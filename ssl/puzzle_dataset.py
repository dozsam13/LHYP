from torch.utils.data import Dataset
import torch
from torchvision import transforms


class PuzzleDataset(Dataset):
    default_augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    def __init__(self, images, device, augmenter=None):
        self.images = images
        self.device = device
        self.augmenter = augmenter

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_ = self.images[index]
        if self.augmenter is not None:
            image_, target_ = self.augmenter(image_)
        else:
            target_ = [i for i in range(4)]
        image_ = transforms.ToTensor()(image_)
        sample = {
            'image': torch.tensor(image_, dtype=torch.float, device=self.device),
            'target': torch.tensor(target_, dtype=torch.float, device=self.device)
        }
        return sample
