from torch.utils.data import Dataset
import torch
from torchvision import transforms


class PuzzleDataset(Dataset):
    def __init__(self, images, device, n_split, augmenter=None):
        self.images = images
        self.device = device
        self.augmenter = augmenter
        self.n_split = n_split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_ = self.images[index]
        if self.augmenter is not None:
            image_, target_ = self.augmenter(image_)
        else:
            target_ = [i for i in range(self.n_split*self.n_split)]
        image_ = transforms.ToTensor()(image_)
        sample = {
            'image': torch.tensor(image_, dtype=torch.float, device=self.device),
            'target': torch.tensor(target_, dtype=torch.float, device=self.device)
        }
        return sample
