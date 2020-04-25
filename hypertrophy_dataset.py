from torch.utils.data import Dataset
import torch

class HypertrophyDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = {
            'image_id': index,
            'image': self.images[index],
            'target': self.targets[index]
        }
        return sample