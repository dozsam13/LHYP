from torch.utils.data import Dataset
from torchvision import transforms
import torch

class HypertrophyDataset(Dataset):
    def __init__(self, images, targets, device):
        self.images = images
        self.targets = targets
        self.device = device
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = {
            'image': torch.tensor(self.images[index], dtype=torch.float, device=self.device),
            'target': torch.tensor(self.targets[index], dtype=torch.float, device=self.device)
        }
        return sample