from torch.utils.data import Dataset
import torch
from torchvision import transforms


class HypertrophyDataset(Dataset):
    default_augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    def __init__(self, sequences, targets, device, augmenter=default_augmenter):
        self.sequences = sequences
        self.targets = targets
        self.device = device
        self.augmenter = augmenter

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence_ = self.sequences[index]
        #sequence_ = self.augmenter(sequence_)
        sequence_ = list(map(lambda e: (torch.tensor(e, dtype=torch.float, device=self.device)), sequence_))

        sequence_ = torch.cat(sequence_).view(len(sequence_), 1, 110, 110)
        sample = {
            'sequence': torch.tensor(sequence_, dtype=torch.float, device=self.device),
            'target': torch.tensor(self.targets[index], dtype=torch.long, device=self.device)
        }
        return sample
