import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os

class TestDataLoader(Dataset):

    def __init__(self, folder_dir, transform=None):
        self.data_dir = folder_dir
        self.labels = torch.load(self.data_dir + 'labels.pth')
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.data_dir))-1

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.data_dir+str(idx)+'.png').float()/255.0
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]

        return img, label