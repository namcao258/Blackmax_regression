# dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from config import MAX_X, MAX_Z, MAX_ANGLE, MAX_HEIGHT

class ImageRegressionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for single channel
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Open image in grayscale mode ('L')

        if self.transform:
            image = self.transform(image)
        
        x = self.data.iloc[idx, 1] / MAX_X
        z = self.data.iloc[idx, 2] / MAX_Z
        angle = self.data.iloc[idx, 3] / MAX_ANGLE
        height = self.data.iloc[idx, 4] / MAX_HEIGHT
        labels = torch.tensor([x, z, angle, height], dtype=torch.float32)
    
        
        return image, labels
