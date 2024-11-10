# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionCNN(nn.Module):
    def __init__(self):
        super(RegressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adjusted fully connected layer input size to match 256x256 image size
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Flattened size after conv3 and pooling
        self.fc2 = nn.Linear(128, 4)  # Output layer for x, z, and angle

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 32 * 32)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
