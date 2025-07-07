import torch
import torch.nn as nn
import torch.nn.functional as F

class EnvSoundCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EnvSoundCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(24, 48, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Compute flattened size dynamically
        dummy_input = torch.zeros(1, 1, 128, 348)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        self.flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x