import torch
import torch.nn as nn

class ResBlock3D(nn.Module):
    """A simple 3D residual block."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class LungNoduleNet3D(nn.Module):
    """A simplified 3D CNN for lung nodule detection."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2))
        self.layer2 = ResBlock3D(16, 32, stride=2)
        self.layer3 = ResBlock3D(32, 64, stride=2)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)