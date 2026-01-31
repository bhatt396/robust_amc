import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertCNN(nn.Module):
    def __init__(self, input_channels=2, num_classes=8, filters=[64, 128, 256]):
        """
        Expert CNN classifier for specific SNR range
        
        Args:
            input_channels: Number of input channels (2 for I/Q)
            num_classes: Number of modulation classes
            filters: List of filter sizes for conv layers
        """
        super(ExpertCNN, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv1d(input_channels, filters[0], kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(filters[0])
        self.pool1 = nn.MaxPool1d(2)
        
        # Second conv block
        self.conv2 = nn.Conv1d(filters[0], filters[1], kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(filters[1])
        self.pool2 = nn.MaxPool1d(2)
        
        # Third conv block
        self.conv3 = nn.Conv1d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(filters[2])
        self.pool3 = nn.MaxPool1d(2)
        
        # Fourth conv block
        self.conv4 = nn.Conv1d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(filters[2])
        self.pool4 = nn.MaxPool1d(2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(filters[2], 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, 2, sample_length)
        Returns:
            Class logits (batch_size, num_classes)
        """
        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Global pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ResNetExpert(nn.Module):
    def __init__(self, input_channels=2, num_classes=8, base_filters=64):
        """
        ResNet-based expert classifier
        """
        super(ResNetExpert, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, base_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.pool1 = nn.MaxPool1d(2)
        
        self.res1 = ResidualBlock(base_filters)
        self.res2 = ResidualBlock(base_filters)
        
        self.conv2 = nn.Conv1d(base_filters, base_filters * 2, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(base_filters * 2)
        
        self.res3 = ResidualBlock(base_filters * 2)
        self.res4 = ResidualBlock(base_filters * 2)
        
        self.conv3 = nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm1d(base_filters * 4)
        
        self.res5 = ResidualBlock(base_filters * 4)
        self.res6 = ResidualBlock(base_filters * 4)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters * 4, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.res1(x)
        x = self.res2(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res3(x)
        x = self.res4(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res5(x)
        x = self.res6(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x