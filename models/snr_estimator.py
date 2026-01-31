import torch
import torch.nn as nn
import torch.nn.functional as F

class SNREstimator(nn.Module):
    def __init__(self, input_channels=2, hidden_dims=[128, 64], output_dim=3):
        """
        SNR Estimator network that predicts SNR bin probabilities
        
        Args:
            input_channels: Number of input channels (2 for I/Q)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of SNR bins (3 for low/mid/high)
        """
        super(SNREstimator, self).__init__()
        
        # Convolutional feature extractor
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        fc_layers = []
        prev_dim = 128
        for hidden_dim in hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        fc_layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, 2, sample_length)
        Returns:
            SNR bin probabilities (batch_size, num_bins)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        
        return x


class SNRRegressorEstimator(nn.Module):
    def __init__(self, input_channels=2, hidden_dims=[128, 64]):
        """
        SNR Estimator that directly regresses SNR value
        """
        super(SNRRegressorEstimator, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        fc_layers = []
        prev_dim = 128
        for hidden_dim in hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        fc_layers.append(nn.Linear(prev_dim, 1))
        self.fc = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        return x