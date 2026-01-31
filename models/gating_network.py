import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    def __init__(self, num_experts=3, hidden_dims=[128, 64]):
        """
        Gating network that produces weights for each expert
        
        Args:
            num_experts: Number of expert networks
            hidden_dims: List of hidden layer dimensions
        """
        super(GatingNetwork, self).__init__()
        
        layers = []
        prev_dim = num_experts  # Input is SNR bin probabilities
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_experts))
        self.network = nn.Sequential(*layers)
        
    def forward(self, snr_probs):
        """
        Args:
            snr_probs: SNR bin probabilities (batch_size, num_experts)
        Returns:
            Expert weights (batch_size, num_experts)
        """
        weights = self.network(snr_probs)
        weights = F.softmax(weights, dim=1)
        return weights


class AdaptiveGatingNetwork(nn.Module):
    def __init__(self, input_channels=2, num_experts=3, hidden_dims=[128, 64]):
        """
        Adaptive gating network that learns from raw I/Q samples
        """
        super(AdaptiveGatingNetwork, self).__init__()
        
        # Feature extractor
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Gating layers
        layers = []
        prev_dim = 64
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_experts))
        self.gating_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Input I/Q signal (batch_size, 2, sample_length)
        Returns:
            Expert weights (batch_size, num_experts)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        weights = self.gating_layers(x)
        weights = F.softmax(weights, dim=1)
        
        return weights


class HardGatingNetwork(nn.Module):
    def __init__(self, num_experts=3):
        """
        Hard gating that selects single expert based on SNR bin
        """
        super(HardGatingNetwork, self).__init__()
        self.num_experts = num_experts
        
    def forward(self, snr_probs):
        """
        Args:
            snr_probs: SNR bin probabilities (batch_size, num_experts)
        Returns:
            One-hot expert selection (batch_size, num_experts)
        """
        selected = torch.argmax(snr_probs, dim=1)
        weights = F.one_hot(selected, num_classes=self.num_experts).float()
        return weights