"""Expert networks for different SNR ranges"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertNetwork(nn.Module):
    """Base expert network with different architectures for SNR ranges"""
    
    def __init__(self, num_classes=11, expert_type='high_snr', input_shape=(2, 128)):
        super(ExpertNetwork, self).__init__()
        
        self.expert_type = expert_type
        self.input_shape = input_shape
        
        if expert_type == 'high_snr':
            self.features = self._create_high_snr_features()
            fc_input = 256 * 16
        elif expert_type == 'mid_snr':
            self.features = self._create_mid_snr_features()
            fc_input = 128 * 16
        else:  # low_snr
            self.features = self._create_low_snr_features()
            fc_input = 64 * 16
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fc_input, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def _create_high_snr_features(self):
        """Complex model for high SNR signals"""
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(2, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.3),
            
            nn.Conv2d(128, 256, kernel_size=(1, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(1, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 16)),
        )
    
    def _create_mid_snr_features(self):
        """Moderate complexity for mid SNR"""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 16)),
        )
    
    def _create_low_snr_features(self):
        """Simpler, more robust model for low SNR"""
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(2, 5), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            
            nn.Conv2d(16, 32, kernel_size=(1, 5), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(1, 5), padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 16)),
        )
    
    def forward(self, x):
        # Reshape from (batch, 2, 128) to (batch, 1, 2, 128)
        x = x.unsqueeze(1)
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        output = self.classifier(features_flat)
        return output