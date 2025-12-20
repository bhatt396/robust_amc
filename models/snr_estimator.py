"""SNR estimation network"""
import torch
import torch.nn as nn

class SNREstimator(nn.Module):
    """CNN-based SNR estimator"""
    
    def __init__(self, input_shape=(2, 128)):
        super(SNREstimator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(2, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        
        # Calculate size after conv layers
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_shape[0], input_shape[1])
            dummy_output = self.conv_layers(dummy_input)
            conv_output_size = dummy_output.view(1, -1).shape[1]
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output normalized to [0, 1]
        )
    
    def forward(self, x):
        # Reshape from (batch, 2, 128) to (batch, 1, 2, 128)
        x = x.unsqueeze(1)
        conv_out = self.conv_layers(x)
        conv_flat = conv_out.view(conv_out.size(0), -1)
        snr_est = self.fc_layers(conv_flat)
        return snr_est