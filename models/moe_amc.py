import torch
import torch.nn as nn
import torch.nn.functional as F
from models.snr_estimator import SNREstimator
from models.expert_networks import ExpertCNN
from models.gating_network import GatingNetwork

class MoEAMC(nn.Module):
    def __init__(self, num_experts=3, num_classes=8, input_channels=2, 
                 expert_filters=[64, 128, 256], gating_mode='soft'):
        """
        Mixture of Experts Automatic Modulation Classification
        
        Args:
            num_experts: Number of expert networks
            num_classes: Number of modulation classes
            input_channels: Number of input channels (2 for I/Q)
            expert_filters: Filter sizes for expert CNNs
            gating_mode: 'soft' for weighted combination, 'hard' for single expert selection
        """
        super(MoEAMC, self).__init__()
        
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.gating_mode = gating_mode
        
        # SNR Estimator
        self.snr_estimator = SNREstimator(
            input_channels=input_channels,
            hidden_dims=[128, 64],
            output_dim=num_experts
        )
        
        # Expert Networks
        self.experts = nn.ModuleList([
            ExpertCNN(
                input_channels=input_channels,
                num_classes=num_classes,
                filters=expert_filters
            ) for _ in range(num_experts)
        ])
        
        # Gating Network
        self.gating = GatingNetwork(
            num_experts=num_experts,
            hidden_dims=[128, 64]
        )
        
    def forward(self, x, return_expert_outputs=False):
        """
        Args:
            x: Input I/Q signal (batch_size, 2, sample_length)
            return_expert_outputs: If True, return individual expert outputs
        Returns:
            Final classification logits (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Estimate SNR bin probabilities
        snr_probs = self.snr_estimator(x)
        
        # Get gating weights
        if self.gating_mode == 'hard':
            # Hard selection: choose expert with highest SNR probability
            selected = torch.argmax(snr_probs, dim=1)
            gating_weights = F.one_hot(selected, num_classes=self.num_experts).float()
        else:
            # Soft gating: weighted combination
            gating_weights = self.gating(snr_probs)
        
        # Get outputs from all experts
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)
            expert_outputs.append(output)
        
        # Stack expert outputs (batch_size, num_experts, num_classes)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Weighted combination of expert outputs
        # gating_weights: (batch_size, num_experts, 1)
        # expert_outputs: (batch_size, num_experts, num_classes)
        gating_weights = gating_weights.unsqueeze(2)
        final_output = torch.sum(expert_outputs * gating_weights, dim=1)
        
        if return_expert_outputs:
            return final_output, expert_outputs, gating_weights.squeeze(2), snr_probs
        
        return final_output
    
    def predict(self, x):
        """
        Get predicted class labels
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


class MoEAMCWithAnalysis(MoEAMC):
    def __init__(self, *args, **kwargs):
        super(MoEAMCWithAnalysis, self).__init__(*args, **kwargs)
        
    def forward_with_analysis(self, x):
        """
        Forward pass with detailed analysis
        Returns predictions, expert contributions, SNR estimates
        """
        batch_size = x.size(0)
        
        # Estimate SNR
        snr_probs = self.snr_estimator(x)
        
        # Get gating weights
        if self.gating_mode == 'hard':
            selected = torch.argmax(snr_probs, dim=1)
            gating_weights = F.one_hot(selected, num_classes=self.num_experts).float()
        else:
            gating_weights = self.gating(snr_probs)
        
        # Expert outputs
        expert_outputs = []
        expert_predictions = []
        for expert in self.experts:
            output = expert(x)
            expert_outputs.append(output)
            expert_predictions.append(torch.argmax(output, dim=1))
        
        expert_outputs = torch.stack(expert_outputs, dim=1)
        expert_predictions = torch.stack(expert_predictions, dim=1)
        
        # Final prediction
        gating_weights_expanded = gating_weights.unsqueeze(2)
        final_output = torch.sum(expert_outputs * gating_weights_expanded, dim=1)
        final_prediction = torch.argmax(final_output, dim=1)
        
        return {
            'final_prediction': final_prediction,
            'final_logits': final_output,
            'expert_predictions': expert_predictions,
            'expert_outputs': expert_outputs,
            'gating_weights': gating_weights,
            'snr_probs': snr_probs
        }