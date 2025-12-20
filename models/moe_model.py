"""Complete SNR-Aware Mixture of Experts Model"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SNRAwareMoE(nn.Module):
    """Main MoE model integrating all components"""
    
    def __init__(self, num_classes=11):
        super(SNRAwareMoE, self).__init__()
        
        # Components
        from .snr_estimator import SNREstimator
        from .router_network import RouterNetwork
        from .expert_networks import ExpertNetwork
        
        self.snr_estimator = SNREstimator()
        self.router = RouterNetwork(num_experts=3)
        
        # Three experts for different SNR ranges
        self.experts = nn.ModuleList([
            ExpertNetwork(num_classes, expert_type='low_snr'),
            ExpertNetwork(num_classes, expert_type='mid_snr'),
            ExpertNetwork(num_classes, expert_type='high_snr'),
        ])
        
        # Auxiliary loss weight
        self.aux_loss_weight = 0.1
    
    def forward(self, x, snr_true=None):
        # 1. Estimate SNR
        snr_est = self.snr_estimator(x)
        
        # 2. Get routing weights
        gating_weights = self.router(snr_est)
        
        # 3. Get predictions from all experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        # Stack outputs: (batch, 3, num_classes)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # 4. Weighted combination
        gating_weights_expanded = gating_weights.unsqueeze(-1)
        final_output = torch.sum(expert_outputs * gating_weights_expanded, dim=1)
        
        # 5. Compute auxiliary loss
        aux_loss = 0
        if snr_true is not None and self.training:
            aux_pred = snr_est.squeeze()
            aux_loss = F.mse_loss(aux_pred, snr_true)
        
        outputs = {
            'logits': final_output,
            'expert_outputs': expert_outputs,
            'gating_weights': gating_weights,
            'snr_estimate': snr_est,
            'aux_loss': aux_loss
        }
        
        return outputs
    
    def predict(self, x):
        """Inference mode prediction"""
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs['logits'], dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            expert_usage = torch.argmax(outputs['gating_weights'], dim=1)
            
        return predictions, probabilities, expert_usage