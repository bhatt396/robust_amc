"""Router/gating network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RouterNetwork(nn.Module):
    """Routes signals to appropriate experts based on SNR"""
    
    def __init__(self, num_experts=3):
        super(RouterNetwork, self).__init__()
        
        self.num_experts = num_experts
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Routing network
        self.router = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )
    
    def forward(self, snr_estimate):
        # snr_estimate shape: (batch, 1)
        router_logits = self.router(snr_estimate)
        
        # Soft gating with temperature
        gating_weights = F.softmax(router_logits / self.temperature, dim=1)
        
        # Optional: Hard routing for inference
        if not self.training:
            hard_weights = torch.zeros_like(gating_weights)
            hard_weights.scatter_(1, torch.argmax(gating_weights, dim=1, keepdim=True), 1.0)
            # Use straight-through estimator
            gating_weights = hard_weights + gating_weights - gating_weights.detach()
        
        return gating_weights