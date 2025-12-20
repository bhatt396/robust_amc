"""Evaluation script for comparing models"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
sys.path.append('..')

from models.moe_model import SNRAwareMoE
from models.expert_networks import ExpertNetwork
from utils.data_utils import load_rml_dataset, create_dataloaders
from utils.config import config

def evaluate_snr_wise(model, test_loader, device):
    """Evaluate model performance across different SNR ranges"""
    
    model.eval()
    
    # Define SNR ranges for evaluation
    snr_bins = [
        (-20, -10),
        (-10, 0),
        (0, 10),
        (10, 20)
    ]
    
    results = {f"snr_{i}": {'correct': 0, 'total': 0} for i in range(len(snr_bins))}
    expert_usage = {0: 0, 1: 0, 2: 0}
    
    with torch.no_grad():
        for data, targets, snr_norm in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Denormalize SNR
            snr_original = snr_norm.numpy() * 40 - 20
            
            # Get predictions
            if isinstance(model, SNRAwareMoE):
                predictions, _, expert_pred = model.predict(data)
                for exp in expert_pred.cpu().numpy():
                    expert_usage[exp] += 1
            else:
                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)
            
            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()
            
            # Calculate accuracy for each SNR bin
            for i, (snr_min, snr_max) in enumerate(snr_bins):
                mask = (snr_original >= snr_min) & (snr_original < snr_max)
                if np.any(mask):
                    correct = np.sum(predictions[mask] == targets[mask])
                    total = np.sum(mask)
                    results[f"snr_{i}"]['correct'] += correct
                    results[f"snr_{i}"]['total'] += total
    
    # Calculate final accuracies
    final_results = {}
    for snr_key in results:
        if results[snr_key]['total'] > 0:
            accuracy = 100 * results[snr_key]['correct'] / results[snr_key]['total']
            final_results[snr_key] = accuracy
    
    # Calculate expert usage percentages
    total_predictions = sum(expert_usage.values())
    if total_predictions > 0:
        expert_usage_percent = {
            f'Expert_{i}': 100 * count / total_predictions 
            for i, count in expert_usage.items()
        }
    else:
        expert_usage_percent = {}
    
    return final_results, expert_usage_percent

def main():
    """Main evaluation function"""
    
    device = config.DEVICE
    
    # Load data
    X, y, snr, mods = load_rml_dataset(config.DATA_PATH)
    _, _, test_loader = create_dataloaders(X, y, snr)
    
    # Load models
    moe_model = SNRAwareMoE(num_classes=config.NUM_CLASSES).to(device)
    moe_model.load_state_dict(torch.load(f"{config.SAVE_DIR}/best_moe_model.pth", 
                                        map_location=device))
    
    baseline_model = ExpertNetwork(num_classes=config.NUM_CLASSES, 
                                  expert_type='high_snr').to(device)
    baseline_model.load_state_dict(torch.load(f"{config.SAVE_DIR}/baseline_model.pth", 
                                             map_location=device))
    
    # Evaluate models
    print("Evaluating MoE model...")
    moe_results, expert_usage = evaluate_snr_wise(moe_model, test_loader, device)
    
    print("Evaluating baseline model...")
    baseline_results, _ = evaluate_snr_wise(baseline_model, test_loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    snr_labels = ["Very Low (-20 to -10 dB)", "Low (-10 to 0 dB)", 
                  "Medium (0 to 10 dB)", "High (10 to 20 dB)"]
    
    print(f"\n{'SNR Range':<25} {'MoE Model':<12} {'Baseline':<12} {'Improvement':<12}")
    print("-"*60)
    
    for i, label in enumerate(snr_labels):
        moe_key = f"snr_{i}"
        baseline_key = f"snr_{i}"
        
        if moe_key in moe_results and baseline_key in baseline_results:
            moe_acc = moe_results[moe_key]
            baseline_acc = baseline_results[baseline_key]
            improvement = moe_acc - baseline_acc
            
            print(f"{label:<25} {moe_acc:<11.2f}% {baseline_acc:<11.2f}% {improvement:<11.2f}%")
    
    print(f"\nExpert Usage Statistics:")
    for expert, usage in expert_usage.items():
        print(f"  {expert}: {usage:.1f}%")
    
    # Save results to JSON
    results_dict = {
        'moe_results': moe_results,
        'baseline_results': baseline_results,
        'expert_usage': expert_usage
    }
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(f"{config.RESULTS_DIR}/evaluation_results.json", 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    # Create visualization
    plot_comparison(moe_results, baseline_results, expert_usage, snr_labels)
    
    return moe_results, baseline_results, expert_usage

def plot_comparison(moe_results, baseline_results, expert_usage, snr_labels):
    """Create comparison plots"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0]
    x = np.arange(len(snr_labels))
    width = 0.35
    
    moe_acc = [moe_results.get(f'snr_{i}', 0) for i in range(len(snr_labels))]
    baseline_acc = [baseline_results.get(f'snr_{i}', 0) for i in range(len(snr_labels))]
    
    ax1.bar(x - width/2, moe_acc, width, label='MoE Model', color='skyblue')
    ax1.bar(x + width/2, baseline_acc, width, label='Baseline', color='lightcoral')
    
    ax1.set_xlabel('SNR Range')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Comparison Across SNR Ranges')
    ax1.set_xticks(x)
    ax1.set_xticklabels(snr_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(moe_acc):
        ax1.text(i - width/2, v + 1, f'{v:.1f}', ha='center', fontsize=9)
    for i, v in enumerate(baseline_acc):
        ax1.text(i + width/2, v + 1, f'{v:.1f}', ha='center', fontsize=9)
    
    # Plot 2: Expert usage
    ax2 = axes[1]
    experts = list(expert_usage.keys())
    usage = list(expert_usage.values())
    
    colors = ['lightgreen', 'gold', 'lightcoral']
    bars = ax2.bar(experts, usage, color=colors)
    ax2.set_xlabel('Expert Network')
    ax2.set_ylabel('Usage Percentage (%)')
    ax2.set_title('Expert Network Usage Distribution')
    ax2.set_xticklabels(['Low SNR\nExpert', 'Mid SNR\nExpert', 'High SNR\nExpert'])
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, v in zip(bars, usage):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(f"{config.RESULTS_DIR}/comparison_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()