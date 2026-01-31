import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config.config import Config
from data.dataset import IQDataset, load_rml_data
from data.generator import SignalGenerator
from models.moe_amc import MoEAMC, MoEAMCWithAnalysis
from models.expert_networks import ExpertCNN


def load_test_data(config):
    """Load and split data consistently with train.py"""
    np.random.seed(42)  # Same seed as train.py
    
    if config.DATA_SOURCE == 'rml':
        signals, labels, snr_vals, mods = load_rml_data(config.RML_FILE)
    else:
        generator = SignalGenerator(samples_per_symbol=config.SAMPLES_PER_SYMBOL, num_symbols=config.NUM_SYMBOLS)
        signals, labels, snr_vals = generator.generate_dataset(config.MODULATIONS, config.SNR_RANGE)
        mods = config.MODULATIONS # Assuming config.MODULATIONS is available for generated data
    
    n = len(signals)
    indices = np.random.permutation(n)
    train_end = int(config.TRAIN_SPLIT * n)
    val_end = train_end + int(config.VAL_SPLIT * n)
    
    test_indices = indices[val_end:]
    
    test_dataset = IQDataset(signals[test_indices], labels[test_indices], snr_vals[test_indices])
    return test_dataset, mods

def evaluate_model(model, loader, config):
    model.eval()
    all_preds = []
    all_labels = []
    all_snrs = []
    
    with torch.no_grad():
        for signals, labels, snrs in tqdm(loader, desc="Testing"):
            signals = signals.to(config.DEVICE)
            outputs = model(signals)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_snrs.extend(snrs.numpy())
            
    return np.array(all_preds), np.array(all_labels), np.array(all_snrs)

def plot_performance(preds, labels, snrs, config, mods):
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    # 1. Accuracy vs SNR
    unique_snrs = sorted(np.unique(snrs))
    snr_accs = []
    for snr in unique_snrs:
        mask = snrs == snr
        acc = 100 * np.mean(preds[mask] == labels[mask])
        snr_accs.append(acc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(unique_snrs, snr_accs, 'b-o', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Overall Accuracy vs SNR')
    plt.savefig(f"{config.RESULTS_PATH}/accuracy_vs_snr.png")
    plt.close()
    
    # 2. Confusion Matrix at specific SNRs
    for target_snr in [0, 10, 18]:
        if target_snr in unique_snrs:
            mask = snrs == target_snr
            cm = confusion_matrix(labels[mask], preds[mask], labels=range(len(mods)))
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=mods, yticklabels=mods)
            plt.title(f'Confusion Matrix at SNR = {target_snr}dB')
            plt.savefig(f"{config.RESULTS_PATH}/cm_snr_{target_snr}.png")
            plt.close()

    # 3. Overall confusion matrix
    cm = confusion_matrix(labels, preds, labels=range(len(mods)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm/cm.sum(axis=1)[:, None], annot=True, fmt='.2f', cmap='Blues',
                xticklabels=mods, yticklabels=mods)
    plt.title('Normalized Overall Confusion Matrix')
    plt.savefig(f"{config.RESULTS_PATH}/cm_overall.png")
    plt.close()

if __name__ == "__main__":
    config = Config()
    
    test_dataset, mods = load_test_data(config)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=len(mods),
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS
    ).to(config.DEVICE)
    
    model_path = f"{config.MODEL_PATH}/moe_amc_best.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}. Training might be required.")
        exit(1)
        
    preds, labels, snrs = evaluate_model(model, test_loader, config)
    
    avg_acc = 100 * np.mean(preds == labels)
    print(f"\nFinal Test Accuracy: {avg_acc:.2f}%")
    
    plot_performance(preds, labels, snrs, config, mods)
    print(f"\nResults saved to {config.RESULTS_PATH}")