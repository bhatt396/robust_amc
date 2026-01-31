import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_curves(train_losses, val_losses=None, train_accs=None, val_accs=None, save_path=None):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    if val_losses is not None:
        axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if train_accs is not None:
        axes[1].plot(train_accs, label='Train Accuracy', linewidth=2)
    if val_accs is not None:
        axes[1].plot(val_accs, label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, class_names, normalize=False, save_path=None):
    """Plot confusion matrix"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_iq_samples(signals, labels, class_names, num_samples=8, save_path=None):
    """Plot I/Q constellation diagrams"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(signals))):
        signal = signals[i]
        label = labels[i]
        
        axes[i].scatter(signal.real, signal.imag, alpha=0.5, s=1)
        axes[i].set_xlabel('In-phase (I)')
        axes[i].set_ylabel('Quadrature (Q)')
        axes[i].set_title(f'{class_names[label]}')
        axes[i].grid(True, alpha=0.3)
        axes[i].axis('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_signal_spectrum(signal, sample_rate=1.0, save_path=None):
    """Plot signal spectrum"""
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    magnitude = np.abs(fft)
    
    plt.figure(figsize=(12, 6))
    plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(20*np.log10(magnitude + 1e-10)))
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Magnitude (dB)', fontsize=12)
    plt.title('Signal Spectrum', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_expert_contributions(gating_weights, snr_bins, save_path=None):
    """Plot expert gating weights distribution"""
    plt.figure(figsize=(10, 6))
    
    for i in range(gating_weights.shape[1]):
        plt.hist(gating_weights[:, i], bins=50, alpha=0.5, label=f'Expert {i} ({snr_bins[i]})')
    
    plt.xlabel('Gating Weight', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Expert Gating Weights', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_snr_estimation_error(true_snrs, estimated_snrs, save_path=None):
    """Plot SNR estimation error"""
    errors = estimated_snrs - true_snrs
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    axes[0].scatter(true_snrs, estimated_snrs, alpha=0.5, s=10)
    axes[0].plot([true_snrs.min(), true_snrs.max()], 
                 [true_snrs.min(), true_snrs.max()], 
                 'r--', linewidth=2, label='Perfect estimation')
    axes[0].set_xlabel('True SNR (dB)', fontsize=12)
    axes[0].set_ylabel('Estimated SNR (dB)', fontsize=12)
    axes[0].set_title('SNR Estimation', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Error histogram
    axes[1].hist(errors, bins=50, edgecolor='black')
    axes[1].set_xlabel('Estimation Error (dB)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'SNR Estimation Error (Mean: {errors.mean():.2f} dB)', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_per_modulation_snr_accuracy(predictions, labels, snrs, class_names, snr_range, save_path=None):
    """Plot accuracy vs SNR for each modulation"""
    num_classes = len(class_names)
    snr_bins = np.arange(snr_range[0], snr_range[1] + 2, 2)
    
    plt.figure(figsize=(12, 8))
    
    for i in range(num_classes):
        mask = labels == i
        if np.sum(mask) > 0:
            class_snrs = snrs[mask]
            class_preds = predictions[mask]
            class_labels = labels[mask]
            
            accuracies = []
            snr_centers = []
            
            for j in range(len(snr_bins) - 1):
                snr_mask = (class_snrs >= snr_bins[j]) & (class_snrs < snr_bins[j + 1])
                if np.sum(snr_mask) > 0:
                    acc = 100 * np.mean(class_preds[snr_mask] == class_labels[snr_mask])
                    accuracies.append(acc)
                    snr_centers.append((snr_bins[j] + snr_bins[j + 1]) / 2)
            
            plt.plot(snr_centers, accuracies, marker='o', label=class_names[i], linewidth=2)
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Modulation Accuracy vs SNR', fontsize=14)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()