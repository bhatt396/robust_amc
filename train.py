import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import json

from config.config import Config
from data.dataset import IQDataset, SNRStratifiedDataset, load_rml_data
from data.generator import SignalGenerator
from models.moe_amc import MoEAMC
from models.expert_networks import ExpertCNN
from models.snr_estimator import SNREstimator
from utils.metrics import calculate_accuracy, calculate_confusion_matrix
from utils.visualization import plot_training_curves, plot_confusion_matrix

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False

    def __call__(self, val_acc, model):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"Validation accuracy increased. Saving model to {self.path}")
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def train_snr_estimator(train_loader, val_loader, config):
    """Train SNR estimator separately"""
    print("\n>>> Stage 1: Training SNR Estimator")
    
    model = SNREstimator(
        input_channels=2,
        hidden_dims=config.SNR_ESTIMATOR_HIDDEN,
        output_dim=config.NUM_EXPERTS
    ).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                          weight_decay=config.WEIGHT_DECAY)
    
    os.makedirs(config.MODEL_PATH, exist_ok=True)
    early_stopping = EarlyStopping(patience=15, path=f"{config.MODEL_PATH}/snr_estimator_best.pth")
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        for signals, labels, snrs in pbar:
            signals = signals.to(config.DEVICE)
            # Vectorized SNR binning
            snr_bins = torch.tensor([config.get_snr_bin(s.item()) for s in snrs]).to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, snr_bins)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation
        val_acc = validate_snr(model, val_loader, config)
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Val SNR Acc: {val_acc:.2f}%")
        
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    return model, history

def validate_snr(model, loader, config):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for signals, _, snrs in loader:
            signals = signals.to(config.DEVICE)
            snr_bins = torch.tensor([config.get_snr_bin(s.item()) for s in snrs]).to(config.DEVICE)
            outputs = model(signals)
            _, predicted = outputs.max(1)
            total += snr_bins.size(0)
            correct += predicted.eq(snr_bins).sum().item()
    return 100. * correct / total

def train_expert(expert_id, train_loader, val_loader, config):
    """Train individual expert for specific SNR range"""
    print(f"\n>>> Stage 2: Training Expert {expert_id} ({config.SNR_BINS[expert_id]} SNR)")
    
    if len(train_loader.dataset) == 0:
        print(f"Warning: No data for expert {expert_id}. Skipping.")
        return None, None

    model = ExpertCNN(
        input_channels=2,
        num_classes=config.NUM_CLASSES,
        filters=config.EXPERT_CNN_FILTERS
    ).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                          weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    early_stopping = EarlyStopping(patience=15, path=f"{config.MODEL_PATH}/expert_{expert_id}_best.pth")
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for signals, labels, _ in pbar:
            signals = signals.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation
        val_acc = validate_expert(model, val_loader, config)
        scheduler.step(val_acc)
        
        print(f"Expert {expert_id} Epoch {epoch+1}: Val Acc: {val_acc:.2f}%")
        
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            break
    
    return model

def validate_expert(model, loader, config):
    model.eval()
    correct = 0
    total = 0
    if len(loader.dataset) == 0: return 0.0
    with torch.no_grad():
        for signals, labels, _ in loader:
            signals = signals.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(signals)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def train_moe_system(train_loader, val_loader, config):
    """Train complete MoE system end-to-end"""
    print("\n>>> Stage 3: Fine-tuning Complete MoE AMC System")
    
    model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=config.NUM_CLASSES,
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS,
        gating_mode='soft'
    ).to(config.DEVICE)
    
    # Load pre-trained components
    try:
        model.snr_estimator.load_state_dict(torch.load(f"{config.MODEL_PATH}/snr_estimator_best.pth"))
        print("Loaded SNR estimator weights")
        for i in range(config.NUM_EXPERTS):
            expert_path = f"{config.MODEL_PATH}/expert_{i}_best.pth"
            if os.path.exists(expert_path):
                model.experts[i].load_state_dict(torch.load(expert_path))
                print(f"Loaded expert {i} weights")
    except Exception as e:
        print(f"Warning: Could not load all pre-trained weights: {e}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE * 0.1,
                          weight_decay=config.WEIGHT_DECAY)
    
    early_stopping = EarlyStopping(patience=15, path=f"{config.MODEL_PATH}/moe_amc_best.pth")
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for signals, labels, _ in pbar:
            signals = signals.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation
        val_acc = validate_expert(model, val_loader, config) # same logic for accuracy
        print(f"MoE System Epoch {epoch+1}: Val Acc: {val_acc:.2f}%")
        
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            break
            
    return model

if __name__ == "__main__":
    config = Config()
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Data Loading
    if config.DATA_SOURCE == 'rml':
        signals, labels, snr_vals, mods = load_rml_data(config.RML_FILE)
    else:
        generator = SignalGenerator(samples_per_symbol=config.SAMPLES_PER_SYMBOL, num_symbols=config.NUM_SYMBOLS)
        signals, labels, snr_vals = generator.generate_dataset(config.MODULATIONS, config.SNR_RANGE)
    
    # Improved data split
    n = len(signals)
    indices = np.random.permutation(n)
    train_end = int(config.TRAIN_SPLIT * n)
    val_end = train_end + int(config.VAL_SPLIT * n)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_dataset = IQDataset(signals[train_indices], labels[train_indices], snr_vals[train_indices])
    val_dataset = IQDataset(signals[val_indices], labels[val_indices], snr_vals[val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Pipeline
    train_snr_estimator(train_loader, val_loader, config)
    
    snr_ranges = [config.SNR_LOW, config.SNR_MID, config.SNR_HIGH]
    for i, snr_range in enumerate(snr_ranges):
        expert_train = SNRStratifiedDataset(signals[train_indices], labels[train_indices], snr_vals[train_indices], snr_range)
        expert_val = SNRStratifiedDataset(signals[val_indices], labels[val_indices], snr_vals[val_indices], snr_range)
        
        e_train_loader = DataLoader(expert_train, batch_size=config.BATCH_SIZE, shuffle=True)
        e_val_loader = DataLoader(expert_val, batch_size=config.BATCH_SIZE, shuffle=False)
        
        train_expert(i, e_train_loader, e_val_loader, config)
    
    train_moe_system(train_loader, val_loader, config)
    
    print("\nTraining complete! Check results in 'checkpoints' directory.")