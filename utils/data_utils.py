"""Data loading and preprocessing utilities"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .config import config

class RMLDataset(Dataset):
    """Dataset class for RML2016.10a"""
    
    def __init__(self, X, y, snr):
        """
        Args:
            X: Signal data
            y: Modulation labels
            snr: SNR values
        """
        self.X = X
        self.y = y
        self.snr = snr
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.snr[idx], dtype=torch.float32)
        )

def load_rml_dataset(data_path):
    """Load and preprocess RML2016.10a dataset"""
    print(f"Loading dataset from {data_path}")
    
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
    
    mods = list(data_dict.keys())
    mod_to_idx = {mod: idx for idx, mod in enumerate(mods)}
    
    X_list = []
    y_list = []
    snr_list = []
    
    for mod in mods:
        for snr in data_dict[mod].keys():
            signals = data_dict[mod][snr]
            for signal in signals:
                # Convert complex to real (I and Q channels)
                iq_data = np.stack([signal.real, signal.imag], axis=0)
                X_list.append(iq_data)
                y_list.append(mod_to_idx[mod])
                snr_list.append(snr)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.long)
    snr = np.array(snr_list, dtype=np.float32)
    
    # Normalize SNR to [0, 1] range
    snr_normalized = (snr - (-20)) / (20 - (-20))
    
    print(f"Dataset loaded: {len(X)} samples, {len(mods)} modulation types")
    print(f"SNR range: {snr.min()} to {snr.max()} dB")
    
    return X, y, snr_normalized, mods

def create_dataloaders(X, y, snr, batch_size=64):
    """Create train, validation, and test dataloaders"""
    
    # Split data
    X_temp, X_test, y_temp, y_test, snr_temp, snr_test = train_test_split(
        X, y, snr, test_size=config.TEST_RATIO, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
        X_temp, y_temp, snr_temp, 
        test_size=config.VAL_RATIO/(config.TRAIN_RATIO + config.VAL_RATIO),
        random_state=42, stratify=y_temp
    )
    
    # Create datasets
    train_dataset = RMLDataset(X_train, y_train, snr_train)
    val_dataset = RMLDataset(X_val, y_val, snr_val)
    test_dataset = RMLDataset(X_test, y_test, snr_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader