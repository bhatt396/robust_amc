"""Training script for MoE model"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('..')

from models.moe_model import SNRAwareMoE
from utils.data_utils import load_rml_dataset, create_dataloaders
from utils.config import config

def train_moe_model():
    """Main training function for MoE model"""
    
    # Set device
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Load data
    X, y, snr, mods = load_rml_dataset(config.DATA_PATH)
    train_loader, val_loader, test_loader = create_dataloaders(
        X, y, snr, batch_size=config.BATCH_SIZE
    )
    
    # Create model
    model = SNRAwareMoE(num_classes=config.NUM_CLASSES).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, targets, snr_true) in enumerate(train_loader):
            data, targets, snr_true = data.to(device), targets.to(device), snr_true.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data, snr_true)
            
            # Calculate losses
            class_loss = criterion(outputs['logits'], targets)
            total_loss = class_loss + model.aux_loss_weight * outputs['aux_loss']
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            running_loss += total_loss.item()
            _, predicted = torch.max(outputs['logits'], 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], '
                      f'Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {total_loss.item():.4f}')
        
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data, targets, _ in val_loader:
                data, targets = data.to(device), targets.to(device)
                predictions, _, _ = model.predict(data)
                correct_val += (predictions == targets).sum().item()
                total_val += targets.size(0)
        
        val_acc = 100 * correct_val / total_val
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_train_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(config.SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), f"{config.SAVE_DIR}/best_moe_model.pth")
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}:")
        print(f"  Training Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Validation Accuracy: {val_acc:.2f}%")
        print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
        print("-" * 50)
    
    # Save final model
    torch.save(model.state_dict(), f"{config.SAVE_DIR}/final_moe_model.pth")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(f"{config.RESULTS_DIR}/training_history.png")
    plt.show()
    
    return model, train_losses, val_accuracies

if __name__ == "__main__":
    model, train_losses, val_accuracies = train_moe_model()