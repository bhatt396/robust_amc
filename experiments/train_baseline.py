"""Training script for baseline CNN model"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('..')

from models.expert_networks import ExpertNetwork
from utils.data_utils import load_rml_dataset, create_dataloaders
from utils.config import config

def train_baseline_model(data_path=None, num_epochs=50, batch_size=64):
    """Train a baseline CNN model for comparison"""
    
    device = config.DEVICE
    if data_path:
        config.DATA_PATH = data_path
    
    # Load data
    X, y, snr, mods = load_rml_dataset(config.DATA_PATH)
    train_loader, val_loader, test_loader = create_dataloaders(
        X, y, snr, batch_size=batch_size
    )
    
    # Create baseline model (high_snr expert)
    model = ExpertNetwork(
        num_classes=config.NUM_CLASSES,
        expert_type='high_snr'
    ).to(device)
    
    print(f"Baseline model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        for data, targets, _ in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets, _ in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_train_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(config.SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), f"{config.SAVE_DIR}/best_baseline_model.pth")
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Accuracy: {val_acc:.2f}%")
        print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
        print("-" * 50)
    
    # Save final model
    torch.save(model.state_dict(), f"{config.SAVE_DIR}/baseline_model.pth")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Baseline Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Baseline Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(f"{config.RESULTS_DIR}/baseline_training_history.png")
    plt.show()
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return model

if __name__ == "__main__":
    train_baseline_model()