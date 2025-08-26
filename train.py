# train.py
"""
Training script for RetinaNet on RSOD dataset
"""

import os
import time
import torch
from torch.utils.data import DataLoader

from dataset import RSODDataset, collate_fn
from transforms import get_train_transform, get_val_transform
from model import get_model, save_model
from visualization import plot_training_curves

def train_one_epoch(model, optimizer, loader, device):
    """
    Train for one epoch
    
    Args:
        model: RetinaNet model
        optimizer: Optimizer
        loader: Training data loader
        device: Device to run on
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    for imgs, targets in loader:
        # Move data to device
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    """
    Evaluate model on validation set
    
    Args:
        model: RetinaNet model
        loader: Validation data loader
        device: Device to run on
    
    Returns:
        Average validation loss
    """
    model.train()  # Keep in train mode for loss computation
    total_loss = 0
    
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            
    return total_loss / len(loader)

def main():
    # Configuration
    train_img_dir = "/path/to/train/images"
    train_ann_dir = "/path/to/train/annotations"
    val_img_dir = "/path/to/val/images"
    val_ann_dir = "/path/to/val/annotations"
    
    batch_size = 4
    num_epochs = 20
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 0.0005
    
    # Check paths
    assert os.path.exists(train_img_dir) and os.path.exists(val_img_dir), "Dataset paths not found!"
    print("Dataset paths verified")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Datasets and dataloaders
    train_dataset = RSODDataset(train_img_dir, train_ann_dir, get_train_transform())
    val_dataset = RSODDataset(val_img_dir, val_ann_dir, get_val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, 
                           num_workers=2, collate_fn=collate_fn)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Model
    model = get_model(num_classes=5)  # 4 classes + background
    model.to(device)
    
    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        t0 = time.time()
        
        # Train and validate
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = evaluate(model, val_loader, device)
        
        # Step scheduler
        scheduler.step()
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
    
    # Save model
    save_model(model, "retinanet_rsod.pth")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    print("Training completed!")

if __name__ == "__main__":
    main()