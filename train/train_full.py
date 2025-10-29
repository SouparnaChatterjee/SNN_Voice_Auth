# train/train_full.py
"""
Full training script for SNN models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from configs.config import cfg
from models.dual_task_snn import DualTaskSNN
from models.losses import CombinedLoss
from utils.data_loader import get_keyword_dataloaders

class SimpleTrainer:
    """Simplified trainer for Week 3"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cpu')
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.get('lr', 0.001)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch['spikes'], task='kws')
            logits = outputs['kws']['logits']
            labels = batch['label'].squeeze()
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(batch['spikes'], task='kws')
                logits = outputs['kws']['logits']
                labels = batch['label'].squeeze()
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    def train(self, train_loader, val_loader, epochs):
        """Main training loop"""
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        best_acc = 0
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_acc = self.validate(val_loader)
            
            # Log
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Acc: {val_acc:.2f}%")
            
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(epoch, val_acc)
        
        print(f"\n✅ Training complete! Best accuracy: {best_acc:.2f}%")
    
    def save_checkpoint(self, epoch, accuracy):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config
        }
        
        # Create directory
        save_dir = Path("experiments") / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save
        save_path = save_dir / "best_model.pt"
        torch.save(checkpoint, save_path)
        print(f"💾 Model saved to {save_path}")
        
        return save_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='kws', choices=['kws', 'speaker', 'both'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    # Config
    config = {
        'task': args.task,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr
    }
    
    print("🎯 Training Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model
    print("\n🧠 Creating model...")
    model = DualTaskSNN(
        input_dim=cfg.audio.n_mfcc,
        time_steps=cfg.spiking.time_steps,
        num_keywords=6,
        backbone_type='lightweight'
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Load data
    print("\n📊 Loading data...")
    try:
        loaders = get_keyword_dataloaders(cfg, batch_size=args.batch_size, num_workers=0)
        train_loader = loaders['train']
        val_loader = loaders['val']
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Please ensure dataset is available or run with synthetic data.")
        return
    
    # Create trainer
    trainer = SimpleTrainer(model, config)
    
    # Train
    trainer.train(train_loader, val_loader, args.epochs)

if __name__ == "__main__":
    main()