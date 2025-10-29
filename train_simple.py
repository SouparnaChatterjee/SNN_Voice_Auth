# train_simple.py
"""Simplified training script for Week 3"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Import our modules
from models.dual_task_snn import DualTaskSNN
from configs.config import cfg
from utils.data_loader import get_keyword_dataloaders

def train_simple_model():
    """Train a simple keyword spotting model"""
    print("[TARGET] Starting simple training...")
    
    # Create model
    model = DualTaskSNN(
        input_dim=cfg.audio.n_mfcc,
        time_steps=cfg.spiking.time_steps,
        num_keywords=6,
        backbone_type='lightweight'
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Simple optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Try to load data
    try:
        loaders = get_keyword_dataloaders(cfg, batch_size=8, num_workers=0)
        train_loader = loaders['train']
        print(f"Loaded {len(train_loader)} batches")
    except:
        print("[WARNING]  Could not load real data, using synthetic batches")
        train_loader = None
    
    # Training loop (synthetic if no real data)
    model.train()
    for epoch in range(5):
        total_loss = 0
        
        if train_loader:
            # Real data
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                outputs = model(batch['spikes'], task='kws')
                loss = criterion(outputs['kws']['logits'], batch['label'].squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        else:
            # Synthetic data
            for _ in range(10):
                batch_input = torch.randn(8, 13, 100)
                batch_labels = torch.randint(0, 6, (8,))
                
                optimizer.zero_grad()
                outputs = model(batch_input, task='kws')
                loss = criterion(outputs['kws']['logits'], batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / (len(train_loader) if train_loader else 10)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs("experiments/simple_training", exist_ok=True)
    save_path = "experiments/simple_training/model.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'task': 'kws', 'epochs': 5}
    }, save_path)
    
    print(f"[OK] Model saved to {save_path}")
    return save_path

if __name__ == "__main__":
    model_path = train_simple_model()
