# train/trainer.py
"""
Training framework for SNN models
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
from pathlib import Path

from models.dual_task_snn import DualTaskSNN
from models.losses import CombinedLoss
from utils.metrics import AccuracyMetric, EERMetric

class SNNTrainer:
    """Trainer for SNN models"""
    
    def __init__(self,
                 model: DualTaskSNN,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss
        self.criterion = CombinedLoss(
            kws_weight=config.get('kws_weight', 1.0),
            speaker_weight=config.get('speaker_weight', 1.0)
        )
        
        # Metrics
        self.kws_metric = AccuracyMetric()
        self.speaker_metric = EERMetric()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
    def _setup_optimizer(self):
        """Setup optimizer with different LR for different parts"""
        params = [
            {'params': self.model.backbone.parameters(), 
             'lr': self.config['lr_backbone']},
            {'params': self.model.kws_head.parameters(), 
             'lr': self.config['lr_head']},
            {'params': self.model.speaker_head.parameters(), 
             'lr': self.config['lr_head']}
        ]
        
        if self.config['optimizer'] == 'adam':
            return torch.optim.Adam(params, weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'sgd':
            return torch.optim.SGD(params, momentum=0.9, 
                                 weight_decay=self.config['weight_decay'])
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config['scheduler'] == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config['epochs']
            )
        elif self.config['scheduler'] == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        epoch_losses = {'total': 0, 'kws': 0, 'speaker': 0}
        epoch_metrics = {'kws_acc': 0, 'speaker_eer': 0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device (CPU in this case)
            spikes = batch['spikes']
            
            # Prepare targets
            targets = {}
            if 'label' in batch:
                targets['kws_labels'] = batch['label']
            if 'speaker_idx' in batch:
                targets['speaker_labels'] = batch['speaker_idx']
            
            # Forward pass
            outputs = self.model(spikes, task='both', 
                               labels=targets.get('speaker_labels'))
            
            # Compute loss
            losses = self.criterion(outputs, targets)
            loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['grad_clip']
            )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_losses['total'] += loss.item()
            if 'kws_loss' in losses:
                epoch_losses['kws'] += losses['kws_loss'].item()
            if 'speaker_ce_loss' in losses:
                epoch_losses['speaker'] += losses['speaker_ce_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), step)
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        val_losses = {'total': 0, 'kws': 0, 'speaker': 0}
        all_kws_preds = []
        all_kws_labels = []
        all_speaker_embeddings = []
        all_speaker_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                spikes = batch['spikes']
                
                # Prepare targets
                targets = {}
                if 'label' in batch:
                    targets['kws_labels'] = batch['label']
                    all_kws_labels.extend(batch['label'].numpy())
                if 'speaker_idx' in batch:
                    targets['speaker_labels'] = batch['speaker_idx']
                    all_speaker_labels.extend(batch['speaker_idx'].numpy())
                
                # Forward pass
                outputs = self.model(spikes, task='both',
                                   labels=targets.get('speaker_labels'))
                
                # Compute loss
                losses = self.criterion(outputs, targets)
                val_losses['total'] += losses['total_loss'].item()
                
                # Collect predictions
                if 'kws' in outputs:
                    preds = outputs['kws']['predictions']
                    all_kws_preds.extend(preds.numpy())
                
                if 'speaker' in outputs:
                    embeddings = outputs['speaker']['embeddings']
                    all_speaker_embeddings.append(embeddings)
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        # Calculate metrics
        metrics = {}
        
        # KWS accuracy
        if all_kws_preds and all_kws_labels:
            kws_acc = self.kws_metric.compute(
                np.array(all_kws_preds), 
                np.array(all_kws_labels)
            )
            metrics['kws_accuracy'] = kws_acc
        
        # Speaker verification EER
        if all_speaker_embeddings:
            embeddings = torch.cat(all_speaker_embeddings, dim=0)
            eer = self.speaker_metric.compute(
                embeddings.numpy(),
                np.array(all_speaker_labels)
            )
            metrics['speaker_eer'] = eer
        
        return {**val_losses, **metrics}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model with val_loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"📂 Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, epochs: int):
        """Main training loop"""
        print(f"\n🚀 Starting training for {epochs} epochs\n")
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_metrics['total']:.4f}")
            
            if 'kws_accuracy' in val_metrics:
                print(f"  KWS Accuracy: {val_metrics['kws_accuracy']:.2%}")
            if 'speaker_eer' in val_metrics:
                print(f"  Speaker EER: {val_metrics['speaker_eer']:.2%}")
            
            # Log to tensorboard
            for key, value in train_losses.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            
            # Save checkpoint
            is_best = val_metrics['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total']
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.early_stop_counter >= self.config['patience']:
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                break
        
        print("\n✅ Training completed!")
        self.writer.close()