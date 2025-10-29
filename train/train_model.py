# train/train_model.py
"""
Main training script
"""
import argparse
import yaml
import torch
from pathlib import Path

from configs.config import cfg
from models.dual_task_snn import DualTaskSNN
from train.trainer import SNNTrainer
from utils.data_loader import get_keyword_dataloaders, get_speaker_dataloaders

def main(args):
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        # Default training config
        train_config = {
            'epochs': 100,
                        'batch_size': 32,
            'lr_backbone': 0.001,
            'lr_head': 0.01,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'weight_decay': 1e-4,
            'grad_clip': 1.0,
            'patience': 10,
            'kws_weight': 1.0,
            'speaker_weight': 1.0,
            'num_workers': 4
        }
    
    print("📋 Training Configuration:")
    for key, value in train_config.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    print("\n📊 Loading datasets...")
    
    if args.task in ['kws', 'both']:
        kws_loaders = get_keyword_dataloaders(
            cfg, 
            batch_size=train_config['batch_size'],
            num_workers=train_config['num_workers']
        )
    
    if args.task in ['speaker', 'both']:
        speaker_loaders = get_speaker_dataloaders(
            cfg,
            batch_size=train_config['batch_size'],
            num_workers=train_config['num_workers'],
            n_speakers=args.num_speakers
        )
    
    # Combine loaders for multi-task training
    if args.task == 'both':
        # Create combined dataset wrapper
        from torch.utils.data import Dataset
        
        class CombinedDataset(Dataset):
            def __init__(self, kws_dataset, speaker_dataset):
                self.kws_dataset = kws_dataset
                self.speaker_dataset = speaker_dataset
                self.len = max(len(kws_dataset), len(speaker_dataset))
            
            def __len__(self):
                return self.len
            
            def __getitem__(self, idx):
                kws_idx = idx % len(self.kws_dataset)
                speaker_idx = idx % len(self.speaker_dataset)
                
                kws_data = self.kws_dataset[kws_idx]
                speaker_data = self.speaker_dataset[speaker_idx]
                
                # Combine data
                return {
                    'spikes': kws_data['spikes'],  # Use KWS spikes
                    'label': kws_data['label'],
                    'speaker_idx': speaker_data['speaker_idx'],
                    'speaker_id': speaker_data['speaker_id']
                }
        
        # Create combined loaders
        train_dataset = CombinedDataset(
            kws_loaders['train'].dataset,
            speaker_loaders['train'].dataset
        )
        val_dataset = CombinedDataset(
            kws_loaders['val'].dataset,
            speaker_loaders['val'].dataset
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=train_config['num_workers']
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config['num_workers']
        )
    elif args.task == 'kws':
        train_loader = kws_loaders['train']
        val_loader = kws_loaders['val']
    else:  # speaker
        train_loader = speaker_loaders['train']
        val_loader = speaker_loaders['val']
    
    # Create model
    print("\n🧠 Creating SNN model...")
    model = DualTaskSNN(
        input_dim=cfg.audio.n_mfcc,
        time_steps=cfg.spiking.time_steps,
        num_keywords=len(cfg.data.target_keywords),
        num_speakers=args.num_speakers if args.task != 'kws' else None,
        backbone_type=args.backbone,
        shared_features=args.shared_features
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = SNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        checkpoint_dir=f'checkpoints/{args.experiment_name}',
        log_dir=f'logs/{args.experiment_name}'
    )
    
    # Load checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    trainer.train(epochs=train_config['epochs'])
    
    print("\n✨ Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SNN model')
    parser.add_argument('--task', type=str, default='both',
                       choices=['kws', 'speaker', 'both'],
                       help='Training task')
    parser.add_argument('--backbone', type=str, default='lightweight',
                       choices=['lightweight', 'full'],
                       help='Backbone architecture')
    parser.add_argument('--num_speakers', type=int, default=100,
                       help='Number of speakers for training')
    parser.add_argument('--shared_features', action='store_true',
                       help='Use shared features for both tasks')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to training config file')
    parser.add_argument('--experiment_name', type=str, 
                       default='snn_dual_task',
                       help='Experiment name for saving')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)