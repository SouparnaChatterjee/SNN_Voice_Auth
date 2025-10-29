# utils/data_loader.py
"""
Data loading utilities and helpers
"""
from torch.utils.data import DataLoader
from typing import Dict, Optional
import torch

from utils.dataset import SpeechCommandsDataset, VoxCelebDataset
from configs.config import Config

def get_keyword_dataloaders(config: Config, 
                           batch_size: int = 32,
                           num_workers: int = 4) -> Dict[str, DataLoader]:
    """Get data loaders for keyword spotting task"""
    
    # Create datasets
    train_dataset = SpeechCommandsDataset(config, split='train')
    val_dataset = SpeechCommandsDataset(config, split='val')
    test_dataset = SpeechCommandsDataset(config, split='test')
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # CPU only
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def get_speaker_dataloaders(config: Config,
                           batch_size: int = 32,
                           num_workers: int = 4,
                           n_speakers: int = 100) -> Dict[str, DataLoader]:
    """Get data loaders for speaker verification task"""
    
    # Create datasets
    train_dataset = VoxCelebDataset(
        config, split='train', n_speakers=n_speakers
    )
    val_dataset = VoxCelebDataset(
        config, split='val', n_speakers=n_speakers
    )
    test_dataset = VoxCelebDataset(
        config, split='test', n_speakers=n_speakers
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset  # For triplet sampling
    }

# Test script
if __name__ == "__main__":
    from configs.config import cfg
    import matplotlib.pyplot as plt
    
    # Test keyword spotting dataset
    print("Testing Speech Commands Dataset...")
    kws_loaders = get_keyword_dataloaders(cfg, batch_size=4, num_workers=0)
    
    # Get a batch
    batch = next(iter(kws_loaders['train']))
    print(f"Spike shape: {batch['spikes'].shape}")
    print(f"Labels: {batch['label']}")
    
    # Visualize spikes
    from utils.spike_encoding import plot_spike_train
    plot_spike_train(
        batch['spikes'][0].numpy(),
        title="Sample Spike Train - Keyword Spotting"
    )
    
    # Test speaker verification dataset
    print("\nTesting VoxCeleb Dataset...")
    speaker_loaders = get_speaker_dataloaders(
        cfg, batch_size=4, num_workers=0, n_speakers=10
    )
    
    # Get triplet batch
    train_dataset = speaker_loaders['train_dataset']
    triplet_batch = train_dataset.get_triplet_batch(batch_size=4)
    
    print(f"Anchor shape: {triplet_batch['anchor'].shape}")
    print(f"Positive shape: {triplet_batch['positive'].shape}")
    print(f"Negative shape: {triplet_batch['negative'].shape}")