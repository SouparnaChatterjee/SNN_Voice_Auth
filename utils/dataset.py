# utils/dataset.py
"""
PyTorch datasets for keyword spotting and speaker verification
"""
import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import hashlib
from pathlib import Path

from utils.audio_utils import AudioProcessor, VAD
from utils.spike_encoding import RateEncoder, DeltaEncoder, LatencyEncoder
from configs.config import Config

class SpeechCommandsDataset(Dataset):
    """Dataset for keyword spotting task"""
    
    def __init__(self, 
                 config: Config,
                 split: str = 'train',
                 transform=None,
                 use_cache: bool = True):
        
        self.config = config
        self.split = split
        self.transform = transform
        self.use_cache = use_cache
        
        self.audio_processor = AudioProcessor(config)
        self.encoder = self._get_encoder()
        
        # Setup paths
        self.data_path = Path(config.data.speech_commands_path)
        self.cache_dir = Path(config.data.cache_path) / f"speech_commands_{split}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load file list
        self.samples = self._load_samples()
        
        # Create label mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(config.data.target_keywords)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
    def _get_encoder(self):
        """Get spike encoder based on config"""
        if self.config.spiking.encoding_type == "rate":
            return RateEncoder(
                time_steps=self.config.spiking.time_steps,
                max_rate=self.config.spiking.max_spike_rate
            )
        elif self.config.spiking.encoding_type == "latency":
            return LatencyEncoder(
                time_steps=self.config.spiking.time_steps
            )
        elif self.config.spiking.encoding_type == "delta":
            return DeltaEncoder(
                time_steps=self.config.spiking.time_steps,
                threshold=self.config.spiking.delta_threshold
            )
        else:
            raise ValueError(f"Unknown encoding type: {self.config.spiking.encoding_type}")
    
    def _load_samples(self) -> List[Dict]:
        """Load and split dataset samples"""
        samples = []
        
        # Check for cached split
        split_cache_file = self.cache_dir / f"{self.split}_files.json"
        if split_cache_file.exists() and self.use_cache:
            with open(split_cache_file, 'r') as f:
                return json.load(f)
        
        # Process all target keywords
        all_files = []
        for keyword in self.config.data.target_keywords:
            keyword_dir = self.data_path / keyword
            if keyword_dir.exists():
                files = list(keyword_dir.glob("*.wav"))
                for f in files:
                    all_files.append({
                        'path': str(f),
                        'label': keyword,
                        'label_idx': self.label_to_idx.get(keyword, -1)
                    })
        
        # Create splits
        np.random.seed(self.config.seed)
        np.random.shuffle(all_files)
        
        n_total = len(all_files)
        n_train = int(n_total * self.config.data.train_ratio)
        n_val = int(n_total * self.config.data.val_ratio)
        
        if self.split == 'train':
            samples = all_files[:n_train]
        elif self.split == 'val':
            samples = all_files[n_train:n_train + n_val]
        else:  # test
            samples = all_files[n_train + n_val:]
        
        # Cache the split
        with open(split_cache_file, 'w') as f:
            json.dump(samples, f)
            
        return samples
    
    def _get_cache_path(self, idx: int) -> Path:
        """Generate cache filename for processed sample"""
        sample = self.samples[idx]
        # Create hash of file path for unique cache name
        hash_id = hashlib.md5(sample['path'].encode()).hexdigest()[:8]
        return self.cache_dir / f"{hash_id}.pkl"
    
    def _process_audio(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process audio file and extract features"""
        # Load and preprocess audio
        audio, sr = self.audio_processor.load_audio(audio_path)
        audio = self.audio_processor.normalize_audio(audio)
        audio = self.audio_processor.pad_or_truncate(
            audio, self.config.audio.audio_length_samples
        )
        
        # Apply augmentations for training
        if self.split == 'train' and np.random.rand() > 0.5:
            if np.random.rand() > 0.5:
                audio = self.audio_processor.add_noise(audio, noise_level=0.005)
            if np.random.rand() > 0.7:
                audio = self.audio_processor.time_stretch(audio, rate=np.random.uniform(0.9, 1.1))
                audio = self.audio_processor.pad_or_truncate(
                    audio, self.config.audio.audio_length_samples
                )
        
        # Extract features
        mfcc = self.audio_processor.extract_mfcc(audio)
        
        # Encode to spikes
        spike_train = self.encoder.encode(mfcc)
        
        return spike_train, audio
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        sample = self.samples[idx]
        
        # Check cache
        cache_path = self._get_cache_path(idx)
        if cache_path.exists() and self.use_cache:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                return {
                    'spikes': torch.FloatTensor(cached_data['spikes']),
                    'label': torch.LongTensor([cached_data['label']]),
                    'audio': torch.FloatTensor(cached_data['audio'])
                                }
        
        # Process audio
        spike_train, audio = self._process_audio(sample['path'])
        
        # Cache processed data
        if self.use_cache:
            cache_data = {
                'spikes': spike_train,
                'label': sample['label_idx'],
                'audio': audio
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        
        # Apply additional transforms if specified
        if self.transform:
            spike_train = self.transform(spike_train)
        
        return {
            'spikes': torch.FloatTensor(spike_train),
            'label': torch.LongTensor([sample['label_idx']]),
            'audio': torch.FloatTensor(audio)
        }
    
    def __len__(self) -> int:
        return len(self.samples)

class VoxCelebDataset(Dataset):
    """Dataset for speaker verification task"""
    
    def __init__(self,
                 config: Config,
                 split: str = 'train',
                 n_speakers: int = 100,
                 samples_per_speaker: int = 10,
                 use_cache: bool = True):
        
        self.config = config
        self.split = split
        self.n_speakers = n_speakers
        self.samples_per_speaker = samples_per_speaker
        self.use_cache = use_cache
        
        self.audio_processor = AudioProcessor(config)
        self.encoder = self._get_encoder()
        self.vad = VAD()
        
        # Setup paths
        self.data_path = Path(config.data.voxceleb_path)
        self.cache_dir = Path(config.data.cache_path) / f"voxceleb_{split}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load speaker data
        self.speakers, self.samples = self._load_speakers()
        
        print(f"Loaded {len(self.speakers)} speakers with {len(self.samples)} total samples")
    
    def _get_encoder(self):
        """Get spike encoder based on config"""
        if self.config.spiking.encoding_type == "rate":
            return RateEncoder(
                time_steps=self.config.spiking.time_steps,
                max_rate=self.config.spiking.max_spike_rate
            )
        elif self.config.spiking.encoding_type == "latency":
            return LatencyEncoder(
                time_steps=self.config.spiking.time_steps
            )
        elif self.config.spiking.encoding_type == "delta":
            return DeltaEncoder(
                time_steps=self.config.spiking.time_steps,
                threshold=self.config.spiking.delta_threshold
            )
        else:
            raise ValueError(f"Unknown encoding type: {self.config.spiking.encoding_type}")
    
    def _load_speakers(self) -> Tuple[List[str], List[Dict]]:
        """Load speaker IDs and audio files"""
        # Check for cached data
        cache_file = self.cache_dir / f"speakers_{self.n_speakers}_{self.split}.pkl"
        if cache_file.exists() and self.use_cache:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                return data['speakers'], data['samples']
        
        # Scan for speaker directories
        all_speakers = []
        if self.data_path.exists():
            speaker_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
            all_speakers = sorted(speaker_dirs)[:self.n_speakers]
        
        if not all_speakers:
            print("⚠️  VoxCeleb dataset not found. Using synthetic data for testing.")
            return self._create_synthetic_data()
        
        # Split speakers by train/val/test
        np.random.seed(self.config.seed)
        np.random.shuffle(all_speakers)
        
        n_total = len(all_speakers)
        n_train = int(n_total * self.config.data.train_ratio)
        n_val = int(n_total * self.config.data.val_ratio)
        
        if self.split == 'train':
            speakers = all_speakers[:n_train]
        elif self.split == 'val':
            speakers = all_speakers[n_train:n_train + n_val]
        else:
            speakers = all_speakers[n_train + n_val:]
        
        # Collect samples for each speaker
        samples = []
        for speaker_idx, speaker_dir in enumerate(speakers):
            audio_files = list(speaker_dir.glob("**/*.wav"))[:self.samples_per_speaker]
            
            for audio_file in audio_files:
                samples.append({
                    'path': str(audio_file),
                    'speaker_id': speaker_dir.name,
                    'speaker_idx': speaker_idx
                })
        
        # Cache the data
        cache_data = {'speakers': [s.name for s in speakers], 'samples': samples}
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        return cache_data['speakers'], samples
    
    def _create_synthetic_data(self) -> Tuple[List[str], List[Dict]]:
        """Create synthetic data for testing when VoxCeleb is not available"""
        speakers = [f"speaker_{i:03d}" for i in range(self.n_speakers)]
        samples = []
        
        for speaker_idx, speaker_id in enumerate(speakers):
            for i in range(self.samples_per_speaker):
                samples.append({
                    'path': None,  # Will generate synthetic audio
                    'speaker_id': speaker_id,
                    'speaker_idx': speaker_idx,
                    'synthetic': True
                })
        
        return speakers, samples
    
    def _generate_synthetic_audio(self, speaker_idx: int) -> np.ndarray:
        """Generate synthetic audio for a speaker"""
        # Create speaker-specific characteristics
        base_freq = 100 + speaker_idx * 10  # Different base frequency per speaker
        harmonics = [1, 2, 3, 4]  # Harmonic structure
        
        t = np.linspace(0, self.config.audio.audio_length_ms / 1000, 
                        self.config.audio.audio_length_samples)
        
        audio = np.zeros_like(t)
        for h in harmonics:
            audio += (1.0 / h) * np.sin(2 * np.pi * base_freq * h * t)
        
        # Add some noise
        audio += 0.1 * np.random.randn(len(audio))
        
        return audio / np.max(np.abs(audio))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        sample = self.samples[idx]
        
        # Process audio
        if sample.get('synthetic', False):
            audio = self._generate_synthetic_audio(sample['speaker_idx'])
        else:
            audio, sr = self.audio_processor.load_audio(sample['path'])
            audio = self.audio_processor.normalize_audio(audio)
            
            # Find speech segments using VAD
            if self.vad.is_speech(audio):
                audio = self.audio_processor.pad_or_truncate(
                    audio, self.config.audio.audio_length_samples
                )
            else:
                # If no speech detected, use the whole audio
                audio = self.audio_processor.pad_or_truncate(
                    audio, self.config.audio.audio_length_samples
                )
        
        # Extract features
        mfcc = self.audio_processor.extract_mfcc(audio)
        
        # Encode to spikes
        spike_train = self.encoder.encode(mfcc)
        
        return {
            'spikes': torch.FloatTensor(spike_train),
            'speaker_idx': torch.LongTensor([sample['speaker_idx']]),
            'speaker_id': sample['speaker_id']
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_triplet_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get triplet batch for contrastive learning (anchor, positive, negative)"""
        anchors, positives, negatives = [], [], []
        anchor_labels, positive_labels, negative_labels = [], [], []
        
        for _ in range(batch_size):
            # Select anchor speaker
            anchor_speaker = np.random.choice(len(self.speakers))
            
            # Get samples from same speaker (positive) and different speaker (negative)
            speaker_samples = [s for s in self.samples if s['speaker_idx'] == anchor_speaker]
            other_samples = [s for s in self.samples if s['speaker_idx'] != anchor_speaker]
            
            if len(speaker_samples) >= 2 and len(other_samples) > 0:
                # Select anchor and positive
                anchor_idx = self.samples.index(np.random.choice(speaker_samples))
                positive_idx = self.samples.index(np.random.choice(
                    [s for s in speaker_samples if self.samples.index(s) != anchor_idx]
                ))
                
                # Select negative from different speaker
                negative_idx = self.samples.index(np.random.choice(other_samples))
                
                # Get data
                anchor_data = self[anchor_idx]
                positive_data = self[positive_idx]
                negative_data = self[negative_idx]
                
                anchors.append(anchor_data['spikes'])
                positives.append(positive_data['spikes'])
                negatives.append(negative_data['spikes'])
                
                anchor_labels.append(anchor_data['speaker_idx'])
                positive_labels.append(positive_data['speaker_idx'])
                negative_labels.append(negative_data['speaker_idx'])
        
        return {
            'anchor': torch.stack(anchors),
            'positive': torch.stack(positives),
            'negative': torch.stack(negatives),
            'anchor_label': torch.stack(anchor_labels),
            'positive_label': torch.stack(positive_labels),
            'negative_label': torch.stack(negative_labels)
        }