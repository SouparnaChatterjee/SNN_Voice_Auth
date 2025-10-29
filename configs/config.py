# configs/config.py
"""
Central configuration for the project
"""
from dataclasses import dataclass, field
from typing import List
import os

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mfcc: int = 13
    n_fft: int = 512
    hop_length: int = 160  # 10ms hop
    win_length: int = 400  # 25ms window
    n_mels: int = 40
    
    # Audio segment parameters
    audio_length_ms: int = 1000  # 1 second clips
    
    @property
    def audio_length_samples(self):
        return int(self.sample_rate * self.audio_length_ms / 1000)

@dataclass
class SpikingConfig:
    # Spike encoding parameters
    encoding_type: str = "rate"  # "rate", "latency", "delta"
    time_steps: int = 100  # Number of time steps for SNN
    threshold: float = 1.0  # Spike threshold
    tau: float = 2.0  # Membrane time constant
    
    # Rate coding parameters
    max_spike_rate: float = 100.0  # Hz
    min_spike_rate: float = 0.0
    
    # Delta coding parameters
    delta_threshold: float = 0.1

@dataclass
class DataConfig:
    # Paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    cache_path: str = "data/cache"
    
    # Speech Commands config
    speech_commands_path: str = os.path.join(raw_data_path, "speech_commands")
    target_keywords: List[str] = field(default_factory=lambda: ["hey", "sasken", "yes", "no", "stop", "go"])
    
    # VoxCeleb config
    voxceleb_path: str = os.path.join(raw_data_path, "voxceleb")
    n_speakers: int = 100  # Use subset for faster training
    
    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Preprocessing
    use_cache: bool = True
    num_workers: int = 4

@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    spiking: SpikingConfig = field(default_factory=SpikingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # General
    seed: int = 42
    device: str = "cpu"
    
# Global config instance
cfg = Config()
