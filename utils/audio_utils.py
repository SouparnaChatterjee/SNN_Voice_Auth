# utils/audio_utils.py
"""
Audio preprocessing utilities
"""
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Union
import torch
import torchaudio
import torchaudio.transforms as T

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.audio.sample_rate
        
        # Initialize transforms
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=config.audio.n_mfcc,
            melkwargs={
                'n_fft': config.audio.n_fft,
                'hop_length': config.audio.hop_length,
                'n_mels': config.audio.n_mels
            }
        )
        
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            n_mels=config.audio.n_mels
        )
    
    def load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and resample if necessary"""
        audio, sr = librosa.load(path, sr=None, mono=True)
        
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
        return audio, self.sample_rate
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def pad_or_truncate(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate audio to target length"""
        if len(audio) > target_length:
            # Random crop for training
            start = np.random.randint(0, len(audio) - target_length + 1)
            audio = audio[start:start + target_length]
        elif len(audio) < target_length:
            # Pad with zeros
            pad_length = target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
            
        return audio
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features"""
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        mfcc = self.mfcc_transform(audio_tensor)
        return mfcc.squeeze(0).numpy()
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram"""
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        mel_spec = self.mel_spectrogram(audio_tensor)
        mel_spec_db = T.AmplitudeToDB()(mel_spec)
        return mel_spec_db.squeeze(0).numpy()
    
    def add_noise(self, audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
        """Add Gaussian noise for data augmentation"""
        noise = np.random.normal(0, noise_level, audio.shape)
        return audio + noise
    
    def time_stretch(self, audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """Time stretch augmentation"""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, n_steps: int = 0) -> np.ndarray:
        """Pitch shift augmentation"""
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)

# Voice Activity Detection
class VAD:
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30):
        import webrtcvad
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (highest)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_length = int(sample_rate * frame_duration_ms / 1000)
        
    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech"""
        # Convert to 16-bit PCM
        audio_16bit = (audio * 32767).astype(np.int16).tobytes()
        
        # Process in frames
        num_frames = len(audio_16bit) // (self.frame_length * 2)
        speech_frames = 0
        
        for i in range(num_frames):
            start = i * self.frame_length * 2
            end = start + self.frame_length * 2
            frame = audio_16bit[start:end]
            
            if self.vad.is_speech(frame, self.sample_rate):
                speech_frames += 1
        
        # Return True if more than 30% frames contain speech
        return speech_frames > num_frames * 0.3