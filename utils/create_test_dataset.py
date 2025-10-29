# create_test_dataset.py
"""
Create minimal test dataset for validation
"""
import numpy as np
import soundfile as sf
from pathlib import Path
import os

def create_test_audio_files():
    """Create synthetic audio files for testing"""
    print("🎵 Creating test audio dataset...")
    
    # Setup paths
    base_path = Path("data/raw/speech_commands")
    keywords = ["yes", "no", "stop", "go", "up", "down"]
    
    # Audio parameters
    sample_rate = 16000
    duration = 1.0  # 1 second
    n_samples_per_keyword = 20
    
    for keyword in keywords:
        keyword_path = base_path / keyword
        keyword_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating samples for '{keyword}'...")
        
        for i in range(n_samples_per_keyword):
            # Generate different synthetic audio for each keyword
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Different frequency pattern for each keyword
            base_freq = 200 + keywords.index(keyword) * 100
            
            # Create audio with harmonics
            audio = np.zeros_like(t)
            for harmonic in range(1, 4):
                audio += (1.0 / harmonic) * np.sin(2 * np.pi * base_freq * harmonic * t)
            
            # Add some noise
            audio += 0.1 * np.random.randn(len(audio))
            
            # Add envelope
            envelope = np.exp(-2 * t) * (1 - np.exp(-10 * t))
            audio *= envelope
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Save
            file_path = keyword_path / f"{keyword}_{i:04d}.wav"
            sf.write(file_path, audio, sample_rate)
        
        print(f"  ✅ Created {n_samples_per_keyword} samples")
    
    print("\n✅ Test dataset created successfully!")

if __name__ == "__main__":
    create_test_audio_files()