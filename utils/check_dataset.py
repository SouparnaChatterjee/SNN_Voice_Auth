# check_dataset.py
"""
Check dataset availability and download if needed
"""
import os
from pathlib import Path
from configs.config import cfg

def check_speech_commands():
    """Check if Speech Commands dataset exists"""
    print("🔍 Checking for Speech Commands dataset...")
    
    data_path = Path(cfg.data.speech_commands_path)
    print(f"Looking in: {data_path}")
    
    if not data_path.exists():
        print(f"❌ Path does not exist: {data_path}")
        print("Creating directory...")
        data_path.mkdir(parents=True, exist_ok=True)
        return False
    
    # Check for audio files
    wav_files = list(data_path.glob("*/*.wav"))
    print(f"Found {len(wav_files)} .wav files")
    
    # Check for target keywords
    print(f"\nTarget keywords: {cfg.data.target_keywords}")
    for keyword in cfg.data.target_keywords:
        keyword_path = data_path / keyword
        if keyword_path.exists():
            n_files = len(list(keyword_path.glob("*.wav")))
            print(f"  ✅ '{keyword}': {n_files} files")
        else:
            print(f"  ❌ '{keyword}': directory not found")
    
    return len(wav_files) > 0

def download_speech_commands():
    """Download Speech Commands dataset"""
    import urllib.request
    import tarfile
    import shutil
    
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    data_path = Path(cfg.data.speech_commands_path)
    tar_path = data_path.parent / "speech_commands.tar.gz"
    
    print(f"\n📥 Downloading Speech Commands dataset...")
    print(f"URL: {url}")
    print(f"Destination: {data_path}")
    
    try:
        # Download
        urllib.request.urlretrieve(url, tar_path, 
                                 reporthook=lambda b, bs, t: print(f"Progress: {b*bs/1024/1024:.1f}/{t/1024/1024:.1f} MB", end='\r'))
        print(f"\n✅ Download complete!")
        
        # Extract
        print("📦 Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_path)
        print("✅ Extraction complete!")
        
        # Clean up
        os.remove(tar_path)
        
        return True
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    if not check_speech_commands():
        print("\n⚠️  Dataset not found!")
        response = input("Download Speech Commands dataset? (y/n): ")
        if response.lower() == 'y':
            if download_speech_commands():
                check_speech_commands()