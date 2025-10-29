#!/usr/bin/env python3
"""
Environment setup and verification script
"""
import subprocess
import sys
import os

def check_python_version():
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def install_requirements():
    print("📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ All packages installed")

def create_project_structure():
    print("📁 Creating project folder structure...")
    folders = [
        "data/raw", "data/processed", "data/cache",
        "models", "train", "export", "deploy",
        "utils", "configs", "tests"
    ]
    for f in folders:
        os.makedirs(f, exist_ok=True)
    print("✅ Folder structure ready")

def download_datasets():
    print("📥 Downloading datasets...")
    
    # Create data directories
    os.makedirs("data/raw/speech_commands", exist_ok=True)
    os.makedirs("data/raw/voxceleb", exist_ok=True)
    
    # Download Speech Commands
    import urllib.request
    import tarfile
    
    speech_commands_url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    speech_commands_path = "data/raw/speech_commands/speech_commands_v0.02.tar.gz"
    
    if not os.path.exists(speech_commands_path):
        print("Downloading Speech Commands dataset...")
        urllib.request.urlretrieve(speech_commands_url, speech_commands_path)
        
        with tarfile.open(speech_commands_path, "r:gz") as tar:
            tar.extractall("data/raw/speech_commands/")
        print("✅ Speech Commands downloaded")
    
    print("⚠️  VoxCeleb requires manual download from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/")
    print("   Place files in data/raw/voxceleb/")

if __name__ == "__main__":
    check_python_version()
    install_requirements()
    create_project_structure()
    download_datasets()
