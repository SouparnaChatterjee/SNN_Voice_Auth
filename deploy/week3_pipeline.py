# week3_pipeline.py
"""
Complete Week 3 training and deployment pipeline
"""
import os
import sys
from pathlib import Path
import subprocess

def check_dataset():
    """Check if dataset is available"""
    data_path = Path("data/raw/speech_commands")
    if not data_path.exists() or len(list(data_path.glob("*/*.wav"))) == 0:
        print("📥 Downloading Speech Commands dataset...")
        
        import urllib.request
        import tarfile
        
        url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
        os.makedirs(data_path, exist_ok=True)
        
        tar_path = "speech_commands.tar.gz"
        urllib.request.urlretrieve(url, tar_path)
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_path)
        os.remove(tar_path)
        
        print("✅ Dataset downloaded!")

def train_models():
    """Train models for both tasks"""
    print("\n🎯 Training Keyword Spotting Model...")
    
    # Train KWS model
    subprocess.run([
        sys.executable, "train/train_full.py",
        "--task", "kws",
        "--epochs", "20",
        "--batch_size", "32"
    ])
    
    print("\n✅ Training complete!")

def quantize_and_export():
    """Quantize and export models"""
    print("\n📦 Quantizing and exporting models...")
    
    # Find latest model
    checkpoints = list(Path("experiments").glob("*/final_model.pt"))
    if not checkpoints:
        print("❌ No trained models found!")
        return None
    
    latest_model = max(checkpoints, key=os.path.getctime)
    print(f"Using model: {latest_model}")
    
    # Quantize
    subprocess.run([
        sys.executable, "export/quantize_model.py",
        "--model", str(latest_model)
    ])
    
    # Export to ONNX
    subprocess.run([
        sys.executable, "export/export_onnx.py",
        "--model", str(latest_model)
    ])
    
    return latest_model

def run_demo():
    """Run real-time demo"""
    print("\n🎙️ Starting real-time demo...")
    
    # Check if ONNX model exists
    onnx_model = Path("export/onnx/kws_model.onnx")
    if not onnx_model.exists():
        print("❌ ONNX model not found. Please train and export first.")
        return
    
    subprocess.run([
        sys.executable, "deploy/realtime_demo.py",
        "--kws_model", str(onnx_model)
    ])

def main():
    """Run complete Week 3 pipeline"""
    print("🚀 Week 3: Training and Deployment Pipeline")
    print("=" * 50)
    
    # Step 1: Check dataset
    print("\n📊 Step 1: Checking dataset...")
    check_dataset()
    
    # Step 2: Train models
    print("\n🧠 Step 2: Training models...")
    response = input("Train new models? (y/n): ")
    if response.lower() == 'y':
        train_models()
    
    # Step 3: Quantize and export
    print("\n📦 Step 3: Model optimization...")
    response = input("Quantize and export models? (y/n): ")
    if response.lower() == 'y':
        model_path = quantize_and_export()
    
    # Step 4: Run demo
    print("\n🎤 Step 4: Real-time demo...")
    response = input("Run real-time inference demo? (y/n): ")
    if response.lower() == 'y':
        run_demo()
    
    print("\n✅ Week 3 pipeline complete!")
    print("\n📁 Generated artifacts:")
    print("- experiments/: Training logs and checkpoints")
    print("- export/quantized/: Quantized models")
    print("- export/onnx/: ONNX models")
    print("- logs/: TensorBoard logs")

if __name__ == "__main__":
    main()