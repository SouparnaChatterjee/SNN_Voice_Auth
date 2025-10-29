# week3_workflow.py
"""
Complete Week 3 workflow - Train, Export, Deploy
"""
import os
import sys
import subprocess
from pathlib import Path

def run_step(step_name, command):
    """Run a step with error handling"""
    print(f"\n{'='*50}")
    print(f"[START] {step_name}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] {step_name} completed successfully!")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"[ERROR] {step_name} failed!")
            if result.stderr:
                print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

def main():
    print("[TARGET] Week 3 Complete Workflow")
    print("=" * 60)
    
    # Step 1: Train simple model
    print("\n1️⃣ Training simple model...")
    if run_step("Training", "python train_simple.py"):
        print("   Training completed!")
    else:
        print("   Using pre-trained model or skipping...")
    
    # Step 2: Export to ONNX
    print("\n2️⃣ Exporting to ONNX...")
    if run_step("ONNX Export", "python export_onnx_simple.py"):
        print("   ONNX model ready!")
    
    # Step 3: Run inference demo
    print("\n3️⃣ Running inference demo...")
    if run_step("Inference Demo", "python inference_demo.py"):
        print("   Inference working!")
    
    # Step 4: Summary
    print("\n" + "="*60)
    print("[INFO] Week 3 Summary")
    print("="*60)
    
    # Check what was created
    artifacts = {
        "Trained Models": list(Path("experiments").glob("*/model.pt")),
        "ONNX Models": list(Path("export/onnx").glob("*.onnx")),
        "Checkpoints": list(Path("checkpoints").glob("**/*.pt"))
    }
    
    for category, files in artifacts.items():
        print(f"\n{category}:")
        if files:
            for f in files[:3]:  # Show first 3
                print(f"  [OK] {f.relative_to('.')}")
        else:
            print(f"  [WARNING]  None found")
    
    print("\n[DONE] Week 3 workflow complete!")
    print("\n[DOCS] Next steps:")
    print("1. Train longer for better accuracy: python train/train_full.py --epochs 50")
    print("2. Export trained models: python export/export_onnx.py --model <path>")
    print("3. Build web demo: python deploy/web_demo.py")

if __name__ == "__main__":
    main()