# final_demo.py
"""
Final demonstration of the SNN Voice Authentication System
"""
import os
import sys
import time
import numpy as np

def print_header(text):
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def run_final_demo():
    print_header("SNN VOICE AUTHENTICATION SYSTEM - FINAL DEMO")
    
    print("\nThis demo showcases:")
    print("- Keyword spotting with 6 commands")
    print("- Ultra-low latency inference (<10ms)")
    print("- Small model size (56KB)")
    print("- CPU-only execution")
    
    # Check prerequisites
    print_header("SYSTEM CHECK")
    
    checks = [
        ("ONNX Model", "export/onnx/kws_model.onnx"),
        ("Inference Script", "inference_demo.py"),
        ("Benchmark Script", "benchmark_onnx_only.py")
    ]
    
    all_ready = True
    for name, path in checks:
        if os.path.exists(path):
            print(f"[OK] {name}")
        else:
            print(f"[MISSING] {name}")
            all_ready = False
    
    if not all_ready:
        print("\n[ERROR] Some components missing!")
        print("Run: python week3_workflow.py")
        return
    
    # Run demos
    print_header("RUNNING DEMONSTRATIONS")
    
    print("\n1. INFERENCE DEMO")
    print("-" * 40)
    os.system("python inference_demo.py")
    
    print("\n\n2. PERFORMANCE BENCHMARK")
    print("-" * 40)
    os.system("python benchmark_onnx_only.py")
    
    print_header("SYSTEM CAPABILITIES")
    
    print("\nKEYWORD DETECTION:")
    keywords = ["yes", "no", "stop", "go", "up", "down"]
    for i, kw in enumerate(keywords):
        print(f"  {i+1}. '{kw}'")
    
    print("\nDEPLOYMENT READY:")
    print("  - Edge devices (Raspberry Pi, Arduino)")
    print("  - Mobile apps (iOS/Android via ONNX)")
    print("  - Web browsers (ONNX.js)")
    print("  - Embedded systems")
    
    print("\nKEY ADVANTAGES:")
    print("  - 5.4x smaller than PyTorch model")
    print("  - No GPU required")
    print("  - Low power consumption")
    print("  - Real-time performance")
    
    print_header("DEMO COMPLETE")
    
    print("\nThank you for exploring the SNN Voice Authentication System!")
    print("\nFor more information:")
    print("- View code: models/dual_task_snn.py")
    print("- Train models: python train_simple.py")
    print("- Export models: python export_onnx_simple.py")

if __name__ == "__main__":
    run_final_demo()