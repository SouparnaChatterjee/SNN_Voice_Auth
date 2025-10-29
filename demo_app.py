# demo_app.py
"""
Complete SNN Voice Authentication Demo
"""
import os
import sys
import argparse
from pathlib import Path

def print_banner():
    print("\n" + "="*60)
    print(" SNN Voice Authentication System - Demo")
    print(" Keyword Spotting + Speaker Verification")
    print("="*60 + "\n")

def run_demo(mode="simulate"):
    """Run the demo application"""
    
    print_banner()
    
    # Check prerequisites
    print("[1] Checking system...")
    
    requirements = {
        "ONNX Model": Path("export/onnx/kws_model.onnx"),
        "PyTorch Model": Path("experiments/simple_training/model.pt"),
        "Inference Script": Path("inference_demo.py")
    }
    
    all_ready = True
    for item, path in requirements.items():
        if path.exists():
            print(f"  [OK] {item}")
        else:
            print(f"  [MISSING] {item}")
            all_ready = False
    
    if not all_ready:
        print("\n[ERROR] Some requirements missing!")
        print("Run: python week3_workflow.py")
        return
    
    print("\n[2] Running Demo...")
    print("-" * 40)
    
    if mode == "inference":
        # Run basic inference demo
        os.system("python inference_demo.py")
    
    elif mode == "realtime":
        # Run real-time demo
        cmd = "python deploy/realtime_demo_simple.py"
        if not _check_audio_available():
            cmd += " --simulate --duration 10"
        os.system(cmd)
    
    elif mode == "benchmark":
        # Run performance benchmark
        os.system("python benchmark_performance.py")
    
    elif mode == "all":
        # Run all demos
        print("\n[INFERENCE DEMO]")
        os.system("python inference_demo.py")
        
        print("\n\n[PERFORMANCE BENCHMARK]")
        os.system("python benchmark_performance.py")
        
        print("\n\n[REAL-TIME SIMULATION]")
        os.system("python deploy/realtime_demo_simple.py --simulate --duration 5")
    
    print("\n[3] Demo Complete!")

def _check_audio_available():
    """Check if audio input is available"""
    try:
        import sounddevice as sd
        return len(sd.query_devices()) > 0
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="SNN Voice Auth Demo")
    parser.add_argument('--mode', choices=['inference', 'realtime', 'benchmark', 'all'],
                       default='all', help='Demo mode to run')
    
    args = parser.parse_args()
    run_demo(args.mode)

if __name__ == "__main__":
    main()