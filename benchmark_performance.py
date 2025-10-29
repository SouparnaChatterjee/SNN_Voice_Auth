# benchmark_performance.py (Fixed to handle DualTaskSNN)
"""
Benchmark model performance
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path

def benchmark_pytorch_model():
    """Benchmark PyTorch model"""
    print("\n[PYTORCH BENCHMARK]")
    print("-" * 40)
    
    # Load model
    model_path = "experiments/simple_training/model.pt"
    if not os.path.exists(model_path):
        print("[ERROR] PyTorch model not found")
        return None
    
    try:
        # Try loading as DualTaskSNN first
        from models.dual_task_snn import DualTaskSNN
        from configs.config import cfg
        
        model = DualTaskSNN(
            input_dim=cfg.audio.n_mfcc,
            time_steps=cfg.spiking.time_steps,
            num_keywords=6,
            backbone_type='lightweight'
        )
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create appropriate input
        input_tensor = torch.randn(1, 13, 100)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(input_tensor, task='kws')
        
        # Time
        times = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                output = model(input_tensor, task='kws')
            end = time.perf_counter()
            times.append((end - start) * 1000)
            
    except Exception as e:
        print(f"[WARNING] Could not load as DualTaskSNN: {e}")
        print("[INFO] Trying SimpleKWSModel...")
        
        # Fallback to simple model
        from export_onnx_simple import create_simple_model
        
        model = create_simple_model()
        # Don't try to load state dict if architectures don't match
        model.eval()
        
        input_tensor = torch.randn(1, 13, 100)
        
        # Time
        times = []
        for _ in range(50):
            start = time.perf_counter()
            with torch.no_grad():
                output = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Average inference time: {avg_time:.2f} +/- {std_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.1f} inferences/second")
    
    # Model size
    model_size = os.path.getsize(model_path) / 1024 / 1024
    print(f"Model size: {model_size:.2f} MB")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    return {"avg_ms": avg_time, "std_ms": std_time, "size_mb": model_size}

def benchmark_onnx_model():
    """Benchmark ONNX model"""
    print("\n[ONNX BENCHMARK]")
    print("-" * 40)
    
    model_path = "export/onnx/kws_model.onnx"
    if not os.path.exists(model_path):
        print("[ERROR] ONNX model not found")
        return None
    
    # Load model
    session = ort.InferenceSession(model_path)
    
    # Print model info
    print(f"Inputs: {[i.name for i in session.get_inputs()]}")
    print(f"Outputs: {[o.name for o in session.get_outputs()]}")
    
    # Benchmark
    input_data = np.random.randn(1, 13, 100).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        _ = session.run(None, {'audio_features': input_data})
    
    # Time
    times = []
    for _ in range(100):
        start = time.perf_counter()
        output = session.run(None, {'audio_features': input_data})
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Average inference time: {avg_time:.2f} +/- {std_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.1f} inferences/second")
    
    # Model size
    model_size = os.path.getsize(model_path) / 1024 / 1024
    print(f"Model size: {model_size:.2f} MB")
    
    return {"avg_ms": avg_time, "std_ms": std_time, "size_mb": model_size}

def main():
    print("[MODEL PERFORMANCE BENCHMARK]")
    print("=" * 40)
    
    pytorch_stats = benchmark_pytorch_model()
    onnx_stats = benchmark_onnx_model()
    
    if pytorch_stats and onnx_stats:
        print("\n[COMPARISON]")
        print("-" * 40)
        speedup = pytorch_stats['avg_ms'] / onnx_stats['avg_ms']
        compression = pytorch_stats['size_mb'] / onnx_stats['size_mb']
        
        print(f"ONNX speedup: {speedup:.2f}x faster")
        print(f"ONNX compression: {compression:.2f}x smaller")
        
        print("\n[DEPLOYMENT METRICS]")
        print("-" * 40)
        print(f"Real-time capable: {'YES' if onnx_stats['avg_ms'] < 100 else 'NO'}")
        print(f"Suitable for edge devices: {'YES' if onnx_stats['size_mb'] < 10 else 'MAYBE'}")
        print(f"Battery friendly: {'YES' if onnx_stats['avg_ms'] < 50 else 'MODERATE'}")
    
    print("\n[SUMMARY]")
    print("-" * 40)
    if onnx_stats:
        print(f"ONNX model ready for deployment!")
        print(f"- Size: {onnx_stats['size_mb']*1024:.1f} KB")
        print(f"- Speed: {onnx_stats['avg_ms']:.1f} ms/inference")
        print(f"- Throughput: {1000/onnx_stats['avg_ms']:.0f} inferences/sec")

if __name__ == "__main__":
    main()