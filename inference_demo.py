# inference_demo.py
"""Simple inference demo"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time

try:
    import onnxruntime as ort
    has_onnx = True
except:
    has_onnx = False
    print("[WARNING]  ONNX Runtime not installed")

def run_inference_demo():
    """Run simple inference demo"""
    print("[MIC] Inference Demo")
    print("=" * 50)
    
    if not has_onnx:
        print("Please install onnxruntime: pip install onnxruntime")
        return
    
    # Check for ONNX model
    model_path = "export/onnx/kws_model.onnx"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run: python export_onnx_simple.py")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    session = ort.InferenceSession(model_path)
    
    # Keywords
    keywords = ["yes", "no", "stop", "go", "up", "down"]
    
    print("\nRunning inference on synthetic audio...")
    print("-" * 50)
    
    # Simulate 5 inferences
    for i in range(5):
        # Generate random input (simulating audio features)
        audio_features = np.random.randn(1, 13, 100).astype(np.float32)
        
        # Run inference
        start_time = time.time()
        outputs = session.run(None, {'audio_features': audio_features})
        inference_time = (time.time() - start_time) * 1000
        
        # Get prediction
        logits = outputs[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        print(f"Sample {i+1}: '{keywords[pred_idx]}' (confidence: {confidence:.2f}, time: {inference_time:.1f}ms)")
    
    print("\n[OK] Inference demo complete!")

if __name__ == "__main__":
    run_inference_demo()
