# deploy/realtime_demo_simple.py
"""
Simple real-time inference demo
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import threading
import queue
from pathlib import Path

try:
    import onnxruntime as ort
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("[WARNING] sounddevice not installed. Install with: pip install sounddevice")

class RealtimeDemo:
    """Simple real-time keyword detection"""
    
    def __init__(self, model_path="export/onnx/kws_model.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load ONNX model
        print(f"[INFO] Loading model from {model_path}")
        self.session = ort.InferenceSession(model_path)
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 1.0  # 1 second chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Keywords
        self.keywords = ["yes", "no", "stop", "go", "up", "down"]
        
        # Audio buffer
        self.audio_queue = queue.Queue()
        self.is_running = False
        
    def audio_callback(self, indata, frames, time, status):
        """Audio input callback"""
        if status:
            print(f"[WARNING] Audio status: {status}")
        
        # Add audio to queue
        self.audio_queue.put(indata[:, 0].copy())
    
    def process_audio_chunk(self, audio_chunk):
        """Process one audio chunk"""
        # Simple feature extraction (mock MFCC)
        # In real implementation, use proper MFCC extraction
        n_features = 13
        n_time_steps = 100
        
        # Create mock features (replace with real MFCC extraction)
        features = np.random.randn(1, n_features, n_time_steps).astype(np.float32)
        
        # Add some correlation with audio energy
        energy = np.sqrt(np.mean(audio_chunk**2))
        features[0, 0, :] *= energy * 10
        
        # Run inference
        outputs = self.session.run(None, {'audio_features': features})
        logits = outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Get prediction
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        return self.keywords[pred_idx], confidence
    
    def run_from_microphone(self):
        """Run real-time inference from microphone"""
        if not HAS_AUDIO:
            print("[ERROR] sounddevice not available")
            return
        
        print("[INFO] Starting real-time inference...")
        print(f"[INFO] Listening for keywords: {', '.join(self.keywords)}")
        print("[INFO] Press Ctrl+C to stop\n")
        
        self.is_running = True
        
        # Start audio stream
        with sd.InputStream(samplerate=self.sample_rate,
                           channels=1,
                           callback=self.audio_callback,
                           blocksize=int(self.sample_rate * 0.05)):  # 50ms blocks
            
            audio_buffer = []
            last_prediction_time = 0
            
            while self.is_running:
                try:
                    # Get audio chunk
                    audio_block = self.audio_queue.get(timeout=0.1)
                    audio_buffer.extend(audio_block)
                    
                    # Process when we have enough audio
                    if len(audio_buffer) >= self.chunk_size:
                        audio_chunk = np.array(audio_buffer[:self.chunk_size])
                        audio_buffer = audio_buffer[self.chunk_size//2:]  # 50% overlap
                        
                        # Get prediction
                        keyword, confidence = self.process_audio_chunk(audio_chunk)
                        
                        # Print if high confidence and not too frequent
                        current_time = time.time()
                        if confidence > 0.6 and (current_time - last_prediction_time) > 1.0:
                            print(f"[DETECTED] '{keyword}' (confidence: {confidence:.2f})")
                            last_prediction_time = current_time
                
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
        
        print("\n[INFO] Stopped")
    
    def run_simulation(self, duration=5):
        """Run simulated inference (no microphone needed)"""
        print("[INFO] Running simulation mode...")
        print(f"[INFO] Simulating {duration} seconds of audio\n")
        
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            # Simulate audio chunk
            audio_chunk = np.random.randn(self.chunk_size) * 0.1
            
            # Process
            keyword, confidence = self.process_audio_chunk(audio_chunk)
            
            # Randomly make some detections
            if np.random.rand() > 0.7:
                print(f"[DETECTED] '{keyword}' (confidence: {confidence:.2f})")
            
            time.sleep(0.5)
        
        print("\n[INFO] Simulation complete")

def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='export/onnx/kws_model.onnx')
    parser.add_argument('--simulate', action='store_true', 
                       help='Run simulation without microphone')
    parser.add_argument('--duration', type=int, default=10,
                       help='Simulation duration in seconds')
    
    args = parser.parse_args()
    
    try:
        demo = RealtimeDemo(args.model)
        
        if args.simulate or not HAS_AUDIO:
            demo.run_simulation(args.duration)
        else:
            demo.run_from_microphone()
            
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Please run export_onnx_simple.py first")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()