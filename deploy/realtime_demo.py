# deploy/realtime_demo.py
"""
Real-time inference demo for keyword spotting and speaker verification
"""
import numpy as np
import pyaudio
import queue
import threading
import time
from pathlib import Path
import torch
import onnxruntime as ort
from collections import deque

from utils.audio_utils import AudioProcessor
from utils.spike_encoding import RateEncoder
from configs.config import cfg

class RealTimeInference:
    """Real-time audio inference system"""
    
    def __init__(self, kws_model_path: str, speaker_model_path: str = None):
        # Audio settings
        self.sample_rate = cfg.audio.sample_rate
        self.chunk_duration = 1.0  # 1 second chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.overlap = 0.5  # 50% overlap
        
        # Audio processor
        self.audio_processor = AudioProcessor(cfg)
        self.spike_encoder = RateEncoder(
            time_steps=cfg.spiking.time_steps,
            max_rate=cfg.spiking.max_spike_rate
        )
        
        # Load models
        self.kws_session = ort.InferenceSession(kws_model_path)
        self.speaker_session = None
        if speaker_model_path:
            self.speaker_session = ort.InferenceSession(speaker_model_path)
        
        # Audio queue
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=self.chunk_size)
        
        # Results
                # Results
        self.keywords = cfg.data.target_keywords[:6]
        self.last_keyword = None
        self.last_keyword_time = 0
        self.speaker_embedding = None
        
        # PyAudio setup
        self.pa = pyaudio.PyAudio()
        self.stream = None
        
        # Control flags
        self.running = False
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self):
        """Process audio chunks for inference"""
        while self.running:
            try:
                # Get audio chunk
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Add to buffer
                self.audio_buffer.extend(audio_chunk)
                
                # Process if we have enough samples
                if len(self.audio_buffer) >= self.chunk_size:
                    # Convert to numpy array
                    audio_data = np.array(list(self.audio_buffer))[:self.chunk_size]
                    
                    # Process audio
                    self._run_inference(audio_data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def _run_inference(self, audio_data: np.ndarray):
        """Run inference on audio chunk"""
        # Preprocess audio
        audio_norm = self.audio_processor.normalize_audio(audio_data)
        
        # Extract features
        mfcc = self.audio_processor.extract_mfcc(audio_norm)
        
        # Encode to spikes
        spikes = self.spike_encoder.encode(mfcc)
        
        # Prepare input
        input_data = spikes.reshape(1, 13, 100).astype(np.float32)
        
        # Keyword spotting
        kws_output = self.kws_session.run(
            None, 
            {self.kws_session.get_inputs()[0].name: input_data}
        )[0]
        
        # Get prediction
        probs = self._softmax(kws_output[0])
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        # Check if keyword detected with high confidence
        if confidence > 0.8:
            keyword = self.keywords[pred_idx]
            current_time = time.time()
            
            # Avoid duplicate detections
            if keyword != self.last_keyword or (current_time - self.last_keyword_time) > 2.0:
                self.last_keyword = keyword
                self.last_keyword_time = current_time
                
                print(f"\n🎯 Keyword detected: '{keyword}' (confidence: {confidence:.2f})")
                
                # If wake word detected, run speaker verification
                if self.speaker_session and keyword in ['yes', 'hey']:
                    self._verify_speaker(input_data)
    
    def _verify_speaker(self, input_data: np.ndarray):
        """Run speaker verification"""
        if not self.speaker_session:
            return
        
        # Get speaker embedding
        embedding = self.speaker_session.run(
            None,
            {self.speaker_session.get_inputs()[0].name: input_data}
        )[0]
        
        # If we have a stored embedding, compare
        if self.speaker_embedding is not None:
            similarity = self._cosine_similarity(embedding[0], self.speaker_embedding)
            print(f"🎤 Speaker similarity: {similarity:.2f}")
            
            if similarity > 0.7:
                print("✅ Speaker verified!")
            else:
                print("❌ Unknown speaker")
        else:
            # Store first embedding as reference
            self.speaker_embedding = embedding[0]
            print("🎤 Speaker embedding stored")
    
    def _softmax(self, x):
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _cosine_similarity(self, a, b):
        """Compute cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def start(self):
        """Start real-time inference"""
        print("🎙️ Starting real-time inference...")
        print(f"Keywords: {self.keywords}")
        print("Speak one of the keywords!\n")
        
        self.running = True
        
        # Start audio stream
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self.audio_callback
        )
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()
        
        self.stream.start_stream()
    
    def stop(self):
        """Stop real-time inference"""
        print("\n🛑 Stopping inference...")
        
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        
        self.pa.terminate()

def main():
    """Run real-time demo"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--kws_model', type=str, default='export/onnx/kws_model.onnx',
                       help='Path to keyword spotting ONNX model')
    parser.add_argument('--speaker_model', type=str, default=None,
                       help='Path to speaker verification ONNX model')
    
    args = parser.parse_args()
    
    # Check if models exist
    if not Path(args.kws_model).exists():
        print(f"❌ KWS model not found: {args.kws_model}")
        print("Please export a model first using export_onnx.py")
        return
    
    # Create inference system
    inference = RealTimeInference(
        kws_model_path=args.kws_model,
        speaker_model_path=args.speaker_model
    )
    
    try:
        # Start inference
        inference.start()
        
        # Run until user stops
        print("\nPress Ctrl+C to stop...")
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        pass
    finally:
        inference.stop()

if __name__ == "__main__":
    main()