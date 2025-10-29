# export/quantize_model.py
"""
Quantize trained model for efficient deployment
"""
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize_static
import numpy as np
from pathlib import Path
import time
from typing import Dict, Tuple

from models.dual_task_snn import DualTaskSNN
from configs.config import cfg

class ModelQuantizer:
    """Quantize SNN models for efficient deployment"""
    
    def __init__(self, model_path: str):
        # Load model
        self.device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        self.model = DualTaskSNN(
            input_dim=cfg.audio.n_mfcc,
            time_steps=cfg.spiking.time_steps,
            num_keywords=6,
            backbone_type='lightweight'
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.model_path = Path(model_path)
        
    def dynamic_quantization(self) -> nn.Module:
        """Apply dynamic quantization (INT8)"""
        print("🔧 Applying dynamic quantization...")
        
        # Quantize model
        quantized_model = quantize_dynamic(
            self.model,
            qconfig_spec={
                nn.Linear: torch.quantization.default_dynamic_qconfig,
                nn.Conv2d: torch.quantization.default_dynamic_qconfig,
            },
            dtype=torch.qint8
        )
        
        print("✅ Dynamic quantization complete!")
        return quantized_model
    
    def export_quantized_model(self, quantized_model: nn.Module, output_path: str):
        """Save quantized model"""
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'quantization_type': 'dynamic_int8',
            'original_model_path': str(self.model_path)
        }, output_path)
        print(f"💾 Quantized model saved to {output_path}")
    
    def benchmark_models(self, test_input: torch.Tensor) -> Dict:
        """Compare original vs quantized model"""
        print("\n📊 Benchmarking models...")
        
        # Original model
        start_time = time.time()
        with torch.no_grad():
            orig_output = self.model(test_input, task='kws')
        orig_time = (time.time() - start_time) * 1000
        
        # Quantized model
        quantized_model = self.dynamic_quantization()
        start_time = time.time()
        with torch.no_grad():
            quant_output = quantized_model(test_input, task='kws')
        quant_time = (time.time() - start_time) * 1000
        
        # Compare outputs
        orig_probs = orig_output['kws']['probabilities']
        quant_probs = quant_output['kws']['probabilities']
        max_diff = torch.max(torch.abs(orig_probs - quant_probs)).item()
        
        # Model sizes
        def get_model_size_mb(model):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            return (param_size + buffer_size) / 1024 / 1024
        
        orig_size = get_model_size_mb(self.model)
        quant_size = get_model_size_mb(quantized_model)
        
        results = {
            'original_inference_ms': orig_time,
            'quantized_inference_ms': quant_time,
            'speedup': orig_time / quant_time,
            'original_size_mb': orig_size,
            'quantized_size_mb': quant_size,
            'compression_ratio': orig_size / quant_size,
            'max_output_diff': max_diff
        }
        
        print("\n📈 Quantization Results:")
        print(f"Inference speedup: {results['speedup']:.2f}x")
        print(f"Model compression: {results['compression_ratio']:.2f}x")
        print(f"Max output difference: {results['max_output_diff']:.6f}")
        
        return results

def quantize_model(model_path: str, output_dir: str = 'export/quantized'):
    """Main quantization function"""
    print(f"\n🚀 Quantizing model: {model_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize quantizer
    quantizer = ModelQuantizer(model_path)
    
    # Test input
    test_input = torch.randn(1, 13, 100)
    
    # Quantize and benchmark
    quantized_model = quantizer.dynamic_quantization()
    results = quantizer.benchmark_models(test_input)
    
    # Save quantized model
    output_path = output_dir / f"quantized_{Path(model_path).stem}.pt"
    quantizer.export_quantized_model(quantized_model, str(output_path))
    
    # Save benchmark results
    import json
    results_path = output_dir / f"quantization_results_{Path(model_path).stem}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Quantization complete! Files saved to {output_dir}")
    
    return quantized_model, results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='export/quantized')
    
    args = parser.parse_args()
    quantize_model(args.model, args.output_dir)