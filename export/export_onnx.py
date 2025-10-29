# export/export_onnx.py
"""
Export model to ONNX format for cross-platform deployment
"""
import torch
import torch.onnx
import numpy as np
from pathlib import Path
import onnx
import onnxruntime as ort

from models.dual_task_snn import DualTaskSNN
from configs.config import cfg

class ONNXExporter:
    """Export SNN models to ONNX format"""
    
    def __init__(self, model_path: str, quantized: bool = False):
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
        self.quantized = quantized
    
    def export_kws_model(self, output_path: str):
        """Export keyword spotting model only"""
        print("📦 Exporting KWS model to ONNX...")
        
        # Create wrapper for KWS-only inference
        class KWSWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                outputs = self.model(x, task='kws')
                return outputs['kws']['logits']
        
        kws_model = KWSWrapper(self.model)
        kws_model.eval()
        
        # Example input
        dummy_input = torch.randn(1, 13, 100)
        
        # Export
        torch.onnx.export(
            kws_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['audio_features'],
            output_names=['keyword_logits'],
            dynamic_axes={
                'audio_features': {0: 'batch_size'},
                'keyword_logits': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX model saved to {output_path}")
        
        # Verify ONNX model
        self._verify_onnx_model(output_path, dummy_input)
        
    def export_speaker_model(self, output_path: str):
        """Export speaker verification model only"""
        print("📦 Exporting Speaker model to ONNX...")
        
        # Create wrapper for speaker embedding extraction
        class SpeakerWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                outputs = self.model(x, task='speaker')
                return outputs['speaker']['embeddings']
        
        speaker_model = SpeakerWrapper(self.model)
        speaker_model.eval()
        
        # Example input
        dummy_input = torch.randn(1, 13, 100)
        
        # Export
        torch.onnx.export(
            speaker_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['audio_features'],
            output_names=['speaker_embeddings'],
            dynamic_axes={
                'audio_features': {0: 'batch_size'},
                'speaker_embeddings': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX model saved to {output_path}")
        
    def _verify_onnx_model(self, onnx_path: str, test_input: torch.Tensor):
        """Verify ONNX model correctness"""
        print("🔍 Verifying ONNX model...")
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path)
        
        # Prepare input
        ort_inputs = {
            ort_session.get_inputs()[0].name: test_input.numpy()
        }
        
        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Compare with PyTorch output
        with torch.no_grad():
            if 'keyword' in onnx_path:
                pytorch_output = self.model(test_input, task='kws')['kws']['logits']
            else:
                pytorch_output = self.model(test_input, task='speaker')['speaker']['embeddings']
        
        # Check difference
        max_diff = np.max(np.abs(ort_outputs[0] - pytorch_output.numpy()))
        print(f"✅ Max difference: {max_diff:.6f}")
        
        if max_diff > 1e-5:
            print("⚠️  Warning: Large difference detected!")
        
        return max_diff < 1e-3
    
    def optimize_onnx_model(self, onnx_path: str):
        """Optimize ONNX model for inference"""
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        print("⚙️  Optimizing ONNX model...")
        
        # Quantize ONNX model
        quantized_path = onnx_path.replace('.onnx', '_quantized.onnx')
        
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QUInt8
        )
        
        print(f"✅ Quantized ONNX model saved to {quantized_path}")
        
        # Compare file sizes
        orig_size = Path(onnx_path).stat().st_size / 1024 / 1024
        quant_size = Path(quantized_path).stat().st_size / 1024 / 1024
        
        print(f"Original size: {orig_size:.2f} MB")
        print(f"Quantized size: {quant_size:.2f} MB")
        print(f"Compression ratio: {orig_size/quant_size:.2f}x")

def export_to_onnx(model_path: str, output_dir: str = 'export/onnx'):
    """Main ONNX export function"""
    print(f"\n🚀 Exporting model to ONNX: {model_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize exporter
    exporter = ONNXExporter(model_path)
    
    # Export both models
    kws_path = output_dir / "kws_model.onnx"
    speaker_path = output_dir / "speaker_model.onnx"
    
    exporter.export_kws_model(str(kws_path))
    exporter.export_speaker_model(str(speaker_path))
    
    # Optimize models
    exporter.optimize_onnx_model(str(kws_path))
    
    print(f"\n✅ ONNX export complete! Models saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='export/onnx')
    
    args = parser.parse_args()
    export_to_onnx(args.model, args.output_dir)