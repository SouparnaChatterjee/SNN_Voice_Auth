# export_onnx_simple.py
"""Simple ONNX export for Week 3"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

def create_simple_model():
    """Create a simple model that mimics SNN output"""
    
    class SimpleKWSModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(13, 64),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.temporal = nn.LSTM(64, 32, batch_first=True)
            self.classifier = nn.Linear(32, 6)
        
        def forward(self, x):
            # x shape: [batch, features, time]
            batch_size, n_features, time_steps = x.shape
            
            # Process each time step
            x = x.permute(0, 2, 1)  # [batch, time, features]
            encoded = self.encoder(x)  # [batch, time, 64]
            
            # LSTM processing
            lstm_out, _ = self.temporal(encoded)
            
            # Take last timestep
            final_features = lstm_out[:, -1, :]  # [batch, 32]
            
            # Classification
            logits = self.classifier(final_features)
            
            return logits
    
    return SimpleKWSModel()

def export_to_onnx():
    """Export model to ONNX"""
    print("[START] Exporting to ONNX...")
    
    # Create model
    model = create_simple_model()
    model.eval()
    
    # Example input
    dummy_input = torch.randn(1, 13, 100)
    
    # Create export directory
    os.makedirs("export/onnx", exist_ok=True)
    output_path = "export/onnx/kws_model.onnx"
    
    # Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        input_names=['audio_features'],
        output_names=['keyword_logits'],
        dynamic_axes={
            'audio_features': {0: 'batch_size'},
            'keyword_logits': {0: 'batch_size'}
        }
    )
    
    print(f"[OK] Model exported to {output_path}")
    
    # Verify
    try:
        import onnx
        import onnxruntime as ort
        
        # Check model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference
        ort_session = ort.InferenceSession(output_path)
        outputs = ort_session.run(None, {'audio_features': dummy_input.numpy()})
        
        print(f"[OK] ONNX model verified! Output shape: {outputs[0].shape}")
        
    except Exception as e:
        print(f"[WARNING]  Verification failed: {e}")
    
    return output_path

if __name__ == "__main__":
    onnx_path = export_to_onnx()
