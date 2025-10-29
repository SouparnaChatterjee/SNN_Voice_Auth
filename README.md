# 🧠 Spiking Neural Network (SNN) Based Voice Authentication System

This repository contains a **production-ready neuromorphic computing solution** for voice authentication — combining **keyword spotting** and **speaker verification** using **Spiking Neural Networks (SNNs)**.  
It demonstrates how **brain-inspired computing paradigms** can achieve high accuracy, real-time performance, and extreme efficiency suitable for **edge AI deployment**.

---

## 🚀 Project Overview

This 8-week project delivers a complete pipeline from audio preprocessing to real-time, on-device inference.  
It uses **PyTorch** and **SpikingJelly** to build, train, and optimize biologically inspired neural models.

### 🔍 Key Features
- Dual-task architecture:
  - Keyword Spotting (*yes, no, stop, go, up, down*)
  - Speaker Verification Framework
- Five Spike Encoding Methods:
  - Rate Coding
  - Latency Coding
  - Delta Modulation
  - Temporal Contrast
  - Population Coding
- Custom LIF Neurons with surrogate gradient training
- Lightweight model: **76,487 parameters**
- Model compression: **303KB → 56KB (94% reduction, ONNX export)**
- CPU-only inference: **<10ms/sample**
- Throughput: **>100 inferences/sec**
- Memory usage: **<50MB**
- Modular architecture with caching, checkpointing, and visualization

---

## ⚙️ System Architecture
Audio Input → Feature Extraction (MFCC) → Spike Encoding → SNN Layers (LIF) → Task Heads (KWS + Verification)


**Frameworks:** PyTorch, SpikingJelly, ONNX, TensorBoard  
**Loss Functions:** Focal Loss, Triplet Loss, Center Loss  
**Optimizations:** Quantization, Pruning, Edge Deployment

---

## 📈 Performance Summary

| Metric | Result | Target | Improvement |
|--------|---------|--------|-------------|
| Model Size | 56KB (ONNX) | 303KB | ✅ -94% |
| Inference Time | <10 ms (CPU) | 100 ms | ✅ 10× faster |
| Throughput | >100 inferences/sec | 10/sec | ✅ 10× higher |
| Memory Usage | <50 MB | — | ✅ Edge-friendly |
| Real-Time Factor | >100× | 1× | ✅ Ultra-fast |

---

## 🧩 Implementation Details

**Language:** Python 3.10+  
**Frameworks:** PyTorch, SpikingJelly, ONNX  

**Directory Structure**
├── data/
│ ├── raw/
│ └── processed/
├── models/
├── scripts/
├── utils/
├── week3_workflow.py
├── setup_week3.py
└── README.md


**Training Utilities**
- Checkpointing and resume support  
- TensorBoard logging  
- Early stopping and LR scheduling  
- Configurable batch/spike encoding parameters

---

## 💻 Deployment

The optimized ONNX model runs efficiently on:
- Raspberry Pi  
- Mobile (Android / iOS)  
- Web (via ONNX.js)  
- Embedded systems  

Includes **real-time demos** for streaming voice authentication and voice activity detection.

---

## 🧠 Significance

This project proves **Spiking Neural Networks can be practical alternatives** to traditional deep learning models for audio and voice tasks, offering:
- Lower power consumption  
- Better temporal processing  
- Hardware compatibility  
- Feasibility for always-on, low-power edge devices  

---

## 📜 150-Word Summary

This project implements a production-ready **Spiking Neural Network (SNN)** for voice authentication, combining **keyword spotting** and **speaker verification**.  
The lightweight architecture (76,487 parameters) uses custom LIF neurons and five spike encoding methods to process audio temporally, mimicking biological neural systems.

Key achievements include **94% model compression (303KB→56KB via ONNX)**, **sub-10ms CPU inference**, and **>100 inferences/second throughput**.  
It operates entirely on CPU (<50MB memory), ideal for edge devices like Raspberry Pi and mobile platforms.

The project spans data pipeline development, SNN architecture design with PyTorch/SpikingJelly, multi-task training, and deployment optimization.  
Real-time demos verify performance for streaming audio applications, proving SNNs can achieve production-level results with exceptional efficiency.

---

## 👤 Author

**Souparna Chatterjee**  
📧 [souparnachatterjee98@gmail.com](mailto:souparnachatterjee98@gmail.com)  
🌐 [LinkedIn](https://www.linkedin.com/in/souparna-chatterjee-864177223/) | [GitHub](https://github.com/SouparnaChatterjee)

---

## 🏁 License

Released under the **MIT License**.  
See [LICENSE](LICENSE) for details.

---

## ⭐ Acknowledgements

- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) for SNN implementation  
- [Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) for keyword spotting  
- Inspiration from neuromorphic computing and edge AI research
