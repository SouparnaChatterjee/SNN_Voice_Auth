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

