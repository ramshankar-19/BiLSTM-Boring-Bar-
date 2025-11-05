# BiLSTM Boring Bar Vibration State Monitoring

Intelligent monitoring system for boring bar vibration classification using Bidirectional LSTM neural networks. Based on research in deep-hole boring operations.

## 🎯 Project Overview

This project implements a BiLSTM (Bidirectional Long Short-Term Memory) network to classify boring bar vibration states into three categories:
- **Stable Cutting** (Class 0)
- **Transition State** (Class 1)  
- **Violent Vibration** (Class 2)

### Key Features

✅ **Time-series analysis** using BiLSTM architecture  
✅ **Multi-sensor fusion** (3-axis accelerometer + sound pressure)  
✅ **Wavelet denoising** (coif5 basis, 3-layer decomposition)  
✅ **Feature extraction** (time-domain + frequency-domain)  
✅ **Experiment-level splitting** (prevents data leakage)  
✅ **Real-time monitoring** capability

## 📊 Model Architecture

Input (timesteps × 56 features)
↓
Bidirectional LSTM (128 units)
↓
Batch Normalization + Dropout (0.3)
↓
Bidirectional LSTM (64 units)
↓
Batch Normalization + Dropout (0.3)
↓
Dense (128 units, ReLU)
↓
Output (3 classes, Softmax)


## 🔬 Technical Details

### Signal Processing Pipeline

1. **Wavelet Denoising**
   - Basis: coif5
   - Decomposition: 3 levels
   - Thresholding: Hard threshold

2. **Feature Extraction** (per sensor, per window)
   - **Time-domain** (8 features): Mean, STD, RMS, Peak-to-peak, Peak, Kurtosis, Skewness, Crest factor
   - **Frequency-domain** (6 features): Dominant frequency, Spectral energy, Spectral centroid, Band energies (0-1kHz, 1-5kHz, 5-10kHz)

3. **Total Features**: 4 sensors × 14 features = **56 features per timestep**

### Model Training

- **Optimizer**: SGD with momentum (0.9)
- **Learning rate**: 0.1 (reduced by 0.1 every 100 epochs)
- **Batch size**: 64
- **Max epochs**: 1000
- **Early stopping**: Patience of 50 epochs
- **Loss function**: Sparse categorical crossentropy

## 📈 Results

### Performance Metrics (Synthetic Data)

| Class | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| Stable | 1.0000 | 1.0000 | 1.0000 | 100% |
| Transition | 1.0000 | 1.0000 | 1.0000 | 100% |
| Violent | 1.0000 | 1.0000 | 1.0000 | 100% |
| **Overall** | **1.0000** | **1.0000** | **1.0000** | **100%** |

*Note: 100% accuracy on synthetic data. Real sensor data expected: 70-85% (excellent for production use)*



