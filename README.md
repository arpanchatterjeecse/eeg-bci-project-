# Electroencephalogram-Based Seizure Detection & Prediction using BCI  

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-GRU%20%7C%20Attention-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Optimization-Bayesian-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Application-Healthcare-green?style=for-the-badge" />
</p>  

---

## 📌 Overview  
This project implements a **Brain–Computer Interface (BCI)** model for **seizure detection and prediction** from EEG signals. Leveraging a **Bidirectional GRU with Temporal Attention**, the architecture captures both sequential dependencies and salient temporal features in brainwave patterns.  

The goal is to provide a **robust, accurate, and explainable AI framework** for healthcare applications, particularly in **epileptic seizure monitoring and prediction**.  

---

## 🏗️ Model Architecture  
- **Input**: EEG sequences (178 time steps, 1 channel)  
- **Layers**:  
  - Bidirectional GRU (2 stacked layers)  
  - Dropout + Batch Normalization  
  - Temporal Attention Mechanism (focus on critical EEG segments)  
  - Dense (ReLU) + Dropout  
  - Sigmoid Output for Binary Classification  

```python
# Core Highlights
- Bidirectional GRU layers for sequential EEG learning  
- Temporal Attention Mechanism for interpretability  
- Bayesian Optimization for hyperparameter tuning  
- Early Stopping to prevent overfitting  
