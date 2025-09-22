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

# RNN-based EEG Classifier (Model- 1)

For my first model, I implemented a **Recurrent Neural Network (RNN)** with **Bidirectional LSTM layers** to capture sequential dependencies in EEG signals. The model uses regularization and hyperparameter tuning for optimal performance.

## Model Architecture

- **Input Layer:** 178 timesteps × 1 feature  
- **Bidirectional LSTM Layer 1:** captures forward and backward temporal dependencies  
- **Dropout Layer 1:** prevents overfitting  
- **Batch Normalization 1:** stabilizes and speeds up training  
- **Bidirectional LSTM Layer 2:** extracts higher-level temporal features  
- **Dropout Layer 2**  
- **Batch Normalization 2**  
- **Dense Layer:** transforms features for classification  
- **Dropout Layer 3**  
- **Output Layer:** single neuron with sigmoid activation for binary classification  

## Performance Metrics

| Dataset      | Accuracy |
|-------------|---------|
| Training    | 0.9877  |
| Validation  | 0.9813  |
| Test        | 0.9800  |

**Highlights:**

- Bidirectional LSTM layers capture temporal patterns in EEG signals.  
- Dropout and Batch Normalization improve generalization and stability.  
- Hyperparameter tuning optimizes layer sizes, dropout rates, and learning rate.  
- High generalization, as shown by consistent validation and test accuracy.  

---

This model serves as the **baseline sequential model** before exploring more advanced architectures like GRU and attention mechanisms.


## 🏗️ Final Model Architecture  (Model -2)
- **Input**: EEG sequences (178 time steps, 1 channel)  
- **Layers**:  
  - Bidirectional GRU (2 stacked layers)  
  - Dropout + Batch Normalization  
  - Temporal Attention Mechanism (focus on critical EEG segments)  
  - Dense (ReLU) + Dropout  
  - Sigmoid Output for Binary Classification  

```python
Core Highlights
- Bidirectional GRU layers for sequential EEG learning
- Temporal Attention Mechanism for interpretability
- Bayesian Optimization for hyperparameter tuning
- Early Stopping to prevent overfitting

These components collectively enhance the model's ability to accurately
capture temporal dependencies in EEG signals while maintaining
generalization and interpretability.

Model Performace:
Training Accuracy:   0.9992
Validation Accuracy: 0.9980
Test Accuracy:       0.9963
```



## 🔄 Data Augmentation Pipeline

The project includes a comprehensive **data augmentation pipeline** designed specifically for EEG time series data to enhance model performance and robustness. The augmentation methods preserve the temporal and frequency characteristics of EEG signals while increasing dataset diversity.

### Augmentation Techniques

1. **Jitter (Noise Addition)**: Adds Gaussian noise to simulate electrode artifacts and environmental interference
2. **Scaling**: Applies random amplitude scaling to account for inter-subject variability
3. **Time Warping**: Non-linear temporal distortion using cubic spline interpolation to simulate natural EEG variations




