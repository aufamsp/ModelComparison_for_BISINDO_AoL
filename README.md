# Model Comparison for BISINDO Hand Gesture Recognition

## Overview
 
This repository contains code for comparing two deep learning models for Indonesian Sign Language (BISINDO) hand gesture recognition. The project implements and evaluates a Custom CNN and MobileNetV2 on the BISINDO v3 dataset from Roboflow, including full data cleaning, class imbalance handling, hyperparameter tuning, and augmentation pipelines.
 
## Dataset
 
The BISINDO v3 dataset contains real-world hand gesture images representing the 26-letter alphabet (A–Z), exported from Roboflow with bounding box annotations.
 
| Split | Images | Ratio |
|-------|--------|-------|
| Train | 6,241  | 70.2% |
| Val   | 1,335  | 15.0% |
| Test  | 1,315  | 14.8% |
| **Total** | **8,891** | — |
 
Zero cross-split leakage was verified (train ∩ val = 0, train ∩ test = 0, val ∩ test = 0). All 26 classes are present in every split.
 
### Class Distribution & Imbalance
 
The dataset has significant class imbalance:
- **Most frequent class** (R): 574 training images
- **Least frequent class** (C): 61 training images
- **Imbalance ratio**: ~9.4×
 
A label case normalisation was the only cleaning step required — Roboflow exported lowercase `z` labels, which were uppercased to `Z` across all splits.
 
## Model Architectures
 
### 1. Custom CNN
 
A four-block convolutional network built from scratch for hand gesture recognition:
- 4× ConvBlock (Conv2d → BatchNorm → ReLU → MaxPool), channel depth: 3 → 32 → 64 → 128 → 256
- Input resolution: 224×224
- AdaptiveAvgPool2d → Dropout(0.3) → Linear(256→128) → Linear(128→26)
- **Total parameters: 533,818**
 
### 2. MobileNetV2
 
A standard MobileNetV2 implementation built from scratch using Inverted Residual Blocks (IRBs):
- Each IRB: 1×1 expansion → depthwise 3×3 conv → 1×1 projection (linear bottleneck)
- Residual skip connections where stride=1 and channels match
- Final classifier: Dropout(0.3) → Linear(1280→26)
- **Total parameters: 2,257,178**
- **Note:** trained from random initialisation, not ImageNet pretrained weights
 
## Hyperparameter Tuning
 
A grid search over 12 configurations (3 learning rates × 2 batch sizes × 2 dropout values) was conducted using 5-epoch proxy runs before full training:
 
| LR     | Batch | Dropout | Val Acc (5 ep) |
|--------|-------|---------|----------------|
| 0.01   | 32    | 0.3     | 24.19%         |
| 0.01   | 64    | 0.3     | 31.46%         |
| 0.001  | 32    | 0.3     | 93.93%         |
| 0.001  | 64    | 0.5     | 95.13%         |
| 0.0005 | 32    | 0.3     | **96.25% ✓**   |
 
**Best config:** `lr=0.0005`, `batch_size=32`, `dropout=0.3` — applied identically to both models.
 
## Key Results
 
| Metric | Custom CNN | MobileNetV2 |
|--------|-----------|-------------|
| Test Accuracy | 53.84% | **99.92%** |
| Best Val Accuracy | 54.38% | **100.00%** |
| Precision (weighted) | 55.29% | **99.93%** |
| Recall (weighted) | 53.84% | **99.92%** |
| F1-Score (weighted) | 50.93% | **99.92%** |
| Parameters | 533,818 | 2,257,178 |
| Training Time | ~23.8 min | ~25.4 min |
| Convergence (val > 98%) | Never reached | Epoch 7 |
| Lowest per-class F1 | 0.0% (R, X) | 98.7% (H) |
 
Training was run for 30 epochs on GPU (CUDA, PyTorch 2.10.0+cu128).
 
### Key Insight
 
The performance gap is primarily **architectural**, not a data issue. MobileNetV2's Inverted Residual Blocks provide deeper feature hierarchies, depthwise separable convolutions, and residual skip connections that allow it to learn discriminative representations across all 26 classes. The Custom CNN's four-block shallow architecture lacks sufficient representational capacity for the inter-class similarity present in real-world BISINDO gesture images — even with class imbalance correction fully applied to both models.
 
## Installation
 
```bash
# Clone the repository
git clone https://github.com/yourusername/ModelComparison_for_BISINDO_AoL.git
cd ModelComparison_for_BISINDO_AoL
 
# Install dependencies
pip install torch torchvision matplotlib seaborn scikit-learn pandas numpy pillow tqdm
```
 
## Usage
 
1. **Prepare Dataset**:
   - Download BISINDO v3 dataset from Roboflow (Roboflow export format)
   - Organise in the expected folder structure:
     ```
     dataset/
     ├── train/
     │   └── _annotations.csv
     ├── valid/
     │   └── _annotations.csv
     └── test/
         └── _annotations.csv
     ```
 
2. **Mount Google Drive** (if using Colab) and update the dataset path in the notebook.
 
3. **Run the Notebook**:
   ```bash
   jupyter notebook ModelComparison_for_BISINDO.ipynb
   ```
 
4. **Follow the Notebook Sections**:
   - Environment Setup
   - Data Cleaning (label normalisation, null checks, split verification)
   - Class Imbalance Analysis & Handling
   - Data Augmentation
   - Hyperparameter Grid Search
   - Model Implementation & Training (30 epochs each)
   - Evaluation: Confusion Matrics, Per-class F1, Final Summary
 
## Class Imbalance Handling
 
Two complementary strategies are applied simultaneously:
 
- **WeightedRandomSampler**: oversamples rare classes during mini-batch construction so each epoch sees a balanced class distribution regardless of raw frequency.
- **Weighted CrossEntropyLoss**: assigns higher loss weight to rare classes using sklearn's balanced formula (`n_samples / (n_classes × count_per_class)`), penalising misclassification of underrepresented gestures more heavily.
 
## Data Augmentation
 
The following 8-step augmentation pipeline is applied to the **training set only**. Validation and test sets receive only resize + normalise.
 
| Step | Transform | Purpose |
|------|-----------|---------|
| 1 | Resize to 248×248 | Oversized for cropping |
| 2 | RandomCrop to 224×224 | Translation invariance |
| 3 | RandomHorizontalFlip (p=0.3) | Partial gesture symmetry |
| 4 | RandomRotation ±15° | Wrist tilt variation |
| 5 | ColorJitter (brightness ±30%, contrast ±30%, saturation ±20%, hue ±5%) | Lighting variation |
| 6 | RandomPerspective | Minor 3D angle shift |
| 7 | ToTensor | Convert to float tensor |
| 8 | Normalize (ImageNet stats) | Zero-centre input distribution |
 
## Training Configuration
 
| Component | Value |
|-----------|-------|
| Hardware | GPU (NVIDIA CUDA) |
| Framework | PyTorch 2.10.0+cu128 |
| Epochs | 30 |
| Optimizer | Adam (lr=0.0005, weight_decay=1e-4) |
| LR Scheduler | StepLR (step_size=10, gamma=0.1) |
| Loss | CrossEntropyLoss with class weights |
| Batch Sampler | WeightedRandomSampler |
| Random Seed | 42 |
 
## Files
 
- `ModelComparison_for_BISINDO.ipynb`: Main Jupyter notebook with complete implementation and training logs
- `README.md`: This file
- `requirements.txt`: Library needed to run the notebook
 
## Requirements
 
- Python 3.8+
- PyTorch 2.x (CUDA support strongly recommended — CPU training will be very slow)
- torchvision
- matplotlib
- seaborn
- scikit-learn
- pandas
- numpy
- Pillow
- tqdm
 
## Acknowledgments
 
- BISINDO v3 dataset via Roboflow
- Pretrained MobileNetV2 architecture
