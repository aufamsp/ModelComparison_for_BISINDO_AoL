# Model Comparison for BISINDO Hand Gesture Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains code for comparing two deep learning models for Indonesian Sign Language (BISINDO) hand gesture recognition. The project implements and evaluates Custom CNN and MobileNetV2 architectures on the BISINDO v3 dataset from Roboflow.

## Dataset

The BISINDO v3 dataset contains images of hand gestures representing letters A-Z. Key characteristics:
- **Classes**: 26 (A-Z)
- **Total Images**: Varied per class (severe class imbalance)
- **Format**: Roboflow export with bounding box annotations

### Class Distribution Issues
The dataset exhibits significant class imbalance:
- Most frequent class (R): 117 images
- Least frequent class (C): 11 images
- Imbalance ratio: ~10.6x

This imbalance significantly impacts model performance, particularly for the Custom CNN architecture.

## Model Architectures

### 1. Custom CNN
A custom convolutional neural network designed specifically for hand gesture recognition:
- Multiple convolutional layers with batch normalization
- Max pooling for spatial dimension reduction
- Fully connected layers for classification
- Dropout for regularization

### 2. MobileNetV2
A pre-trained lightweight CNN architecture adapted for BISINDO classification:
- Uses ImageNet weights as starting point
- Global average pooling replaces original classifier
- New fully connected head for 26-class classification
- Fine-tuning approach with selective layer unfreezing

## Key Findings

### Performance Comparison
- **MobileNetV2**: Achieved ~99.85% accuracy
- **Custom CNN**: Achieved ~85% accuracy (without proper imbalance handling)

### Critical Insight
The primary reason for the Custom CNN's poor performance was the severe class imbalance in the training data. When implemented with proper techniques:
- WeightedRandomSampler for balanced batch sampling
- Extensive data augmentation tailored for hand gestures
- The Custom CNN performance improved significantly

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
   - Download BISINDO v3 dataset from Roboflow
   - Organize in the expected folder structure:
     ```
     bisindo_v3/
     ├── train/
     ├── val/
     └── test/
     ```

2. **Run the Notebook**:
   ```bash
   jupyter notebook ModelComparison_for_BISINDO.ipynb
   ```

3. **Follow the Notebook Sections**:
   - Environment Setup
   - Mount Google Drive & Set Dataset Path (for Colab)
   - Data Cleaning
   - Class Imbalance Analysis & Handling
   - Data Augmentation
   - Model Implementation & Training
   - Results & Comparison

## Data Augmentation Strategies

For hand gesture recognition, specific augmentation rules were applied:
- **Color jitter**: Simulates lighting variations
- **Random crop**: Accounts for varying hand positions
- **Small rotation (±15°)**: Represents slight hand tilts
- **Horizontal flip (50% probability)**: Some letters are direction-specific
- **Vertical flip (0%)**: Hands are never upside down in practice

## Results

With proper imbalance handling and augmentation:
- Both models achieved high accuracy (>99%)
- MobileNetV2 showed slightly better performance and faster convergence
- Custom CNN became competitive when trained with balanced sampling

## Files

- `ModelComparison_for_BISINDO.ipynb`: Main Jupyter notebook with complete implementation
- `README.md`: This file

## Requirements

- Python 3.8+
- PyTorch 1.7+ (with CUDA support recommended)
- torchvision
- matplotlib
- seaborn
- scikit-learn
- pandas
- numpy
- Pillow
- tqdm

## Acknowledgments

- BISINDO dataset providers via Roboflow
- Open-source deep learning community
- Pretrained model architectures (MobileNetV2)

## License

This project is licensed under the MIT License - see the LICENSE file for details.