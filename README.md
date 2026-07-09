# PCAM Tumor Detection

> A lightweight CNN for tumor detection designed for resource-constrained environments

## Overview

This project demonstrates a **cost-effective, resource-efficient Convolutional Neural Network (CNN)** for detecting tumors in medical images. Built specifically for environments with limited computational resources, this model achieves **~81% accuracy** while maintaining a small footprint and fast inference times.

## Key Features

- **Lightweight Architecture**: Optimized for devices with limited GPU/CPU resources
- **Budget-Friendly**: Designed to work with minimal computational power
- **High Performance**: Achieves 81%+ accuracy with ROC AUC scores above 0.89
- **Complete Pipeline**: Includes training, validation, testing, and comprehensive evaluation metrics

## Model Architecture

The CNN model (`tumordetection`) features:
- 3 Convolutional layers with batch normalization
- Dropout regularization (30%) to prevent overfitting
- ReLU activation for non-linearity
- Adaptive average pooling for dimension reduction
- Final sigmoid activation for binary classification

## Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| Accuracy | 81.1% |
| F1 Score | 0.82 |
| Precision | 0.85 |
| Recall | 0.79 |
| ROC AUC | 0.89 |
| Average Loss | 0.462 |

### Validation Set Performance

| Metric | Score |
|--------|-------|
| Accuracy | 82.6% |
| F1 Score | 0.83 |
| Precision | 0.88 |
| Recall | 0.79 |
| ROC AUC | 0.91 |
| Average Loss | 0.427 |

### Confusion Matrix Breakdown (Test Set)

- **True Positives**: 13,916 (correctly identified tumors)
- **True Negatives**: 12,666 (correctly identified non-tumors)
- **False Positives**: 3,725 (incorrectly flagged as tumors)
- **False Negatives**: 2,461 (missed tumor cases)

## Why this matters 
- Small and efficient to train without requirment of expensive GPU
- Provides a foundation for affordable medical ai <img width="1073" height="216" alt="WhatsApp Image 2026-07-09 at 10 10 10 PM" src="https://github.com/user-attachments/assets/61af42c5-c8c0-4b50-94c8-60d77726ce92" />
<img width="1006" height="279" alt="WhatsApp Image 2026-07-09 at 10 09 35 PM" src="https://github.com/user-attachments/assets/beaced2f-183b-4821-9a8e-072df334cd70" />
<img width="1347" height="745" alt="WhatsApp Image 2026-07-09 at 10 09 10 PM" src="https://github.com/user-attachments/assets/67c5b363-a1fe-4d3f-9cc0-e9a18f527a57" />
<img width="1336" height="752" alt="WhatsApp Image 2026-07-09 at 10 10 46 PM" src="https://github.com/user-attachments/assets/1edc0d41-fe4f-411d-887c-1339fac795a2" />
