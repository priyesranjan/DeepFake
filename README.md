# Deepfake Image Detection using CNN and Transformer Architectures

This repository presents the implementation of multiple independent deep learning models for the detection of deepfake images. The project is part of an internship at the Indian Institute of Information Technology (IIIT) Bhagalpur, conducted under the guidance of Dr. Om Prakash Singh.

## 📄 Project Overview

Deepfake technology has evolved rapidly, making it increasingly difficult to distinguish between real and manipulated images. This project aims to develop and evaluate various deep learning-based detection models to accurately identify deepfake images while balancing computational efficiency.

## 🔍 Models Implemented

The following models were implemented and evaluated individually:

* **EfficientNet B0, B1, B3, B5** (CNN-based)
* **DenseNet121** (CNN-based)
* **MobileNetV2** (Lightweight CNN)
* **Vision Transformer (ViT)** (Transformer-based)

## 🗂 Dataset Used

* **FaceForensics++** (2 frames per video extracted for real and fake images)
* **NVIDIA Deepfake Dataset**
* **My data set link** 
https://www.kaggle.com/datasets/priyesranjan/deepfake-detection-split-dataset

All images were preprocessed to a size of **224x224** pixels with data augmentation techniques such as flipping, rotation, scaling, and brightness adjustment.

## ⚙️ Methodology

1. **Preprocessing:**

   * Face detection and cropping
   * Image resizing and augmentation

2. **Model Training:**

   * Independent training of each model using Binary Cross-Entropy loss
   * Evaluation using accuracy, precision, recall, F1-score, and AUC

3. **Analysis:**

   * Performance compared across models
   * Trade-off analysis between accuracy and computational efficiency

## 📊 Key Findings

* **DenseNet121** achieved the highest accuracy of **99.74%** with strong robustness.
* **EfficientNet B3** emerged as the most stable model balancing accuracy and computational cost.
* **Vision Transformers (ViT)** underperformed with smaller datasets, reinforcing the need for larger datasets for transformers.
* Parameter tuning was essential for model stability.

## 🚀 Proposed New Approach (HMS-DDS)

A new **Hybrid Multi-Stage Deepfake Detection System (HMS-DDS)** is proposed combining:

1. Lightweight CNNs (EfficientNet B0 or MobileNetV2)
2. Global semantic analysis via Transformer blocks
3. Anomaly detection using Diffusion or Frequency domain analysis

✅ Features:

* Adaptive model selection
* Cross-scale feature fusion
* Explainable AI integration

## 💻 How to Run

1. Clone the repository:

```bash
https://github.com/priyesranjan/deepfake
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Train models:

```bash
python train_model.py --model EfficientNetB3
```

4. Evaluate models:

```bash
python evaluate_model.py --model EfficientNetB3
```

## 📈 Results Summary

| Model              | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC   |
| ------------------ | ------------ | ------------- | ---------- | ------------ | ----- |
| EfficientNet B0    | 89.00        | 88.00         | 92.00      | 87.50        | 0.91  |
| EfficientNet B3    | 93.20        | 92.00         | 95.00      | 95.00        | 0.994 |
| DenseNet121        | 99.74        | 92.00         | 92.50      | 92.30        | 0.99  |
| MobileNetV2        | 88.50        | 87.50         | 88.00      | 87.70        | 0.90  |
| Vision Transformer | 85.00        | 83.00         | 84.00      | 83.50        | 0.87  |

## 📌 Practical Applications

* Digital media authentication
* Social media moderation
* Law enforcement forensics
* Identity verification systems

## 📚 References

* FaceForensics++ Dataset
* NVIDIA DeepFake Dataset
* Research papers on EfficientNet, DenseNet, MobileNet, ViT

## 📧 Contact

**Developer:** Priyranjan Kumar
**Email:** [priyesranjan@gmail.com](priyesranjan@gmail.com)
**College:** Government Engineering College, Banka
