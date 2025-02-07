# Brain Tumor Detection using MRI Scans

## Overview
The **Brain Tumor Detection** project applies machine learning and deep learning techniques to detect brain tumors from MRI scans. The goal is to develop a reliable and automated system for early diagnosis.

## Features
- Processes MRI scan images for tumor detection
- Implements deep learning models for classification
- Evaluates model performance using various metrics
- Supports real-time image analysis

## Installation
Clone the repository using:
```bash
git clone https://github.com/omar0930/Brain-Tumor-Detection-using-MRI-Scans.git
cd Brain-Tumor-Detection-using-MRI-Scans
```


## Dataset (https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
The dataset consists of MRI scans labeled as either tumorous or non-tumorous. The images are preprocessed using normalization, augmentation, and feature extraction techniques before being fed into the model.

## Workflow
1. Load and preprocess the MRI scan dataset.
2. Apply data augmentation and normalization.
3. Train deep learning models (e.g., CNN, ResNet, VGG16) on the dataset.
4. Evaluate model performance using accuracy, precision, recall, and F1-score.
5. Deploy the trained model for real-time tumor detection.

## Results
The models achieved the following classification performance:
- **Convolutional Neural Network (CNN):** 91.8% accuracy
- **ResNet-50:** 94.3% accuracy
- **VGG16:** 95.6% accuracy

These results demonstrate that VGG16 performed the best in detecting brain tumors from MRI scans. Further improvements can be achieved by using larger datasets and fine-tuning hyperparameters.

## Technologies Used
- Python
- NumPy & Pandas
- OpenCV (for image processing)
- TensorFlow/Keras (for deep learning models)
- Scikit-learn
