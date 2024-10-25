# COS30082 - Bird Species Classification

This project focuses on developing a machine learning model for multi-class classification of bird species using the Caltech-UCSD Birds 200 (CUB-200) dataset. The dataset consists of images from 200 bird species, predominantly found in North America.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
The goal of this project is to classify bird species based on images from the CUB-200 dataset. This dataset contains subtle visual differences between species, making it a challenging classification task.

The model was trained and evaluated using various deep learning architectures, and preprocessing techniques were employed to improve accuracy and reduce overfitting.

## Dataset
The CUB-200 dataset contains:
- **Training Images**: 4,829 images
- **Testing Images**: 1,204 images
- **Total Classes**: 200 bird species

Each image is labeled with a specific bird species, and the dataset presents challenges such as:
- **Class Imbalance**: Unequal distribution of images across species.
- **Image Variability**: Differences in image resolution and quality.
- **Complex Backgrounds and Occlusions**: Birds are often partially hidden or surrounded by noisy backgrounds.

## Preprocessing
To address dataset challenges, the following preprocessing steps were applied:
- **Image Resizing**: All images were resized to 224x224 pixels for uniformity.
- **Normalization**: Images were normalized using ImageNet mean and standard deviation values.
- **Background Removal**: Using a DeepLabV3 model, the background was removed to focus the model on the birds.
- **Data Augmentation**: Techniques such as random horizontal flips and color jitter were applied to make the model more robust.

## Models
Several deep learning architectures were tested for bird species classification:
1. **ResNet-50**: A widely used residual network architecture.
2. **ResNeXt-50 (32x4d)**: A variant of ResNet that increases feature learning through parallel paths.
3. **EfficientNet-B0**: A small but efficient architecture that balances depth, width, and resolution.
4. **EfficientNet-B5**: A larger version of EfficientNet with more capacity for feature extraction.

All models were initialized with pre-trained weights from ImageNet, and the final classification layer was modified for 200 classes.

## Training and Evaluation
The training process involved:
- **Optimizer**: Adam optimizer was used with a learning rate of 1e-4.
- **Loss Function**: Cross-entropy loss was used to measure the difference between predicted and true labels.
- **Batch Size**: 16 images per batch.
- **Learning Rate Scheduler**: A StepLR scheduler with a decay factor of 0.1 every 5 epochs was used.
- **Early Stopping**: Training was halted if validation accuracy plateaued to prevent overfitting.

## Results
The models were evaluated based on **Top-1 Accuracy** and **Average Accuracy per Class**:
- **Best Model**: EfficientNet-B5 achieved the highest Top-1 accuracy (78.49%) and average accuracy per class (78.04%) after preprocessing.
- Preprocessing, especially background removal, significantly improved model performance by reducing noise and distractions.

| Model               | Top-1 Accuracy | Average Accuracy per Class |
|---------------------|----------------|-----------------------------|
| ResNet-50           | 75.45%         | 75.02%                      |
| ResNeXt-50 (32x4d)  | 77.77%         | 77.35%                      |
| EfficientNet-B0     | 75.86%         | 75.25%                      |
| **EfficientNet-B5** | **78.49%**     | **78.04%**                  |

## Conclusion
The project demonstrates the effectiveness of deep learning for fine-grained bird species classification. EfficientNet-B5, coupled with preprocessing techniques such as background removal and data augmentation, produced the best results. Further improvements could be made by exploring more advanced augmentation techniques or fine-tuning hyperparameters.

## References
1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
2. Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated Residual Transformations for Deep Neural Networks.
3. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
4. Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2017). DeepLab: Semantic Image Segmentation with Deep Convolutional Nets.
