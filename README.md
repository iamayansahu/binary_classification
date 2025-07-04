# binary_classification
A PyTorch-based binary image classification project with pretrained models, data augmentation, class imbalance handling, and evaluation/inference support.
Gender Classification using CNN (ResNet18).

**Dataset link**: https://drive.google.com/file/d/1u3N-01BSv9ARruveQdCqsgOEzIiqLkTq/view?usp=drive_link

This project implements a deep learning model to classify gender (male / female) from face images using *PyTorch* and *ResNet18*. The training pipeline handles class imbalance with a *weighted sampler* and *weighted loss*, and applies various data augmentations for better generalization.
## Features & Techniques Used

- Model: ResNet18 (pretrained)
- Loss: BCEWithLogitsLoss with class balancing
- Sampling: WeightedRandomSampler to handle class imbalance
- Transforms:
  - RandomHorizontalFlip
  - ColorJitter
  - Gaussian Blur
  - Normalization
- Evaluation: classification_report from sklearn
- Hardware Support: GPU acceleration (cuda if available)
## How to Run

1. Place your dataset in the following format:

Task_A/ ├── train/  ├── female/ │   └── male/ 
        └── val/    ├── female/ | └── male/

2. Update DATA_DIR path in train.py if necessary.

3. Run the training and evaluation:
python train.py

4. Trained model will be saved as: gender_model___final.pth

📊 Sample Output

📦 Epoch [30/30] - Loss: 0.2408

# 📊 Training Classification Report:
              precision    recall  f1-score   support
     female       0.94      0.88      0.91       1528
       male       0.89      0.94      0.91       1528
    accuracy                          0.91       3056


    Training Accuracy: 90.97 %


# 📊 Validation Classification Report:
              precision    recall  f1-score   support
     female       0.69      0.85      0.76       79
       male       0.96      0.91      0.94       343
    accuracy                          0.90       422


    Validation Accuracy: 90.05%
    
    
# Dependencies

torch

torchvision

scikit-learn

numpy

PIL

**Install them using:**

pip install torch torchvision scikit-learn numpy pillow
