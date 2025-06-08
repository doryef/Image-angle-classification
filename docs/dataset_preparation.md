# Dataset Preparation Guide

This document explains how to prepare and organize your image dataset for camera angle classification.

## Directory Structure

Your dataset should be organized into three main splits:
```
data/
├── train/  (70-80% of your data)
│   ├── 0-30/     # Near horizontal camera angles
│   ├── 30-60/    # Medium elevation angles
│   └── 60-90/    # High/top-down angles
├── val/    (10-15% of your data)
│   ├── 0-30/
│   ├── 30-60/
│   └── 60-90/
└── test/   (10-15% of your data)
    ├── 0-30/
    ├── 30-60/
    └── 60-90/
```

## Image Requirements

- **Supported Formats**: PNG, JPG, JPEG, WebP
- **Processing**: Images will be automatically resized to 224x224 pixels during training
- **Color Space**: RGB (grayscale images will be converted to RGB)

## Data Collection Guidelines

1. **Angle Measurement**
   - 0-30°: Images taken from near ground level
   - 30-60°: Images taken from medium elevation
   - 60-90°: Images taken from high elevation or aerial view

2. **Dataset Balance**
   - Try to collect a similar number of images for each angle category
   - Minimum recommended: 100 images per category in training set

3. **Data Splitting**
   - Training set: Use for model training (70-80% of data)
   - Validation set: Use for monitoring training progress (10-15%)
   - Test set: Use only for final evaluation (10-15%)

4. **Quality Considerations**
   - Ensure clear visibility
   - Include variety in lighting conditions
   - Include variety in distances
   - Include different types of scenes/subjects

## Data Augmentation

The training pipeline automatically applies the following augmentations:
1. Random cropping
2. Optional horizontal flipping
3. Zoom variations (configurable)
4. Color jittering:
   - Brightness variation: ±20%
   - Contrast variation: ±20%

## Dataset Validation

Before training, ensure:
1. All image files are readable
2. Images are properly categorized by angle
3. Class distribution is relatively balanced
4. No duplicate images across train/val/test splits