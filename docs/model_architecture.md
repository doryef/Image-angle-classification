# Model Architecture

The camera angle classification model uses transfer learning with a ResNet50 backbone pretrained on ImageNet.

## Architecture Overview

### Backbone
- **Model**: ResNet50
- **Weights**: ImageNet pretrained (IMAGENET1K_V2)
- **Freezing**: By default, the backbone is frozen to preserve pretrained features

### Classification Head
```
Sequential(
    Linear(2048 -> 256)
    ReLU
    Dropout(0.5)
    Linear(256 -> 3)  # 3 classes: 0-30°, 30-60°, 60-90°
)
```

## Training Configuration

### Learning Rates
- Backbone: 0.0001 (when unfrozen)
- Classification head: 0.001

### Optimizer
- Adam optimizer with different learning rates for backbone and head
- ReduceLROnPlateau scheduler for adaptive learning rate

### Data Augmentation
- Random cropping
- Horizontal flipping (optional)
- Zoom variation
- Color jittering (brightness and contrast)

## Model Capabilities

The model is designed to classify images based on their camera viewing angle into three categories:
1. 0-30° (near horizontal view)
2. 30-60° (intermediate angle)
3. 60-90° (top-down view)

## Fine-tuning

The model supports progressive unfreezing of the backbone layers using the `unfreeze_layers()` method. This can be useful if the initial frozen backbone performance isn't satisfactory.