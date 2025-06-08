# Training and Evaluation Guide

This document explains how to train and evaluate the camera angle classification model.

## Training Process

### 1. Configuration
First, review and adjust `configs/config.yaml` if needed:
```yaml
model:
  pretrained: true        # Use pretrained weights
  freeze_backbone: true   # Freeze ResNet backbone initially

training:
  num_epochs: 20         # Number of training epochs
  batch_size: 32        # Batch size for training
  backbone_lr: 0.0001   # Learning rate for backbone (if unfrozen)
  head_lr: 0.001       # Learning rate for classification head

augmentation:
  zoom_range: 0.1      # Range of zoom variation
  brightness: 0.2      # Brightness adjustment range
  contrast: 0.2        # Contrast adjustment range
  horizontal_flip: true # Enable horizontal flipping
```

### 2. Starting Training
Run:
```bash
python train.py
```

The training script will:
- Use GPU if available, fall back to CPU if not
- Save model checkpoints in `checkpoints/`
- Log metrics to TensorBoard in `logs/`
- Print progress updates

### 3. Monitoring Training
You can monitor training in two ways:

#### Local Access
```bash
python src/utils/launch_tensorboard.py
```

#### Remote Access
To monitor training from another computer:
1. On your lab computer (with GPU):
   ```bash
   python src/utils/launch_tensorboard.py
   ```
2. The script will display the IP address and port
3. From your remote computer:
   - Open a web browser
   - Navigate to the provided URL (http://<lab-computer-ip>:6006)

TensorBoard shows:
- Training/validation loss curves
- Accuracy metrics
- Learning rate changes

### 4. Model Checkpoints
- Best models are saved based on validation loss
- Checkpoints include:
  - Model state
  - Optimizer state
  - Training metrics

## Evaluation Process

### 1. Running Evaluation
After training, evaluate on test set:
```bash
python evaluate.py
```

This will:
- Load the best checkpoint
- Run inference on test set
- Generate performance metrics
- Create visualization plots

### 2. Evaluation Metrics
The evaluation provides:
- Overall accuracy
- Per-class precision/recall
- Confusion matrix
- Detailed classification report

### 3. Interpreting Results
View results in:
- Terminal output for metrics
- `plots/confusion_matrix.png` for visualization
- Logs directory for detailed history

## Fine-tuning Tips

1. If initial performance is unsatisfactory:
   - Unfreeze some backbone layers
   - Reduce learning rate
   - Increase training epochs

2. If overfitting occurs:
   - Increase dropout
   - Reduce model capacity
   - Add more training data

3. If underfitting occurs:
   - Unfreeze more layers
   - Increase model capacity
   - Train for more epochs