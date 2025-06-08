# Camera Angle Classification

This project uses deep learning to classify images based on the camera angle (viewing angle) from which they were taken. It categorizes images into three classes based on the vertical camera angle:
- 0-30° (near horizontal viewing angle)
- 30-60° (intermediate viewing angle)
- 60-90° (steep/top-down viewing angle)

## Project Structure
```
├── configs/              # Configuration files
├── data/                # Dataset directory
│   ├── train/           # Training data
│   ├── val/            # Validation data
│   └── test/           # Test data
├── src/                 # Source code
├── checkpoints/         # Saved model checkpoints
├── logs/               # Training logs
└── plots/              # Evaluation plots
```

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Data Organization
Organize your images based on their camera angles:
```
data/
├── train/
│   ├── 0-30/     # Images taken from a near-horizontal viewpoint
│   ├── 30-60/    # Images taken from a medium elevation viewpoint
│   └── 60-90/    # Images taken from a high/top-down viewpoint
├── val/          # Same structure as train/
└── test/         # Same structure as train/
```

### Dataset Guidelines
- Supported formats: PNG, JPG, JPEG, WebP
- Images will be resized to 224x224 pixels during training
- Try to maintain balanced classes (similar number of images in each camera angle category)
- For accurate angle classification:
  - Label images based on the vertical angle between the camera's viewing direction and the horizontal plane
  - Keep subject matter consistent across different angles if possible
  - Consider using a reference object photographed from known angles to calibrate your dataset

## Training

1. Adjust hyperparameters in `configs/config.yaml` if needed
2. Run training:
```bash
python train.py
```

The training script will:
- Save model checkpoints in the `checkpoints/` directory
- Log training metrics using TensorBoard
- Display progress and metrics during training

## Evaluation

After training, evaluate the model on the test set:
```bash
python evaluate.py
```

This will:
- Load the best model checkpoint
- Generate classification metrics
- Create a confusion matrix visualization in `plots/`

## Training Progress Visualization

View training progress using TensorBoard:
```bash
tensorboard --logdir=logs
```

## Model Architecture

The model uses transfer learning with:
- ResNet50 backbone (pretrained on ImageNet)
- Custom classification head for camera angle categories
- Configurable frozen/unfrozen layers for fine-tuning