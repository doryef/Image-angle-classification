# Training Monitoring Features

This document describes the TensorBoard monitoring capabilities available during model training.

## Configuration

Monitoring can be enabled/disabled in `configs/config.yaml`:
```yaml
training:
  enable_monitoring: true  # Set to false to disable all monitoring
```

## Core Features (Always Available)

These features are available with the base dependencies:

1. **Training Metrics**
   - Loss and accuracy curves (training/validation)
   - Per-class accuracy tracking
   - Learning rate visualization
   - Basic model architecture visualization

2. **Sample Predictions**
   - Grid of sample images
   - Predictions vs ground truth
   - Updates every 50 training steps

3. **Confusion Matrix**
   - Per-epoch confusion matrix heatmap
   - Class distribution visualization

## Enhanced Features (Optional)

These features require additional packages. Install them with:
```bash
pip install -r requirements.txt
```

1. **Advanced Model Visualization** (requires `torchviz` and `graphviz`)
   - Detailed computational graph
   - Layer connectivity visualization
   - Static PNG export of model architecture

2. **System Monitoring** (requires `psutil`)
   - CPU usage tracking
   - Memory utilization
   - System resource metrics

3. **GPU Monitoring** (requires `gputil`, Linux/Mac only)
   - GPU memory usage
   - GPU utilization
   - Temperature tracking
   - Note: Basic CUDA memory monitoring still available on Windows

## Accessing TensorBoard

1. Start training:
```bash
python train.py
```

2. In a separate terminal, launch TensorBoard:
```bash
python src/utils/launch_tensorboard.py
```

3. Open your browser and navigate to:
   - Local: http://localhost:6006
   - Remote: http://<your-ip>:6006

## Data Location

- TensorBoard logs are stored in the `logs/` directory
- Each training run creates a timestamped subdirectory
- Model architecture diagrams (if enabled) are saved in the run's log directory

## Best Practices

1. **Resource Usage**
   - Monitoring has minimal performance impact
   - System monitoring samples every 10 steps
   - Parameter histograms update every 100 steps

2. **Storage**
   - Log files are automatically managed by TensorBoard
   - Old runs can be safely deleted from `logs/`
   - Each run typically uses 10-50MB of disk space

3. **Remote Monitoring**
   - Ensure port 6006 is accessible if monitoring remotely
   - Use SSH tunneling for secure remote access
   - Multiple users can view the same TensorBoard instance