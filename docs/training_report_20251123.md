# Training Report - Camera Angle Classification
**Date**: November 23, 2025  
**Dataset**: Real VisDrone Data Only  
**Model**: ResNet50 with Custom Classification Head

---

## Executive Summary

Trained a camera angle classifier to categorize images into three viewing angles (0-30°, 30-60°, 60-90°). The model shows **severe overfitting** with 93% training accuracy but only 51% validation accuracy, indicating insufficient training data for the model complexity.

**Key Results:**
- ✅ Training Accuracy: **93.02%**
- ⚠️ Validation Accuracy: **51.33%** 
- ⚠️ Test Accuracy: **58.00%**
- 🚨 Overfitting Gap: **41.69%**

---

## 1. Dataset Configuration

### Training Data
- **Total Training Images**: 385 images
- **Class Distribution** (Balanced):
  - 0-30°: 128 images (33.2%)
  - 30-60°: 128 images (33.2%)
  - 60-90°: 129 images (33.5%)

### Data Splits
- **Training**: Towns with real VisDrone data
- **Validation**: Real data from separate town
- **Test Set**: 150 images (50 per class)

### Data Source
```yaml
train_dir: "data/real_only/train"
val_dir: "data/real_only/val/real"
test_dir: "data/real_only/test_real"
```

---

## 2. Model Architecture

### Backbone
- **Architecture**: ResNet50
- **Pretrained Weights**: ImageNet (IMAGENET1K_V2)
- **Total Backbone Parameters**: ~23.5 million
- **Training Status**: **FROZEN** ❄️
  - All backbone parameters fixed
  - Only classification head is trainable

### Classification Head
```
Input: 2048 features (from ResNet50)
    ↓
Linear(2048 → 512)
    ↓
ReLU Activation
    ↓
Dropout(0.6)
    ↓
Linear(512 → 256)
    ↓
ReLU Activation
    ↓
Dropout(0.6)
    ↓
Linear(256 → 3)  [Output: 3 classes]
```

**Trainable Parameters**: ~1,313,795
- Layer 1: 2048 × 512 + 512 = 1,049,088
- Layer 2: 512 × 256 + 256 = 131,328
- Layer 3: 256 × 3 + 3 = 771

**Parameters-to-Data Ratio**: 3,412 parameters per training sample ⚠️

---

## 3. Training Configuration

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Epochs** | 20 | With early stopping |
| **Batch Size** | 32 | |
| **Optimizer** | Adam | Adaptive learning rate |
| **Learning Rate (Head)** | 0.001 | Only head is trainable |
| **Learning Rate (Backbone)** | 0.0001 | Not used (frozen) |
| **LR Scheduler** | ReduceLROnPlateau | Patience=2, factor=0.5 |
| **Loss Function** | CrossEntropyLoss | |

### Regularization Techniques

| Technique | Value | Purpose |
|-----------|-------|---------|
| **Dropout** | 0.6 (60%) | Very aggressive - may be too high |
| **Weight Decay** | 0.0001 | L2 regularization |
| **Early Stopping** | Patience=10 | Stop if no val improvement |

### Data Augmentation (Training Only)

**Geometric Transforms:**
- RandomResizedCrop: 960px → 224px
- Scale range: 0.8-1.0 (zoom variation: ±20%)
- Aspect ratio: 0.75-1.33
- Horizontal flip: 50% probability

**Color Augmentations:**
- Brightness: ±30%
- Contrast: ±30%
- Saturation: ±20%
- Hue: ±10%
- Gaussian Blur: 10% probability (kernel=3, σ=0.1-2.0)

**Preprocessing:**
- ImageNet normalization
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

---

## 4. Training Results

### Final Metrics (Epoch 11)
- **Training Loss**: 0.2015
- **Training Accuracy**: 93.02%
- **Validation Loss**: 1.4706
- **Validation Accuracy**: 51.33%

### Best Performance
- **Best Val Loss**: 1.0051 (achieved during training)
- **Best Val Accuracy**: 54.33%

### Training Progression
```
Epoch  | Train Acc | Val Acc | Train Loss | Val Loss
-------|-----------|---------|------------|----------
1      | ~50%      | ~33%    | High       | High
5      | ~80%      | ~45%    | Medium     | High
11     | 93.02%    | 51.33%  | 0.20       | 1.47
```

![Training History](../plots/training_history_20251123.png)

**Observation**: Training accuracy increases steadily while validation accuracy plateaus early, indicating overfitting.

---

## 5. Test Set Evaluation

### Overall Performance
- **Test Accuracy**: 58.00%
- **Total Samples**: 150 (50 per class)
- **Misclassified**: 63 samples (42%)

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0-30°** | 0.55 | 0.88 | 0.68 | 50 |
| **30-60°** | 0.37 | 0.30 | 0.33 | 50 |
| **60-90°** | 0.97 | 0.56 | 0.71 | 50 |
| **Macro Avg** | 0.63 | 0.58 | 0.57 | 150 |
| **Weighted Avg** | 0.63 | 0.58 | 0.57 | 150 |

### Key Observations

**Strong Performance:**
- 60-90° class has excellent precision (0.97) - when predicted, it's almost always correct
- 0-30° class has good recall (0.88) - model finds most of these angles

**Weak Performance:**
- 30-60° class struggles significantly (F1=0.33)
- This intermediate angle is confused with both extremes
- Low precision indicates many false positives

### Confusion Analysis

![Confusion Matrix](../plots/confusion_matrix_20251123.png)

**Common Misclassifications:**
1. Many 30-60° images classified as 0-30° or 60-90°
2. Some 60-90° images confused with 30-60°
3. Model is uncertain on borderline cases (probabilities around 35-42%)

### Sample Misclassification Examples

![Misclassified Samples](../plots/misclassified_samples_20251123.png)

```
Sample 1: True=0-30°, Pred=30-60° (40.24% confidence)
  Probabilities: 0-30°: 37.79% | 30-60°: 40.24% | 60-90°: 21.97%
  → Very close call, model uncertain

Sample 2: True=60-90°, Pred=30-60° (37.56% confidence)
  Probabilities: 0-30°: 37.56% | 30-60°: 35.65% | 60-90°: 26.80%
  → Model confused, all classes have similar probability

Sample 3: True=30-60°, Pred=0-30° (41.50% confidence)
  Probabilities: 0-30°: 41.50% | 30-60°: 37.95% | 60-90°: 20.55%
  → Close decision between two most similar angles
```

---

## 6. Analysis & Diagnosis

### Overfitting Indicators

| Metric | Value | Status |
|--------|-------|--------|
| Train-Val Accuracy Gap | 41.69% | 🚨 Severe |
| Train Loss | 0.20 | ✅ Good |
| Val Loss | 1.47 | ⚠️ High |
| Test vs Val Consistency | 58% vs 51% | ✅ Reasonable |

### Root Causes of Overfitting

#### 1. **Dataset Size (Critical)** 🚨
- Only **385 training images** for 1.3M trainable parameters
- Ratio: **3,412 parameters per training sample**
- Modern deep learning typically needs 1,000-10,000+ images per class
- **Impact**: Model memorizes training data instead of learning generalizable patterns

#### 2. **Model Complexity**
- 3-layer classification head may be overkill for 3 classes
- With frozen backbone, the head must do all the learning
- High capacity allows memorization of small dataset
- **Impact**: Model fits training noise rather than signal

#### 3. **Frozen Backbone Limitation**
- ResNet50 trained on ImageNet (everyday objects)
- Camera angle features are different from object recognition
- Frozen weights cannot adapt to this specific task
- **Impact**: Suboptimal features for angle classification

#### 4. **Excessive Dropout (0.6)**
- 60% dropout is very aggressive
- Creates large train-test discrepancy
- May be preventing effective learning on small dataset
- **Impact**: Unstable training, reduced model capacity

#### 5. **Limited Feature Diversity**
- Small dataset from limited number of scenes/towns
- Model overfits to specific scene characteristics
- Heavy augmentation helps but can't fully compensate
- **Impact**: Poor generalization to new environments

---

## 7. Recommendations

### Immediate Actions (High Priority)

#### 1. **Scale Up Training Data** 🎯
**Problem**: 385 images insufficient  
**Solutions**:
- Use full synthetic dataset: `data/synthetic_only/` (24,811 images)
- Collect more real VisDrone images (target: 1,000+ per class)
- Combine synthetic + real data for hybrid training

#### 2. **Simplify Classification Head** 
**Current**: 2048 → 512 → 256 → 3  
**Proposed**: 2048 → 128 → 3

```python
nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(),
    nn.Dropout(0.3),  # Reduced from 0.6
    nn.Linear(128, 3)
)
```
**Expected Impact**: Reduce parameters by ~90%, less memorization

#### 3. **Reduce Dropout Rate**
**Current**: 0.6  
**Proposed**: 0.3-0.4  
**Reason**: Less aggressive regularization for small dataset

#### 4. **Unfreeze Backbone Layers**
**Current**: All frozen  
**Proposed Options**:
- Unfreeze last residual block (layer4)
- Unfreeze last 2 blocks (layer3 + layer4)
- Full fine-tuning with lower LR (0.00001)

**Expected Impact**: Allow task-specific feature learning

### Medium-Term Improvements

#### 5. **Adjust Learning Rate**
- Reduce head LR: 0.001 → 0.0005
- Add warmup schedule for stability
- Extend training to 30-50 epochs with more data

#### 6. **Class Weighting**
- If using more data and classes become imbalanced
- Weight loss inversely to class frequency

#### 7. **Advanced Augmentation**
- Implement MixUp or CutMix
- Add AutoAugment policies
- Test Time Augmentation (TTA) for inference

### Long-Term Optimizations

#### 8. **Architecture Exploration**
- Try smaller backbones: ResNet18, ResNet34
- Test EfficientNet or MobileNet variants
- Experiment with Vision Transformers (ViT)

#### 9. **Ensemble Methods**
- Train multiple models with different initializations
- Combine predictions for improved accuracy
- Use different architectures in ensemble

#### 10. **Self-Supervised Pretraining**
- Pretrain on unlabeled drone images
- Fine-tune on labeled angle data
- Leverage large unlabeled datasets

---

## 8. Next Steps

### Immediate Experiment Plan

**Experiment 1: Use Synthetic Data**
```yaml
data:
  train_dir: "data/synthetic_only/train"
  val_dir: "data/synthetic_only/val"
  test_dir: "data/synthetic_only/test"
```
- Expected: Much better generalization with 24,811 images
- Anticipated accuracy: 80-90%

**Experiment 2: Simplified Architecture**
- Modify `angle_classifier.py` to use 2-layer head
- Reduce dropout to 0.3
- Keep same data and training settings
- Compare results

**Experiment 3: Unfreeze Backbone**
```python
model = AngleClassifier(
    pretrained=True,
    freeze_backbone=False,  # Changed
    dropout_rate=0.3
)
```
- Use lower learning rate for backbone (0.00001)
- Expected: Better feature adaptation

### Success Criteria
- Reduce overfitting gap to <10%
- Achieve >80% validation accuracy
- Improve 30-60° class F1-score to >0.6

---

## 9. Generated Artifacts

### Visualizations
1. **Confusion Matrix**: `plots/confusion_matrix_20251123.png`
   - Shows classification patterns across all classes
   
2. **Misclassified Samples**: `plots/misclassified_samples_20251123.png`
   - 10 example images with predictions and probability distributions
   - Color-coded bars: Green (true), Red (predicted), Gray (other)

3. **Training History**: `plots/training_history_20251123.png`
   - Training/validation loss curves
   - Training/validation accuracy curves
   - Test accuracy reference line

### Data Files
- Training history: `logs/training_history_20251123_153743.csv`
- Model checkpoint: `checkpoints/angle_classifier_20251123_153907.pth`
- Evaluation metrics: Console output from `src/evaluate.py`

---

## 10. Conclusion

The current model demonstrates the classic **small dataset overfitting problem**. With only 385 training images and 1.3M trainable parameters, the model memorizes rather than generalizes. The 93% training accuracy shows the model has sufficient capacity, but the 51% validation accuracy reveals poor generalization.

**Critical Path Forward:**
1. **Increase training data** (use 24K synthetic images available)
2. **Simplify model architecture** (reduce head complexity)
3. **Fine-tune regularization** (reduce dropout, unfreeze backbone)

With these changes, the model should achieve 80-90% accuracy with proper generalization to unseen environments.

---

**Report Generated**: November 23, 2025  
**Configuration Files**: 
- Model: `src/models/angle_classifier.py`
- Config: `configs/config.yaml`
- Training: `src/train.py`
- Evaluation: `src/evaluate.py`
