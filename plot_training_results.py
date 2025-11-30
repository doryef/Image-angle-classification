#!/usr/bin/env python3
"""
Parse training log and create validation accuracy plot
"""

import re
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(log_file_path):
    """Parse the training log to extract validation accuracy data"""
    
    epochs = []
    val_accuracies = []
    train_accuracies = []
    val_losses = []
    train_losses = []
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Split by epochs
    epoch_pattern = r'Epoch (\d+)/\d+'
    epoch_matches = list(re.finditer(epoch_pattern, content))
    
    for i, match in enumerate(epoch_matches):
        epoch_num = int(match.group(1))
        
        # Find the end of this epoch's section
        if i + 1 < len(epoch_matches):
            epoch_section = content[match.start():epoch_matches[i+1].start()]
        else:
            epoch_section = content[match.start():]
        
        # Extract validation accuracy
        val_acc_pattern = r'Validation Loss: ([\d.]+), Validation Acc: ([\d.]+)%'
        val_match = re.search(val_acc_pattern, epoch_section)
        
        # Extract training accuracy and loss
        train_pattern = r'Training Loss: ([\d.]+), Training Acc: ([\d.]+)%'
        train_match = re.search(train_pattern, epoch_section)
        
        if val_match and train_match:
            epochs.append(epoch_num)
            val_losses.append(float(val_match.group(1)))
            val_accuracies.append(float(val_match.group(2)))
            train_losses.append(float(train_match.group(1)))
            train_accuracies.append(float(train_match.group(2)))
    
    return epochs, val_accuracies, train_accuracies, val_losses, train_losses

def create_accuracy_plot(epochs, val_accuracies, train_accuracies, output_path='training_accuracy_plot.png'):
    """Create and save the accuracy plot"""
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Training vs Validation Accuracy
    ax1.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training vs Validation Accuracy Over Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add annotations for max values
    max_val_acc = max(val_accuracies)
    max_val_epoch = epochs[val_accuracies.index(max_val_acc)]
    ax1.annotate(f'Max Val Acc: {max_val_acc:.1f}%\\n(Epoch {max_val_epoch})', 
                xy=(max_val_epoch, max_val_acc), 
                xytext=(max_val_epoch + len(epochs)*0.1, max_val_acc + 5),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Plot 2: Overfitting Gap
    accuracy_gap = [train - val for train, val in zip(train_accuracies, val_accuracies)]
    ax2.plot(epochs, accuracy_gap, 'g-', label='Overfitting Gap (Train - Val)', linewidth=2, marker='^', markersize=4)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy Gap (%)')
    ax2.set_title('Overfitting Gap Over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    avg_gap = np.mean(accuracy_gap)
    max_gap = max(accuracy_gap)
    final_gap = accuracy_gap[-1] if accuracy_gap else 0
    
    stats_text = f'Avg Gap: {avg_gap:.1f}%\\nMax Gap: {max_gap:.1f}%\\nFinal Gap: {final_gap:.1f}%'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_path

def print_summary_stats(epochs, val_accuracies, train_accuracies):
    """Print summary statistics"""
    
    print("\\n" + "="*50)
    print("TRAINING SUMMARY STATISTICS")
    print("="*50)
    
    if not epochs:
        print("No training data found in log file!")
        return
    
    print(f"Total epochs completed: {len(epochs)}")
    print(f"Epoch range: {min(epochs)} - {max(epochs)}")
    
    print(f"\\nValidation Accuracy:")
    print(f"  - Maximum: {max(val_accuracies):.2f}% (Epoch {epochs[val_accuracies.index(max(val_accuracies))]})")
    print(f"  - Minimum: {min(val_accuracies):.2f}% (Epoch {epochs[val_accuracies.index(min(val_accuracies))]})")
    print(f"  - Final: {val_accuracies[-1]:.2f}%")
    print(f"  - Average: {np.mean(val_accuracies):.2f}%")
    
    print(f"\\nTraining Accuracy:")
    print(f"  - Maximum: {max(train_accuracies):.2f}% (Epoch {epochs[train_accuracies.index(max(train_accuracies))]})")
    print(f"  - Final: {train_accuracies[-1]:.2f}%")
    print(f"  - Average: {np.mean(train_accuracies):.2f}%")
    
    # Overfitting analysis
    accuracy_gaps = [train - val for train, val in zip(train_accuracies, val_accuracies)]
    print(f"\\nOverfitting Analysis:")
    print(f"  - Average gap (Train - Val): {np.mean(accuracy_gaps):.2f}%")
    print(f"  - Maximum gap: {max(accuracy_gaps):.2f}%")
    print(f"  - Final gap: {accuracy_gaps[-1]:.2f}%")
    
    if np.mean(accuracy_gaps) > 20:
        print("  - ⚠️  HIGH OVERFITTING DETECTED!")
    elif np.mean(accuracy_gaps) > 10:
        print("  - ⚠️  Moderate overfitting")
    else:
        print("  - ✅ Good generalization")

def main():
    log_file = '/home/p24s09/image_angle_classifier/train.log'
    
    print("Parsing training log...")
    epochs, val_accuracies, train_accuracies, val_losses, train_losses = parse_training_log(log_file)
    
    if not epochs:
        print("❌ No training data found in the log file!")
        return
    
    print(f"✅ Found data for {len(epochs)} epochs")
    
    # Print summary statistics
    print_summary_stats(epochs, val_accuracies, train_accuracies)
    
    # Create the plot
    print("\\nCreating accuracy plot...")
    output_path = create_accuracy_plot(epochs, val_accuracies, train_accuracies)
    print(f"✅ Plot saved to: {output_path}")
    
    # Save data to CSV for further analysis
    import pandas as pd
    df = pd.DataFrame({
        'Epoch': epochs,
        'Training_Accuracy': train_accuracies,
        'Validation_Accuracy': val_accuracies,
        'Training_Loss': train_losses,
        'Validation_Loss': val_losses
    })
    csv_path = 'training_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Data saved to: {csv_path}")

if __name__ == "__main__":
    main()
