#!/usr/bin/env python3
"""
Plot training metrics from CSV file
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_training_metrics():
    """Plot training and validation metrics from CSV"""
    
    # Read the CSV file
    df = pd.read_csv('training_metrics.csv')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy over epochs
    ax1.plot(df['Epoch'], df['Training_Accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(df['Epoch'], df['Validation_Accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add horizontal line at 50% for validation (shows overfitting clearly)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random Baseline (33.3%)')
    ax1.axhline(y=33.3, color='orange', linestyle='--', alpha=0.5, label='Random Baseline (33.3%)')
    
    # Plot 2: Loss over epochs
    ax2.plot(df['Epoch'], df['Training_Loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(df['Epoch'], df['Validation_Loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training vs Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better loss visualization
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('training_metrics_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'training_metrics_plot.png'")
    
    # Show statistics
    print("\n=== Training Statistics ===")
    print(f"Final Training Accuracy: {df['Training_Accuracy'].iloc[-1]:.2f}%")
    print(f"Final Validation Accuracy: {df['Validation_Accuracy'].iloc[-1]:.2f}%")
    print(f"Best Validation Accuracy: {df['Validation_Accuracy'].max():.2f}% (Epoch {df.loc[df['Validation_Accuracy'].idxmax(), 'Epoch']})")
    print(f"Accuracy Gap (Train-Val): {df['Training_Accuracy'].iloc[-1] - df['Validation_Accuracy'].iloc[-1]:.2f}%")
    
    print(f"\nFinal Training Loss: {df['Training_Loss'].iloc[-1]:.4f}")
    print(f"Final Validation Loss: {df['Validation_Loss'].iloc[-1]:.4f}")
    print(f"Best Validation Loss: {df['Validation_Loss'].min():.4f} (Epoch {df.loc[df['Validation_Loss'].idxmin(), 'Epoch']})")
    
    # Check for overfitting indicators
    accuracy_gap = df['Training_Accuracy'].iloc[-1] - df['Validation_Accuracy'].iloc[-1]
    if accuracy_gap > 30:
        print(f"\n⚠️  SEVERE OVERFITTING DETECTED!")
        print(f"   Training accuracy ({df['Training_Accuracy'].iloc[-1]:.1f}%) is {accuracy_gap:.1f}% higher than validation ({df['Validation_Accuracy'].iloc[-1]:.1f}%)")
    elif accuracy_gap > 15:
        print(f"\n⚠️  Overfitting detected. Gap: {accuracy_gap:.1f}%")
    else:
        print(f"\n✅ Good generalization. Gap: {accuracy_gap:.1f}%")
    
    plt.show()

if __name__ == "__main__":
    plot_training_metrics()
