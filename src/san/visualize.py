"""
Visualization utilities for SAN-RAD model analysis and reporting.
"""
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

def plot_training_curves(history, save_dir='results'):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with 'epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'best_epoch'
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = history['epochs']
    best_epoch = history.get('best_epoch', None)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    # Mark best epoch where model was saved
    if best_epoch and best_epoch in epochs:
        idx = epochs.index(best_epoch)
        ax1.plot(best_epoch, history['val_loss'][idx], 'go', markersize=8, 
                label=f'Best Model (Epoch {best_epoch})', zorder=5)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, [acc*100 for acc in history['train_acc']], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, [acc*100 for acc in history['val_acc']], 'r-', label='Validation Accuracy', linewidth=2)
    
    # Mark best epoch on accuracy plot too
    if best_epoch and best_epoch in epochs:
        idx = epochs.index(best_epoch)
        ax2.plot(best_epoch, history['val_acc'][idx]*100, 'go', markersize=8,
                label=f'Best Model (Epoch {best_epoch})', zorder=5)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_path}")
    
def plot_test_metrics(stats, save_dir='results'):
    """
    Plot test set accuracy breakdown by question type.
    Shows both raw and adjusted accuracy for open-ended questions.
    
    Args:
        stats: Dictionary with 'open' and 'closed' statistics
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    # Open-ended: both raw and adjusted
    open_raw_acc = stats['open']['correct'] / max(stats['open']['total'], 1) * 100
    open_adjusted_total = stats['open']['total'] - stats['open']['unk_label']
    open_adjusted_acc = stats['open']['correct'] / max(open_adjusted_total, 1) * 100
    
    # Closed-ended: raw (adjusted is similar since few UNKs)
    closed_acc = stats['closed']['correct'] / max(stats['closed']['total'], 1) * 100
    
    # Overall: raw
    overall_acc = (stats['open']['correct'] + stats['closed']['correct']) / \
                  max(stats['open']['total'] + stats['closed']['total'], 1) * 100
    
    # Categories and accuracies
    categories = ['Open\n(Raw)', 'Open\n(Adjusted)', 'Closed', 'Overall']
    accuracies = [open_raw_acc, open_adjusted_acc, closed_acc, overall_acc]
    colors = ['#3498db', '#5dade2', '#2ecc71', '#9b59b6']  # Two shades of blue for open
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add subtitle annotation for adjusted
    ax.text(0.5, -0.15, 
            'Adjusted Open-Ended excludes 60.3% UNK labels (unseen answers in training)',
            ha='center', transform=ax.transAxes, fontsize=10, style='italic', color='gray')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Test Set Accuracy Breakdown', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(accuracies) * 1.2)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'test_accuracy_breakdown.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Test metrics saved to {save_path}")

def visualize_attention_map(model, image_tensor, question_tensor, q_type_idx, organ_idx, 
                            original_image_path, question_text, prediction, ground_truth,
                            device='cpu', save_path=None):
    """
    Visualize attention heatmap overlaid on the original image.
    Shows only the original image and the overlay (2 panels).
    
    Args:
        model: Trained SAN model (modified to return attention)
        image_tensor: Preprocessed image tensor [1, 3, 224, 224]
        question_tensor: Question tensor [1, seq_len]
        q_type_idx: Question type index [1]
        organ_idx: Organ index [1]
        original_image_path: Path to original image file
        question_text: Question as string
        prediction: Predicted answer
        ground_truth: Ground truth answer
        device: Device to run model on
        save_path: Path to save visualization
    """
    model.eval()
    
    with torch.no_grad():
        # Get model outputs with attention
        image_tensor = image_tensor.to(device)
        question_tensor = question_tensor.to(device)
        q_type_idx = q_type_idx.to(device)
        organ_idx = organ_idx.to(device)
        
        # Forward pass (we'll need to modify model to return attention)
        output, attention_weights = model.forward_with_attention(
            image_tensor, question_tensor, q_type_idx, organ_idx
        )
    
    # Get attention from last layer (14x14 or 7x7)
    attn = attention_weights.cpu().numpy().squeeze()  # [49] or [196]
    grid_size = int(np.sqrt(len(attn)))
    attn_map = attn.reshape(grid_size, grid_size)
    
    # Load original image
    original_img = Image.open(original_image_path).convert('RGB')
    img_array = np.array(original_img)
    
    # Resize attention map to image size using bicubic interpolation
    attn_resized = cv2.resize(attn_map, (img_array.shape[1], img_array.shape[0]), 
                              interpolation=cv2.INTER_CUBIC)
    
    # Normalize attention
    attn_resized = (attn_resized - attn_resized.min()) / \
                   (attn_resized.max() - attn_resized.min() + 1e-8)
    
    # Create figure with 2 panels only
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Overlay only
    axes[1].imshow(img_array)
    axes[1].imshow(attn_resized, cmap='jet', alpha=0.5)
    axes[1].set_title('Attention Overlay', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Add text info
    match_color = 'green' if prediction == ground_truth else 'red'
    info_text = f" Q: {question_text} \n Prediction: {prediction} | Ground Truth: {ground_truth}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             color=match_color, weight='bold')
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"Attention visualization saved to {save_path}")  # Removed verbose output
    else:
        plt.show()
    
    plt.close()
