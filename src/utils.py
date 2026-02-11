"""
Utility functions for visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from typing import List, Dict, Tuple, Optional
import os


def plot_roc_curve(y_true: np.ndarray, 
                   y_pred_prob: np.ndarray, 
                   num_classes: int,
                   class_names: List[str],
                   save_path: str = None):
    """
    Plot ROC curve for multi-class classification.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        num_classes: Number of classes
        class_names: List of class names
        save_path: Path to save figure
    """
    if num_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
    else:
        # Multi-class classification
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        plt.figure(figsize=(10, 8))
        
        # Compute micro-average ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_prob.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        plt.plot(fpr_micro, tpr_micro, 'k--', lw=2,
                label=f'Micro-average ROC (AUC = {roc_auc_micro:.2f})')
        
        # Compute ROC curve for each class
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        
        for i, color in zip(range(num_classes), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curve')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_precision_recall_curve(y_true: np.ndarray,
                                y_pred_prob: np.ndarray,
                                num_classes: int,
                                class_names: List[str],
                                save_path: str = None):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        num_classes: Number of classes
        class_names: List of class names
        save_path: Path to save figure
    """
    if num_classes == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
    else:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        
        for i, color in zip(range(num_classes), colors):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
            plt.plot(recall, precision, color=color, lw=2, label=class_names[i])
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Multi-class Precision-Recall Curve')
        plt.legend(loc='lower left', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(feature_names: List[str],
                           importance_scores: np.ndarray,
                           top_n: int = 20,
                           save_path: str = None):
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importance_scores: Feature importance scores
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    # Sort features by importance
    indices = np.argsort(importance_scores)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance_scores[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_class_distribution(labels: List[str], 
                           class_names: List[str],
                           save_path: str = None):
    """
    Plot class distribution.
    
    Args:
        labels: List of labels
        class_names: List of class names
        save_path: Path to save figure
    """
    from collections import Counter
    
    counts = Counter(labels)
    classes = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, values, color='steelblue', edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_spectrogram_comparison(original: np.ndarray,
                                processed: np.ndarray,
                                title1: str = 'Original',
                                title2: str = 'Processed',
                                save_path: str = None):
    """
    Compare original and processed spectrograms.
    
    Args:
        original: Original spectrogram
        processed: Processed spectrogram
        title1: Title for original
        title2: Title for processed
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im1 = axes[0].imshow(original, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(title1)
    axes[0].set_xlabel('Time Frames')
    axes[0].set_ylabel('Frequency Bins')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(processed, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(title2)
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Frequency Bins')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_attention_weights(attention_weights: np.ndarray,
                          spectrogram: np.ndarray,
                          save_path: str = None):
    """
    Visualize attention weights overlaid on spectrogram.
    
    Args:
        attention_weights: Attention weights
        spectrogram: Input spectrogram
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Spectrogram
    im1 = axes[0].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Mel-Spectrogram')
    axes[0].set_xlabel('Time Frames')
    axes[0].set_ylabel('Frequency Bins')
    plt.colorbar(im1, ax=axes[0])
    
    # Attention weights
    im2 = axes[1].imshow(attention_weights, aspect='auto', origin='lower', cmap='hot')
    axes[1].set_title('Attention Weights')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Features')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_gwo_convergence(convergence_history: List[float],
                         save_path: str = None):
    """
    Plot GWO convergence curve.
    
    Args:
        convergence_history: List of best scores per iteration
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(convergence_history) + 1), convergence_history, 
            marker='o', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Best F1-Score')
    plt.title('Grey Wolf Optimizer Convergence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(results: Dict[str, Dict[str, float]],
                         metrics: List[str] = None,
                         save_path: str = None):
    """
    Plot comparison of multiple models.
    
    Args:
        results: Dictionary of model results
        metrics: List of metrics to compare
        save_path: Path to save figure
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [results[model].get(m, 0) for m in metrics]
        ax.bar(x + i*width, values, width, label=model.upper(), color=color)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_results_table(results: Dict[str, Dict[str, float]],
                        save_path: str = None) -> str:
    """
    Create a formatted results table.
    
    Args:
        results: Dictionary of model results
        save_path: Path to save table
        
    Returns:
        Formatted table string
    """
    # Create table header
    models = list(results.keys())
    metrics = list(results[models[0]].keys())
    
    # Create table
    table = "| Model | " + " | ".join([m.replace('_', ' ').title() for m in metrics]) + " |\n"
    table += "|" + "---|" * (len(metrics) + 1) + "\n"
    
    for model in models:
        row = f"| {model.upper()} |"
        for metric in metrics:
            value = results[model].get(metric, 0)
            row += f" {value:.4f} |"
        table += row + "\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table)
    
    return table


def save_training_plots(history: Dict,
                       save_dir: str):
    """
    Save all training plots.
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
