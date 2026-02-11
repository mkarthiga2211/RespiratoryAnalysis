"""
Training and Evaluation Module for Respiratory Sound Classification
Implements training pipeline with 5-fold cross-validation and early stopping.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from models import create_model, OptimizedCNN, TransferLearningCNN
from grey_wolf_optimizer import GreyWolfOptimizer, get_search_space


class ModelTrainer:
    """
    Model trainer with cross-validation and early stopping.
    """
    
    def __init__(self,
                 model_type: str,
                 input_shape: Tuple,
                 num_classes: int,
                 hyperparameters: Optional[Dict] = None,
                 results_dir: str = 'results'):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model ('optimized_cnn', 'transfer_cnn', 'rnn', 'lstm', 'cnn')
            input_shape: Input shape
            num_classes: Number of classes
            hyperparameters: Hyperparameters for optimized CNN
            results_dir: Directory to save results
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hyperparameters = hyperparameters
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Training history
        self.history = None
        self.model = None
        
    def create_model(self) -> keras.Model:
        """Create model instance."""
        return create_model(
            self.model_type,
            self.input_shape,
            self.num_classes,
            self.hyperparameters
        )
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32,
              use_early_stopping: bool = True,
              patience: int = 10) -> keras.callbacks.History:
        """
        Train model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            use_early_stopping: Whether to use early stopping
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        # Create model
        self.model = self.create_model()
        
        # Callbacks
        callbacks = []
        
        if use_early_stopping:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.results_dir, f'{self.model_type}_best.h5')
        checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Learning rate reduction
        lr_reducer = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        # Train model
        print(f"\nTraining {self.model_type}...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        
        if self.num_classes == 2:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            y_test_flat = y_test.flatten()
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_test_flat = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test_flat, y_pred),
            'precision': precision_score(y_test_flat, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test_flat, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test_flat, y_pred, average='weighted', zero_division=0)
        }
        
        # AUC-ROC
        if self.num_classes == 2:
            metrics['auc'] = roc_auc_score(y_test_flat, y_pred_prob.flatten())
        else:
            try:
                y_test_bin = label_binarize(y_test_flat, classes=range(self.num_classes))
                metrics['auc'] = roc_auc_score(y_test_bin, y_pred_prob, average='macro', multi_class='ovr')
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        if self.history is None:
            raise ValueError("Model not trained yet!")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray, 
                              class_names: List[str], save_path: str = None):
        """Plot confusion matrix."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        y_pred_prob = self.model.predict(X_test, verbose=0)
        
        if self.num_classes == 2:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            y_test_flat = y_test.flatten()
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_test_flat = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_test_flat, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save model."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load model."""
        self.model = keras.models.load_model(filepath)


class CrossValidator:
    """
    Cross-validation trainer with patient-level separation.
    """
    
    def __init__(self,
                 model_type: str,
                 input_shape: Tuple,
                 num_classes: int,
                 n_splits: int = 5,
                 results_dir: str = 'results'):
        """
        Initialize cross-validator.
        
        Args:
            model_type: Type of model
            input_shape: Input shape
            num_classes: Number of classes
            n_splits: Number of folds
            results_dir: Results directory
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.n_splits = n_splits
        self.results_dir = results_dir
        
        os.makedirs(results_dir, exist_ok=True)
        
        self.fold_results = []
        self.best_hyperparameters = None
    
    def run_cross_validation(self, 
                            X: np.ndarray, 
                            y: np.ndarray, 
                            groups: np.ndarray,
                            epochs: int = 50,
                            batch_size: int = 32) -> Dict:
        """
        Run cross-validation with patient-level separation.
        
        Args:
            X: Features
            y: Labels
            groups: Patient IDs for group separation
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Dictionary of aggregated results
        """
        group_kfold = GroupKFold(n_splits=self.n_splits)
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups)):
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{self.n_splits}")
            print(f"{'='*50}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            
            # Create and train model
            trainer = ModelTrainer(
                self.model_type,
                self.input_shape,
                self.num_classes,
                results_dir=os.path.join(self.results_dir, f'fold_{fold + 1}')
            )
            
            # Train
            trainer.train(X_train, y_train, X_val, y_val, 
                         epochs=epochs, batch_size=batch_size)
            
            # Evaluate
            metrics = trainer.evaluate(X_val, y_val)
            fold_metrics.append(metrics)
            
            print(f"\nFold {fold + 1} Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Save fold results
            self.fold_results.append({
                'fold': fold + 1,
                'metrics': metrics,
                'history': trainer.history.history
            })
        
        # Aggregate results
        aggregated = self._aggregate_results(fold_metrics)
        
        print(f"\n{'='*50}")
        print("Cross-Validation Results (Mean ± Std)")
        print(f"{'='*50}")
        for metric, (mean, std) in aggregated.items():
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
        
        # Save results
        self._save_results(aggregated, fold_metrics)
        
        return aggregated
    
    def _aggregate_results(self, fold_metrics: List[Dict]) -> Dict:
        """Aggregate results across folds."""
        aggregated = {}
        
        for metric in fold_metrics[0].keys():
            values = [fold[metric] for fold in fold_metrics]
            aggregated[metric] = (np.mean(values), np.std(values))
        
        return aggregated
    
    def _save_results(self, aggregated: Dict, fold_metrics: List[Dict]):
        """Save cross-validation results."""
        results = {
            'model_type': self.model_type,
            'n_splits': self.n_splits,
            'aggregated_results': aggregated,
            'fold_results': fold_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = os.path.join(self.results_dir, 'cross_validation_results.json')
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filepath}")


class GWOTrainer:
    """
    Trainer with Grey Wolf Optimizer for hyperparameter optimization.
    """
    
    def __init__(self,
                 input_shape: Tuple,
                 num_classes: int,
                 results_dir: str = 'results'):
        """
        Initialize GWO trainer.
        
        Args:
            input_shape: Input shape
            num_classes: Number of classes
            results_dir: Results directory
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.results_dir = results_dir
        
        os.makedirs(results_dir, exist_ok=True)
        
        self.best_hyperparameters = None
        self.gwo = None
    
    def optimize(self, 
                 X_train: np.ndarray, 
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 num_wolves: int = 20,
                 max_iterations: int = 50) -> Dict:
        """
        Run GWO optimization.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            num_wolves: Number of wolves
            max_iterations: Maximum iterations
            
        Returns:
            Best hyperparameters
        """
        # Get search space
        search_space = get_search_space()
        
        # Initialize GWO
        self.gwo = GreyWolfOptimizer(
            search_space=search_space,
            num_wolves=num_wolves,
            max_iterations=max_iterations
        )
        
        # Run optimization
        self.best_hyperparameters = self.gwo.optimize(
            X_train, y_train, X_val, y_val,
            self.input_shape, self.num_classes
        )
        
        # Save best hyperparameters
        filepath = os.path.join(self.results_dir, 'best_hyperparameters.json')
        with open(filepath, 'w') as f:
            json.dump(self.best_hyperparameters, f, indent=2)
        
        print(f"\nBest hyperparameters saved to {filepath}")
        
        return self.best_hyperparameters
    
    def train_with_best_params(self,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_val: np.ndarray,
                               y_val: np.ndarray,
                               epochs: int = 50) -> ModelTrainer:
        """
        Train model with best hyperparameters.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of epochs
            
        Returns:
            Trained ModelTrainer
        """
        if self.best_hyperparameters is None:
            raise ValueError("Run optimize() first!")
        
        trainer = ModelTrainer(
            'optimized_cnn',
            self.input_shape,
            self.num_classes,
            self.best_hyperparameters,
            results_dir=self.results_dir
        )
        
        trainer.train(X_train, y_train, X_val, y_val, epochs=epochs)
        
        return trainer


def compare_models(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   input_shape: Tuple,
                   num_classes: int,
                   results_dir: str = 'results') -> Dict:
    """
    Compare different models (RNN, LSTM, CNN, Optimized CNN).
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        input_shape: Input shape
        num_classes: Number of classes
        results_dir: Results directory
        
    Returns:
        Dictionary of comparison results
    """
    models = ['rnn', 'lstm', 'cnn', 'optimized_cnn']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*50}")
        
        trainer = ModelTrainer(
            model_type,
            input_shape,
            num_classes,
            results_dir=os.path.join(results_dir, model_type)
        )
        
        # Train
        trainer.train(X_train, y_train, X_val, y_val, epochs=50)
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)
        results[model_type] = metrics
        
        print(f"\n{model_type.upper()} Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Plot training history
        trainer.plot_training_history(
            save_path=os.path.join(results_dir, model_type, 'training_history.png')
        )
    
    # Save comparison results
    filepath = os.path.join(results_dir, 'model_comparison.json')
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    print("Training module ready!")
