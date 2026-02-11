"""
Main Script for Respiratory Sound Classification
Complete implementation of the paper's methodology.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import to_categorical
import json
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from preprocessing import AudioPreprocessor, FeatureNormalizer, DataAugmenter
from dataset import load_dataset, RespiratoryDataset
from models import create_model, OptimizedCNN
from grey_wolf_optimizer import GreyWolfOptimizer, get_search_space
from train import ModelTrainer, CrossValidator, GWOTrainer, compare_models


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_data_for_task(X: np.ndarray, 
                          y: np.ndarray, 
                          groups: np.ndarray,
                          task: str = 'multiclass',
                          test_size: float = 0.2):
    """
    Prepare data for binary or multiclass classification.
    
    Args:
        X: Features
        y: Labels
        groups: Patient groups
        task: 'binary' or 'multiclass'
        test_size: Test set proportion
        
    Returns:
        Prepared data splits
    """
    # Convert labels to categorical for multiclass
    if task == 'multiclass':
        num_classes = len(np.unique(y))
        y_categorical = to_categorical(y, num_classes=num_classes)
    else:
        num_classes = 2
        y_categorical = y
    
    # Split data (patient-aware)
    unique_groups = np.unique(groups)
    n_test_groups = max(1, int(len(unique_groups) * test_size))
    
    np.random.shuffle(unique_groups)
    test_groups = unique_groups[:n_test_groups]
    train_groups = unique_groups[n_test_groups:]
    
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    
    X_train = X[train_mask]
    y_train = y_categorical[train_mask]
    X_test = X[test_mask]
    y_test = y_categorical[test_mask]
    
    # Further split training into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"\nData splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


def run_experiment(config: dict):
    """
    Run complete experiment based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("="*60)
    print("Respiratory Sound Classification - Paper Implementation")
    print("="*60)
    
    # Set random seeds
    set_random_seeds(config.get('seed', 42))
    
    # Load dataset
    print("\n" + "="*60)
    print("1. Loading Dataset")
    print("="*60)
    
    data_dir = config['data_dir']
    diagnosis_file = config.get('diagnosis_file')
    task = config.get('task', 'multiclass')
    target_shape = tuple(config.get('target_shape', [40, 40]))
    augment = config.get('augment', False)
    
    dataset, X, y, groups = load_dataset(
        data_dir=data_dir,
        diagnosis_file=diagnosis_file,
        task=task,
        target_shape=target_shape,
        augment=augment
    )
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = prepare_data_for_task(
        X, y, groups, task
    )
    
    input_shape = X_train.shape[1:]
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    results_dir = config.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Experiment mode
    mode = config.get('mode', 'full')
    
    if mode == 'preprocessing_only':
        print("\nPreprocessing completed. Exiting.")
        return
    
    # Hyperparameter optimization with GWO
    if mode in ['full', 'optimize_only', 'train_optimized']:
        print("\n" + "="*60)
        print("2. Hyperparameter Optimization (GWO)")
        print("="*60)
        
        gwo_trainer = GWOTrainer(
            input_shape=input_shape,
            num_classes=num_classes,
            results_dir=os.path.join(results_dir, 'gwo_optimization')
        )
        
        best_params = gwo_trainer.optimize(
            X_train, y_train, X_val, y_val,
            num_wolves=config.get('gwo_wolves', 20),
            max_iterations=config.get('gwo_iterations', 50)
        )
        
        if mode == 'optimize_only':
            print("\nOptimization completed. Exiting.")
            return
    else:
        best_params = None
    
    # Model comparison
    if mode in ['full', 'compare_models']:
        print("\n" + "="*60)
        print("3. Model Comparison (RNN, LSTM, CNN, Optimized CNN)")
        print("="*60)
        
        comparison_results = compare_models(
            X_train, y_train, X_val, y_val, X_test, y_test,
            input_shape, num_classes,
            results_dir=os.path.join(results_dir, 'model_comparison')
        )
        
        print("\nModel Comparison Results:")
        print("-" * 60)
        for model, metrics in comparison_results.items():
            print(f"\n{model.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    # Cross-validation
    if mode in ['full', 'cross_validation']:
        print("\n" + "="*60)
        print("4. Cross-Validation (5-Fold with Patient Separation)")
        print("="*60)
        
        cross_validator = CrossValidator(
            model_type='optimized_cnn',
            input_shape=input_shape,
            num_classes=num_classes,
            n_splits=config.get('n_splits', 5),
            results_dir=os.path.join(results_dir, 'cross_validation')
        )
        
        # Use best hyperparameters if available
        if best_params:
            cross_validator.best_hyperparameters = best_params
        
        cv_results = cross_validator.run_cross_validation(
            X, y, groups,
            epochs=config.get('epochs', 50),
            batch_size=config.get('batch_size', 32)
        )
    
    # Final training with best model
    if mode in ['full', 'train_final']:
        print("\n" + "="*60)
        print("5. Final Model Training")
        print("="*60)
        
        final_trainer = ModelTrainer(
            model_type='optimized_cnn',
            input_shape=input_shape,
            num_classes=num_classes,
            hyperparameters=best_params,
            results_dir=os.path.join(results_dir, 'final_model')
        )
        
        # Train on full training set
        final_trainer.train(
            np.concatenate([X_train, X_val]),
            np.concatenate([y_train, y_val]),
            X_test, y_test,
            epochs=config.get('epochs', 50),
            batch_size=config.get('batch_size', 32)
        )
        
        # Evaluate
        final_metrics = final_trainer.evaluate(X_test, y_test)
        
        print("\nFinal Model Results:")
        print("-" * 60)
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Plot results
        final_trainer.plot_training_history(
            save_path=os.path.join(results_dir, 'final_model', 'training_history.png')
        )
        
        final_trainer.plot_confusion_matrix(
            X_test, y_test, dataset.label_encoder.classes_,
            save_path=os.path.join(results_dir, 'final_model', 'confusion_matrix.png')
        )
        
        # Save model
        final_trainer.save_model(
            os.path.join(results_dir, 'final_model', 'optimized_cnn_model.h5')
        )
        
        # Save final results
        results = {
            'config': config,
            'final_metrics': final_metrics,
            'best_hyperparameters': best_params,
            'class_names': list(dataset.label_encoder.classes_)
        }
        
        with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Experiment Completed!")
    print(f"Results saved to: {results_dir}")
    print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Respiratory Sound Classification - Paper Implementation'
    )
    
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset directory')
    parser.add_argument('--diagnosis_file', type=str, default=None,
                       help='Path to diagnosis file')
    parser.add_argument('--mode', type=str, 
                       choices=['full', 'preprocessing_only', 'optimize_only', 
                               'train_optimized', 'compare_models', 
                               'cross_validation', 'train_final'],
                       default='full',
                       help='Experiment mode')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'],
                       default='multiclass',
                       help='Classification task')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Results directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--gwo_wolves', type=int, default=20,
                       help='Number of wolves for GWO')
    parser.add_argument('--gwo_iterations', type=int, default=50,
                       help='Number of iterations for GWO')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.diagnosis_file:
        config['diagnosis_file'] = args.diagnosis_file
    config['mode'] = args.mode
    config['task'] = args.task
    config['results_dir'] = args.results_dir
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['gwo_wolves'] = args.gwo_wolves
    config['gwo_iterations'] = args.gwo_iterations
    config['n_splits'] = args.n_splits
    config['augment'] = args.augment
    config['seed'] = args.seed
    config['target_shape'] = [40, 40]
    
    # Check required parameters
    if 'data_dir' not in config:
        print("Error: data_dir must be specified")
        sys.exit(1)
    
    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
