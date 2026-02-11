"""
Grey Wolf Optimizer (GWO) for Hyperparameter Optimization
Implements the metaheuristic algorithm described in the paper.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Callable, Tuple, Any
import copy


class GreyWolfOptimizer:
    """
    Grey Wolf Optimizer for CNN hyperparameter optimization.
    Based on the social behavior and hunting tactics of grey wolves.
    """
    
    def __init__(self, 
                 search_space: Dict[str, Tuple],
                 num_wolves: int = 20,
                 max_iterations: int = 50,
                 objective_function: Callable = None):
        """
        Initialize GWO optimizer.
        
        Args:
            search_space: Dictionary defining hyperparameter search space
                         Format: {'param_name': (min, max, 'type')}
            num_wolves: Number of wolves in the population (20 as per paper)
            max_iterations: Maximum number of iterations (50 as per paper)
            objective_function: Function to evaluate model performance
        """
        self.search_space = search_space
        self.num_wolves = num_wolves
        self.max_iterations = max_iterations
        self.objective_function = objective_function
        
        # Initialize wolves population
        self.wolves = self._initialize_wolves()
        
        # Initialize alpha, beta, delta wolves (best three solutions)
        self.alpha_position = None
        self.alpha_score = float('-inf')
        self.beta_position = None
        self.beta_score = float('-inf')
        self.delta_position = None
        self.delta_score = float('-inf')
        
        # History for tracking convergence
        self.convergence_history = []
        
    def _initialize_wolves(self) -> np.ndarray:
        """
        Initialize wolf positions randomly within search space.
        
        Returns:
            Array of wolf positions
        """
        wolves = []
        for _ in range(self.num_wolves):
            wolf = {}
            for param_name, (min_val, max_val, param_type) in self.search_space.items():
                if param_type == 'int':
                    wolf[param_name] = np.random.randint(min_val, max_val + 1)
                elif param_type == 'float':
                    wolf[param_name] = np.random.uniform(min_val, max_val)
                elif param_type == 'choice':
                    wolf[param_name] = np.random.choice(min_val)  # min_val is list of choices
            wolves.append(wolf)
        return np.array(wolves)
    
    def _calculate_A_C(self, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate coefficient vectors A and C.
        
        A controls exploration and exploitation:
        - |A| > 1: exploration (searching globally)
        - |A| < 1: exploitation (searching locally)
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Tuple of (A, C) coefficient vectors
        """
        # a decreases linearly from 2 to 0
        a = 2 - iteration * (2 / self.max_iterations)
        
        # Random vectors r1 and r2 in [0, 1]
        r1 = np.random.random(len(self.search_space))
        r2 = np.random.random(len(self.search_space))
        
        # Calculate A = 2 * a * r1 - a
        A = 2 * a * r1 - a
        
        # Calculate C = 2 * r2
        C = 2 * r2
        
        return A, C
    
    def _calculate_distance(self, wolf_pos: Dict, leader_pos: Dict, C: np.ndarray) -> np.ndarray:
        """
        Calculate distance between wolf and leader (alpha/beta/delta).
        
        Args:
            wolf_pos: Current wolf position
            leader_pos: Leader position (alpha/beta/delta)
            C: Coefficient vector
            
        Returns:
            Distance vector
        """
        distance = []
        for i, (param_name, (_, _, param_type)) in enumerate(self.search_space.items()):
            if param_type in ['int', 'float']:
                dist = abs(C[i] * leader_pos[param_name] - wolf_pos[param_name])
                distance.append(dist)
        return np.array(distance)
    
    def _update_position(self, wolf_pos: Dict, alpha_pos: Dict, beta_pos: Dict, 
                         delta_pos: Dict, A_alpha: np.ndarray, C_alpha: np.ndarray,
                         A_beta: np.ndarray, C_beta: np.ndarray, 
                         A_delta: np.ndarray, C_delta: np.ndarray) -> Dict:
        """
        Update wolf position based on alpha, beta, and delta wolves.
        
        Args:
            wolf_pos: Current wolf position
            alpha_pos: Alpha wolf position (best solution)
            beta_pos: Beta wolf position (second best)
            delta_pos: Delta wolf position (third best)
            A_alpha, C_alpha: Coefficients for alpha
            A_beta, C_beta: Coefficients for beta
            A_delta, C_delta: Coefficients for delta
            
        Returns:
            Updated wolf position
        """
        new_position = {}
        
        for i, (param_name, (min_val, max_val, param_type)) in enumerate(self.search_space.items()):
            if param_type == 'choice':
                # For categorical parameters, randomly select from choices
                new_position[param_name] = np.random.choice(min_val)
            else:
                # Calculate distances to alpha, beta, delta
                D_alpha = abs(C_alpha[i] * alpha_pos[param_name] - wolf_pos[param_name])
                D_beta = abs(C_beta[i] * beta_pos[param_name] - wolf_pos[param_name])
                D_delta = abs(C_delta[i] * delta_pos[param_name] - wolf_pos[param_name])
                
                # Calculate positions towards alpha, beta, delta
                X1 = alpha_pos[param_name] - A_alpha[i] * D_alpha
                X2 = beta_pos[param_name] - A_beta[i] * D_beta
                X3 = delta_pos[param_name] - A_delta[i] * D_delta
                
                # New position is average of three positions
                new_value = (X1 + X2 + X3) / 3
                
                # Clip to search space bounds
                if param_type == 'int':
                    new_value = int(np.clip(round(new_value), min_val, max_val))
                else:
                    new_value = np.clip(new_value, min_val, max_val)
                
                new_position[param_name] = new_value
        
        return new_position
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 input_shape: Tuple, num_classes: int) -> Dict:
        """
        Run GWO optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            input_shape: Input shape for CNN
            num_classes: Number of output classes
            
        Returns:
            Best hyperparameters (alpha position)
        """
        print("Starting Grey Wolf Optimization...")
        print(f"Population size: {self.num_wolves}")
        print(f"Max iterations: {self.max_iterations}")
        
        # Evaluate initial population
        fitness_scores = []
        for wolf in self.wolves:
            score = self._evaluate_wolf(wolf, X_train, y_train, X_val, y_val, 
                                       input_shape, num_classes)
            fitness_scores.append(score)
        
        fitness_scores = np.array(fitness_scores)
        
        # Sort wolves by fitness and initialize alpha, beta, delta
        sorted_indices = np.argsort(fitness_scores)[::-1]
        self.alpha_position = copy.deepcopy(self.wolves[sorted_indices[0]])
        self.alpha_score = fitness_scores[sorted_indices[0]]
        self.beta_position = copy.deepcopy(self.wolves[sorted_indices[1]])
        self.beta_score = fitness_scores[sorted_indices[1]]
        self.delta_position = copy.deepcopy(self.wolves[sorted_indices[2]])
        self.delta_score = fitness_scores[sorted_indices[2]]
        
        print(f"Initial best score: {self.alpha_score:.4f}")
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Update each wolf position
            for i in range(self.num_wolves):
                # Calculate A and C for alpha, beta, delta
                A_alpha, C_alpha = self._calculate_A_C(iteration)
                A_beta, C_beta = self._calculate_A_C(iteration)
                A_delta, C_delta = self._calculate_A_C(iteration)
                
                # Update wolf position
                self.wolves[i] = self._update_position(
                    self.wolves[i], self.alpha_position, self.beta_position, 
                    self.delta_position, A_alpha, C_alpha, A_beta, C_beta, 
                    A_delta, C_delta
                )
                
                # Evaluate new position
                score = self._evaluate_wolf(
                    self.wolves[i], X_train, y_train, X_val, y_val,
                    input_shape, num_classes
                )
                
                # Update alpha, beta, delta if needed
                if score > self.alpha_score:
                    self.delta_position = copy.deepcopy(self.beta_position)
                    self.delta_score = self.beta_score
                    self.beta_position = copy.deepcopy(self.alpha_position)
                    self.beta_score = self.alpha_score
                    self.alpha_position = copy.deepcopy(self.wolves[i])
                    self.alpha_score = score
                elif score > self.beta_score:
                    self.delta_position = copy.deepcopy(self.beta_position)
                    self.delta_score = self.beta_score
                    self.beta_position = copy.deepcopy(self.wolves[i])
                    self.beta_score = score
                elif score > self.delta_score:
                    self.delta_position = copy.deepcopy(self.wolves[i])
                    self.delta_score = score
            
            # Record convergence
            self.convergence_history.append(self.alpha_score)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, "
                      f"Best Score: {self.alpha_score:.4f}")
        
        print(f"\nOptimization completed!")
        print(f"Best score: {self.alpha_score:.4f}")
        print(f"Best hyperparameters: {self.alpha_position}")
        
        return self.alpha_position
    
    def _evaluate_wolf(self, wolf: Dict, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       input_shape: Tuple, num_classes: int) -> float:
        """
        Evaluate a wolf (hyperparameter set) by training and evaluating a CNN.
        
        Args:
            wolf: Hyperparameter dictionary
            X_train, y_train: Training data
            X_val, y_val: Validation data
            input_shape: Input shape for CNN
            num_classes: Number of classes
            
        Returns:
            F1-score (higher is better)
        """
        try:
            # Build model with wolf's hyperparameters
            model = self._build_cnn(wolf, input_shape, num_classes)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=wolf.get('batch_size', 32),
                epochs=10,  # Short training for evaluation
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # Evaluate and return F1-score
            from sklearn.metrics import f1_score
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            # Clear session to free memory
            tf.keras.backend.clear_session()
            
            return f1
            
        except Exception as e:
            print(f"Error evaluating wolf: {e}")
            return 0.0
    
    def _build_cnn(self, params: Dict, input_shape: Tuple, num_classes: int) -> keras.Model:
        """
        Build CNN model with given hyperparameters.
        
        Args:
            params: Hyperparameter dictionary
            input_shape: Input shape
            num_classes: Number of classes
            
        Returns:
            Compiled CNN model
        """
        model = keras.Sequential()
        
        # First Conv block
        model.add(layers.Conv2D(
            filters=params.get('conv_filters', 64),
            kernel_size=(params.get('kernel_size', 3), params.get('kernel_size', 3)),
            activation='relu',
            input_shape=input_shape,
            padding='same'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
        # Second Conv block
        model.add(layers.Conv2D(
            filters=params.get('conv_filters', 64) * 2,
            kernel_size=(params.get('kernel_size', 3), params.get('kernel_size', 3)),
            activation='relu',
            padding='same'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
        # Third Conv block
        model.add(layers.Conv2D(
            filters=params.get('conv_filters', 64) * 4,
            kernel_size=(params.get('kernel_size', 3), params.get('kernel_size', 3)),
            activation='relu',
            padding='same'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
        # Flatten and Dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(params.get('dense_units', 256), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(params.get('dropout_rate', 0.3)))
        
        # Output layer
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001))
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return model


def get_search_space() -> Dict[str, Tuple]:
    """
    Get hyperparameter search space as defined in Table 2 of the paper.
    
    Returns:
        Dictionary defining search space for each hyperparameter
    """
    return {
        'learning_rate': (1e-5, 1e-2, 'float'),
        'batch_size': (32, 64, 'int'),  # Will be rounded to nearest valid value
        'conv_filters': (16, 128, 'int'),
        'kernel_size': (3, 7, 'int'),  # 3, 5, or 7
        'dropout_rate': (0.2, 0.5, 'float'),
        'dense_units': (64, 512, 'int')
    }


if __name__ == "__main__":
    # Test GWO
    print("Testing Grey Wolf Optimizer...")
    
    search_space = get_search_space()
    print("Search space:")
    for param, bounds in search_space.items():
        print(f"  {param}: {bounds}")
    
    gwo = GreyWolfOptimizer(
        search_space=search_space,
        num_wolves=5,  # Small number for testing
        max_iterations=3
    )
    
    print("\nGWO initialized successfully!")
