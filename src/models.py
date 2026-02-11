"""
CNN Models for Respiratory Sound Classification
Implements: Optimized CNN, Transfer Learning with Attention Mechanism
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class AttentionLayer(layers.Layer):
    """
    Attention mechanism for focusing on relevant time-frequency regions.
    Implements the attention technique described in the paper.
    """
    
    def __init__(self, units: int = 128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_W',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_b',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_u',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Calculate attention scores
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        
        # Apply attention weights
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        
        return context_vector, attention_weights
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


class OptimizedCNN:
    """
    Optimized CNN for respiratory sound classification.
    Architecture based on hyperparameters optimized by GWO.
    """
    
    def __init__(self, 
                 input_shape: Tuple,
                 num_classes: int,
                 hyperparameters: Optional[Dict] = None):
        """
        Initialize Optimized CNN.
        
        Args:
            input_shape: Shape of input spectrograms (height, width, channels)
            num_classes: Number of output classes
            hyperparameters: Dictionary of optimized hyperparameters
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hyperparameters = hyperparameters or self._default_hyperparameters()
        self.model = None
        
    def _default_hyperparameters(self) -> Dict:
        """Default hyperparameters from Table 2 of the paper."""
        return {
            'conv_filters': 64,
            'kernel_size': 3,
            'dropout_rate': 0.3,
            'dense_units': 256,
            'learning_rate': 0.001
        }
    
    def build_model(self, use_attention: bool = True) -> Model:
        """
        Build the optimized CNN architecture.
        
        Args:
            use_attention: Whether to use attention mechanism
            
        Returns:
            Compiled Keras model
        """
        hp = self.hyperparameters
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # First Convolutional Block
        x = layers.Conv2D(
            filters=hp['conv_filters'],
            kernel_size=(hp['kernel_size'], hp['kernel_size']),
            activation='relu',
            padding='same',
            name='conv1'
        )(inputs)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
        
        # Second Convolutional Block
        x = layers.Conv2D(
            filters=hp['conv_filters'] * 2,
            kernel_size=(hp['kernel_size'], hp['kernel_size']),
            activation='relu',
            padding='same',
            name='conv2'
        )(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
        
        # Third Convolutional Block
        x = layers.Conv2D(
            filters=hp['conv_filters'] * 4,
            kernel_size=(hp['kernel_size'], hp['kernel_size']),
            activation='relu',
            padding='same',
            name='conv3'
        )(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
        
        # Fourth Convolutional Block (optional, for deeper feature extraction)
        x = layers.Conv2D(
            filters=hp['conv_filters'] * 4,
            kernel_size=(hp['kernel_size'], hp['kernel_size']),
            activation='relu',
            padding='same',
            name='conv4'
        )(x)
        x = layers.BatchNormalization(name='bn4')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool4')(x)
        
        # Store shape before flatten for attention
        conv_shape = x.shape
        
        if use_attention:
            # Apply attention mechanism
            # Reshape for attention: (batch, time, features)
            x_att = layers.Reshape((conv_shape[1] * conv_shape[2], conv_shape[3]), name='reshape_att')(x)
            x_att, attention_weights = AttentionLayer(units=128, name='attention')(x_att)
        else:
            # Global Average Pooling
            x_att = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Dense layers with dropout
        x_dense = layers.Dense(hp['dense_units'], activation='relu', name='dense1')(x_att)
        x_dense = layers.BatchNormalization(name='bn_dense')(x_dense)
        x_dense = layers.Dropout(hp['dropout_rate'], name='dropout')(x_dense)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x_dense)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x_dense)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='Optimized_CNN')
        
        # Compile model
        optimizer = keras.optimizers.Adam(
            learning_rate=hp['learning_rate'],
            weight_decay=1e-5
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return self.model
    
    def get_model_summary(self):
        """Print model summary."""
        if self.model is None:
            self.build_model()
        return self.model.summary()


class TransferLearningCNN:
    """
    Transfer Learning CNN using pre-trained VGGish or similar audio model.
    Implements transfer learning with attention as described in the paper.
    """
    
    def __init__(self,
                 input_shape: Tuple,
                 num_classes: int,
                 base_model_name: str = 'vgg',
                 fine_tune_at: int = None):
        """
        Initialize Transfer Learning CNN.
        
        Args:
            input_shape: Input shape
            num_classes: Number of classes
            base_model_name: Name of pre-trained model to use
            fine_tune_at: Layer index to start fine-tuning from
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model_name = base_model_name
        self.fine_tune_at = fine_tune_at
        self.base_model = None
        self.model = None
        
    def _create_base_model(self) -> Model:
        """Create base model (VGG-like architecture for audio)."""
        inputs = layers.Input(shape=self.input_shape)
        
        # VGG-like architecture
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        return Model(inputs, x, name='vgg_base')
    
    def build_model(self, use_attention: bool = True, 
                    fine_tune_learning_rate: float = 1e-4) -> Model:
        """
        Build transfer learning model.
        
        Args:
            use_attention: Whether to use attention mechanism
            fine_tune_learning_rate: Learning rate for fine-tuning
            
        Returns:
            Compiled Keras model
        """
        # Create base model
        self.base_model = self._create_base_model()
        
        # Freeze base model layers initially
        self.base_model.trainable = False
        
        # Build classification head
        inputs = layers.Input(shape=self.input_shape)
        x = self.base_model(inputs, training=False)
        
        # Store shape for attention
        conv_shape = x.shape
        
        if use_attention:
            # Apply attention mechanism
            x_att = layers.Reshape((conv_shape[1] * conv_shape[2], conv_shape[3]))(x)
            x_att, _ = AttentionLayer(units=256)(x_att)
        else:
            x_att = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x_dense = layers.Dense(512, activation='relu')(x_att)
        x_dense = layers.BatchNormalization()(x_dense)
        x_dense = layers.Dropout(0.3)(x_dense)
        
        x_dense = layers.Dense(256, activation='relu')(x_dense)
        x_dense = layers.BatchNormalization()(x_dense)
        x_dense = layers.Dropout(0.3)(x_dense)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x_dense)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x_dense)
            loss = 'categorical_crossentropy'
        
        self.model = Model(inputs, outputs, name='Transfer_Learning_CNN')
        
        # Compile with lower learning rate for transfer learning
        optimizer = keras.optimizers.Adam(learning_rate=fine_tune_learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return self.model
    
    def unfreeze_for_fine_tuning(self, fine_tune_at: int = None, 
                                  learning_rate: float = 1e-4):
        """
        Unfreeze layers for fine-tuning.
        
        Args:
            fine_tune_at: Layer index to start fine-tuning from
            learning_rate: Learning rate for fine-tuning
        """
        if fine_tune_at is None:
            fine_tune_at = len(self.base_model.layers) // 2
        
        # Unfreeze layers from fine_tune_at onwards
        for layer in self.base_model.layers[fine_tune_at:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=self.model.loss,
            metrics=['accuracy']
        )


class BaselineModels:
    """
    Baseline models for comparison: RNN, LSTM, CNN
    """
    
    @staticmethod
    def build_rnn(input_shape: Tuple, num_classes: int) -> Model:
        """
        Build RNN baseline model.
        
        Args:
            input_shape: Input shape (time_steps, features)
            num_classes: Number of classes
            
        Returns:
            Compiled RNN model
        """
        model = keras.Sequential([
            layers.SimpleRNN(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.SimpleRNN(64),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes if num_classes > 2 else 1, 
                        activation='softmax' if num_classes > 2 else 'sigmoid')
        ], name='RNN_Baseline')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def build_lstm(input_shape: Tuple, num_classes: int) -> Model:
        """
        Build LSTM baseline model.
        
        Args:
            input_shape: Input shape (time_steps, features)
            num_classes: Number of classes
            
        Returns:
            Compiled LSTM model
        """
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes if num_classes > 2 else 1,
                        activation='softmax' if num_classes > 2 else 'sigmoid')
        ], name='LSTM_Baseline')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def build_simple_cnn(input_shape: Tuple, num_classes: int) -> Model:
        """
        Build simple CNN baseline model.
        
        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of classes
            
        Returns:
            Compiled CNN model
        """
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes if num_classes > 2 else 1,
                        activation='softmax' if num_classes > 2 else 'sigmoid')
        ], name='CNN_Baseline')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model


def create_model(model_type: str,
                 input_shape: Tuple,
                 num_classes: int,
                 hyperparameters: Optional[Dict] = None) -> Model:
    """
    Factory function to create different model types.
    
    Args:
        model_type: Type of model ('optimized_cnn', 'transfer_cnn', 'rnn', 'lstm', 'cnn')
        input_shape: Input shape
        num_classes: Number of classes
        hyperparameters: Optional hyperparameters for optimized CNN
        
    Returns:
        Compiled Keras model
    """
    if model_type == 'optimized_cnn':
        model_builder = OptimizedCNN(input_shape, num_classes, hyperparameters)
        return model_builder.build_model(use_attention=True)
    
    elif model_type == 'transfer_cnn':
        model_builder = TransferLearningCNN(input_shape, num_classes)
        return model_builder.build_model(use_attention=True)
    
    elif model_type == 'rnn':
        return BaselineModels.build_rnn(input_shape, num_classes)
    
    elif model_type == 'lstm':
        return BaselineModels.build_lstm(input_shape, num_classes)
    
    elif model_type == 'cnn':
        return BaselineModels.build_simple_cnn(input_shape, num_classes)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test Optimized CNN
    input_shape = (40, 40, 1)  # Mel-spectrogram shape
    num_classes = 6
    
    print("\n1. Testing Optimized CNN:")
    opt_cnn = OptimizedCNN(input_shape, num_classes)
    model = opt_cnn.build_model(use_attention=True)
    opt_cnn.get_model_summary()
    
    print("\n2. Testing Baseline CNN:")
    baseline_cnn = BaselineModels.build_simple_cnn(input_shape, num_classes)
    print(baseline_cnn.summary())
    
    print("\nAll models created successfully!")
