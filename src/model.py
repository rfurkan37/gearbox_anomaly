import tensorflow as tf
import numpy as np
from typing import Tuple

class GearboxAnomalyDetector:
    """Lightweight anomaly detector for gearbox sounds."""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build optimized model architecture."""
        model = tf.keras.Sequential([
            # Input layer with normalization
            tf.keras.layers.InputLayer(input_shape=(self.input_dim,)),
            tf.keras.layers.BatchNormalization(),
            
            # Encoder - reduced size
            tf.keras.layers.Dense(64, activation='relu'),  # Reduced from 96
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),  # Reduced from 48
            tf.keras.layers.BatchNormalization(),
            
            # Bottleneck - kept same size for feature preservation
            tf.keras.layers.Dense(12, activation='relu'),
            
            # Decoder - reduced size
            tf.keras.layers.Dense(32, activation='relu'),  # Reduced from 48
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),  # Reduced from 96
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.input_dim)
        ])
        
        # Use mixed precision for training
        model.compile(
            optimizer=tf.keras.optimizers.Adam(5e-4),
            loss='mse',
            jit_compile=True  # Enable XLA compilation
        )
        
        return model
    
    def train(self, 
             train_dataset: tf.data.Dataset,
             val_dataset: tf.data.Dataset,
             epochs: int = 100) -> tf.keras.callbacks.History:
        """Train the model."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        return self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    def get_threshold(self, normal_features: np.ndarray, 
                 percentile: float = 97.0,
                 min_anomaly_ratio: float = 0.03) -> float:
        """Calculate anomaly threshold with safety checks.
        
        Args:
            normal_features: Features from normal training data
            percentile: Percentile to use for threshold (default: 97.0)
            min_anomaly_ratio: Minimum ratio of anomalies to expect (default: 0.03)
        
        Returns:
            float: Calculated threshold value
        """
        # Get reconstruction errors
        predictions = self.model.predict(normal_features)
        errors = np.mean(np.square(normal_features - predictions), axis=1)
        
        # Calculate basic threshold
        base_threshold = float(np.percentile(errors, percentile))
        
        # Safety checks
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        
        # Ensure threshold isn't too close to mean
        min_threshold = error_mean + (2 * error_std)
        
        # Calculate what percentage would be flagged as anomalies
        anomaly_ratio = np.mean(errors > base_threshold)
        
        if anomaly_ratio < min_anomaly_ratio:
            # Adjust threshold down if detecting too few anomalies
            sorted_errors = np.sort(errors)
            idx = int(len(errors) * (1 - min_anomaly_ratio))
            base_threshold = sorted_errors[idx]
        
        # Return maximum of base and minimum threshold
        return max(base_threshold, min_threshold)
    
    def save(self, path: str):
        """Save the model."""
        self.model.save(path)
    
    def convert_to_tflite(self) -> bytes:
        """Convert model to TFLite format with optimized quantization."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Enable all available optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        
        # Set input/output types to INT8
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Generate representative dataset from random normal distribution
        def representative_dataset():
            for _ in range(500):  # Increased from 100 to 500 for better statistics
                data = np.random.normal(0, 1, (1, self.input_dim)).astype(np.float32)
                # Add noise for better robustness
                data += np.random.normal(0, 0.01, data.shape).astype(np.float32)
                yield [data]
        
        # Set representative dataset
        converter.representative_dataset = representative_dataset
        
        # Force full integer quantization
        converter.target_spec.supported_types = [tf.int8]
        converter._experimental_disable_per_channel = True  # Force per-tensor quantization
        
        # Convert model
        tflite_model = converter.convert()
        
        return tflite_model