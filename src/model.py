import tensorflow as tf
import numpy as np
from typing import Tuple

class GearboxAnomalyDetector:
    """Lightweight anomaly detector for gearbox sounds."""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build model optimized for M0+ architecture."""
        # Use power-of-2 layer sizes for efficient memory alignment
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.InputLayer(input_shape=(self.input_dim,)),
            # Use individual scaling/shifting instead of BatchNorm for M0+
            tf.keras.layers.Lambda(lambda x: x * 0.125),  # Power-of-2 scaling
            
            # Encoder optimized for integer math
            tf.keras.layers.Dense(64, activation='relu'),  # Power of 2
            tf.keras.layers.Dense(32, activation='relu'),  # Power of 2
            
            # Bottleneck with power-of-2 size
            tf.keras.layers.Dense(16, activation='relu'),
            
            # Decoder matching encoder
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.input_dim)
        ])
        
        # Use fixed learning rate for better quantization
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            jit_compile=True
        )
        
        return model
    
    def train(self, 
         train_dataset: tf.data.Dataset,
         val_dataset: tf.data.Dataset,
         epochs: int = 100) -> tf.keras.callbacks.History:
        """Train with simple, robust callbacks."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Power of 2 for better quantization
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        return self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    def get_threshold(self, normal_features: np.ndarray) -> float:
        """Calculate robust threshold with accuracy consideration."""
        predictions = self.model.predict(normal_features, batch_size=32)
        errors = np.mean(np.square(normal_features - predictions), axis=1)
        
        # Calculate multiple threshold candidates
        percentile_threshold = np.percentile(errors, 97)
        mad_threshold = np.median(errors) + (2.5 * np.median(
            np.abs(errors - np.median(errors))))
        
        # Use cross-validation to select best threshold
        def evaluate_threshold(threshold):
            predictions = errors > threshold
            # Assuming all training data is normal
            accuracy = np.mean(predictions == 0)
            return accuracy
        
        accuracies = [
            evaluate_threshold(percentile_threshold),
            evaluate_threshold(mad_threshold)
        ]
        
        # Return threshold with best accuracy
        best_threshold = [percentile_threshold, mad_threshold][
            np.argmax(accuracies)]
        
        return float(best_threshold)
    
    def save(self, path: str):
        """Save the model."""
        self.model.save(path)
    
    def convert_to_tflite(self) -> bytes:
        """Convert to M0+ optimized TFLite model."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimize for M0+ architecture
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        
        # Force fixed-point quantization for M0+
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        def representative_dataset():
            # Generate data matching expected signal characteristics
            for _ in range(500):  # More samples for better calibration
                # Generate base signal
                base = np.random.normal(0, 1, (1, self.input_dim))
                # Add typical gearbox harmonics
                harmonics = np.sin(np.linspace(0, 10, self.input_dim)) * 0.1
                # Add noise
                noise = np.random.normal(0, 0.05, (1, self.input_dim))
                
                data = (base + harmonics.reshape(1, -1) + noise).astype(np.float32)
                # Scale to expected range
                data = data / np.max(np.abs(data))
                yield [data]
        
        converter.representative_dataset = representative_dataset
        
        # Additional M0+ specific optimizations
        converter.target_spec.supported_types = [tf.int8]
        converter._experimental_disable_per_channel = True  # Force per-tensor quant
        
        return converter.convert()