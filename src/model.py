import tensorflow as tf
import numpy as np

class GearboxAnomalyDetector:
    """Ultra-lightweight anomaly detector for Pico deployment."""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build tiny model for Pico."""
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.InputLayer(input_shape=(self.input_dim,)),
            tf.keras.layers.Lambda(lambda x: x * 0.125),  # Power-of-2 scaling
            
            # Minimal encoder (reduced from original)
            tf.keras.layers.Dense(32, activation='relu'),   # Reduced from 64
            
            # Tiny bottleneck
            tf.keras.layers.Dense(8, activation='relu'),    # Reduced from 16
            
            # Direct anomaly score output
            tf.keras.layers.Dense(1, activation='sigmoid')  # Changed to direct scoring
        ])
        
        # Use fixed learning rate for better quantization
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',  # Changed to match new output
            jit_compile=True
        )
        
        return model
    
    def train(self, 
         train_dataset: tf.data.Dataset,
         val_dataset: tf.data.Dataset,
         epochs: int = 50) -> tf.keras.callbacks.History:
        """Train with minimal callbacks."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,  # Reduced from 15
                restore_best_weights=True,
                min_delta=1e-4
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
        """Calculate simple threshold."""
        predictions = self.model.predict(normal_features, batch_size=32)
        
        # Simple percentile threshold
        threshold = np.percentile(predictions, 95)
        return float(threshold)
    
    def convert_to_tflite(self) -> bytes:
        """Convert to heavily optimized TFLite model."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Maximum optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        
        # Force INT8 quantization
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        def representative_dataset():
            for _ in range(100):  # Reduced from 500
                # Simpler synthetic data
                data = np.random.normal(0, 1, (1, self.input_dim))
                data = data.astype(np.float32)
                data = data / np.max(np.abs(data))
                yield [data]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_types = [tf.int8]
        converter._experimental_disable_per_channel = True
        
        return converter.convert()