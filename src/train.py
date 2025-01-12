import os
import argparse
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from feature_extraction import GearboxFeatureExtractor
from model import GearboxAnomalyDetector

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def prepare_data(data_dir: str, extractor: GearboxFeatureExtractor):
    """Prepare training data."""
    train_dir = Path(data_dir) / "gearbox" / "train"
    train_files = list(train_dir.glob("normal_*.wav"))
    
    logging.info(f"Found {len(train_files)} training files")
    
    # Extract features
    features = []
    for file in tqdm(train_files, desc="Extracting features"):
        feature = extractor.extract_features(str(file))
        features.append(feature)
    
    features = np.vstack(features)
    logging.info(f"Extracted features shape: {features.shape}")
    
    # Split train/validation
    np.random.shuffle(features)
    split_idx = int(len(features) * 0.9)
    train_features = features[:split_idx]
    val_features = features[split_idx:]
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_features))
    train_dataset = train_dataset.shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_features))
    val_dataset = val_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, train_features

def main(args):
    """Main training function."""
    setup_logging()
    logging.info("Starting training process")
    
    # Initialize feature extractor
    extractor = GearboxFeatureExtractor()
    
    # Prepare data
    train_dataset, val_dataset, train_features = prepare_data(
        args.data_dir,
        extractor
    )
    
    # Create and train model
    model = GearboxAnomalyDetector(extractor.get_feature_dim())
    history = model.train(train_dataset, val_dataset, args.epochs)
    
    # Calculate threshold
    threshold = model.get_threshold(train_features)
    
    # Convert to TFLite
    tflite_model = model.convert_to_tflite()
    
    # Save everything
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(output_dir / "model.keras")
    np.save(output_dir / "threshold.npy", threshold)
    
    with open(output_dir / "model.tflite", "wb") as f:
        f.write(tflite_model)
    
    logging.info(f"Model saved to {output_dir}")
    logging.info(f"Final model size: {len(tflite_model) / 1024:.1f} KB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gearbox anomaly detector")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output-dir", type=str, default="models/gearbox", 
                      help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    
    args = parser.parse_args()
    main(args)