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
    """Prepare training data with flexible file pattern matching."""
    train_dir = Path(data_dir) / "gearbox" / "train"
    
    # Add debug information
    logging.info(f"Looking for training files in: {train_dir}")
    if not train_dir.exists():
        raise ValueError(f"Training directory not found: {train_dir}")
    
    # Get normal and anomaly files
    normal_files = [f for f in train_dir.glob("*.wav") if "normal" in f.name.lower()]
    anomaly_files = [f for f in train_dir.glob("*.wav") if "anomaly" in f.name.lower()]
    
    if not normal_files:
        raise ValueError("No normal training files found")
    
    logging.info(f"Found {len(normal_files)} normal and {len(anomaly_files)} anomaly files")
    
    # Extract features and create labels
    features = []
    labels = []
    
    # Process normal files
    for file in tqdm(normal_files, desc="Extracting normal features"):
        feature = extractor.extract_features(str(file))
        features.append(feature)
        labels.extend([0] * len(feature))  # 0 for normal
    
    # Process anomaly files if available
    if anomaly_files:
        for file in tqdm(anomaly_files, desc="Extracting anomaly features"):
            feature = extractor.extract_features(str(file))
            features.append(feature)
            labels.extend([1] * len(feature))  # 1 for anomaly
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    logging.info(f"Extracted features shape: {features.shape}")
    
    # Split train/validation
    indices = np.random.permutation(len(features))
    split_idx = int(len(features) * 0.9)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    val_features = features[val_idx]
    val_labels = labels[val_idx]
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, train_labels)
    ).shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_features, val_labels)
    ).batch(64).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, train_features[train_labels == 0]  # Only normal for threshold

def main(args):
    """Main training function."""
    setup_logging()
    logging.info("Starting training process")
    
    # Initialize feature extractor
    extractor = GearboxFeatureExtractor()
    
    # Prepare data
    train_dataset, val_dataset, normal_features = prepare_data(
        args.data_dir,
        extractor
    )
    
    # Create and train model
    model = GearboxAnomalyDetector(extractor.get_feature_dim())
    history = model.train(train_dataset, val_dataset, args.epochs)
    
    # Calculate threshold using only normal data
    threshold = model.get_threshold(normal_features)
    
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
    logging.info(f"Threshold value: {threshold:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gearbox anomaly detector")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output-dir", type=str, default="models/gearbox", 
                      help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    
    args = parser.parse_args()
    main(args)