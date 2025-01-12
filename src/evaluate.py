import argparse
import logging
import numpy as np
from pathlib import Path
from sklearn import metrics
from tqdm import tqdm
import tensorflow as tf

from feature_extraction import GearboxFeatureExtractor

def load_model_and_threshold(model_dir: str):
    """Load the trained model and threshold."""
    model = tf.keras.models.load_model(Path(model_dir) / "model.keras")
    threshold = np.load(Path(model_dir) / "threshold.npy")
    return model, threshold

def evaluate_file(file_path: str, model, extractor: GearboxFeatureExtractor) -> np.ndarray:
    """Calculate anomaly score for a file."""
    # Extract features
    features = extractor.extract_features(file_path)
    
    # Get predictions
    predictions = model.predict(features)
    
    # Calculate reconstruction error
    errors = np.mean(np.square(features - predictions), axis=1)
    
    # Return mean error as anomaly score
    return float(np.mean(errors))

def evaluate_model(model_dir: str, test_dir: str):
    """Evaluate model performance."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load model and threshold
    model, threshold = load_model_and_threshold(model_dir)
    extractor = GearboxFeatureExtractor()
    
    # Get test files
    test_dir = Path(test_dir)
    normal_files = list(test_dir.glob("normal_*.wav"))
    anomaly_files = list(test_dir.glob("anomaly_*.wav"))
    
    logger.info(f"Found {len(normal_files)} normal and {len(anomaly_files)} anomaly files")
    
    # Calculate scores
    scores = []
    labels = []
    
    # Process normal files
    for file in tqdm(normal_files, desc="Processing normal files"):
        score = evaluate_file(str(file), model, extractor)
        scores.append(score)
        labels.append(0)
    
    # Process anomaly files
    for file in tqdm(anomaly_files, desc="Processing anomaly files"):
        score = evaluate_file(str(file), model, extractor)
        scores.append(score)
        labels.append(1)
    
    scores = np.array(scores)
    labels = np.array(labels)
    predictions = scores > threshold
    
    # Calculate metrics
    results = {
        'auc': metrics.roc_auc_score(labels, scores),
        'accuracy': metrics.accuracy_score(labels, predictions),
        'precision': metrics.precision_score(labels, predictions),
        'recall': metrics.recall_score(labels, predictions),
        'f1': metrics.f1_score(labels, predictions)
    }
    
    # Print results
    logger.info("\nEvaluation Results:")
    for metric, value in results.items():
        logger.info(f"{metric.upper()}: {value:.3f}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate gearbox anomaly detector")
    parser.add_argument("--model-dir", type=str, required=True, help="Model directory")
    parser.add_argument("--test-dir", type=str, required=True, help="Test data directory")
    
    args = parser.parse_args()
    evaluate_model(args.model_dir, args.test_dir)