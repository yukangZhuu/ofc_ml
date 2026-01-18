import sys
import numpy as np
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ofc_ml.data import load_data
from ofc_ml.features import preprocess_features
from ofc_ml.model import train_model
from ofc_ml.utils import create_submission
from ofc_ml.network import compute_baseline_gain
from ofc_ml import config as cfg

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-use", choices=["kaggle", "cosmos", "both"], default=None)
    return p.parse_args()

def main():
    args = parse_args()
    if args.dataset_use is not None:
        cfg.DATASET_USE = args.dataset_use

    train_features, train_labels, test_features = load_data()
    
    assert len(train_features) == len(train_labels)
    
    X_train, X_test, mask_cols, preprocessor = preprocess_features(train_features, test_features)
    
    target_cols = [c for c in train_labels.columns if 'calculated_gain_spectra_' in c]
    target_cols.sort()
    y_train = train_labels[target_cols].values
    
    model, metrics = train_model(X_train, y_train, preprocessor, train_features, mask_cols)
    
    print("Predicting on test set...")
    test_masks = test_features[mask_cols].values
    test_target_gain = test_features['target_gain'].values
    test_target_gain_tilt = test_features['target_gain_tilt'].values
    
    y_test_pred = model.predict(X_test, test_target_gain, test_target_gain_tilt, mask=test_masks)
    
    print(f"Final prediction shape: {y_test_pred.shape}")
    
    create_submission(y_test_pred, test_features, target_cols)

if __name__ == "__main__":
    main()
