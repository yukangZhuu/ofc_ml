import pandas as pd
from .config import TRAIN_FEATURES_PATH, TRAIN_LABELS_PATH, TEST_FEATURES_PATH

def load_data():
    """
    Load training and test data from CSV files.
    
    Returns:
        tuple: (train_features, train_labels, test_features) as pandas DataFrames.
    """
    print(f"Loading data from {TRAIN_FEATURES_PATH.parent}...")
    try:
        train_features = pd.read_csv(TRAIN_FEATURES_PATH)
        train_labels = pd.read_csv(TRAIN_LABELS_PATH)
        test_features = pd.read_csv(TEST_FEATURES_PATH)
        
        print(f"Train features shape: {train_features.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Test features shape: {test_features.shape}")
        
        return train_features, train_labels, test_features
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise