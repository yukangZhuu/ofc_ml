import pandas as pd
from pathlib import Path
from . import config as cfg

def _load_cosmos_dataset():
    """
    Load cosmos training dataset.

    Supports two layouts:
    1) cfg.COSMOS_TRAIN_FEATURES_PATH / cfg.COSMOS_TRAIN_LABELS_PATH exist as files
    2) cfg.COSMOS_DATA_DIR contains multiple subfolders with train_features.csv/train_labels.csv
    """
    if Path(cfg.COSMOS_TRAIN_FEATURES_PATH).exists() and Path(cfg.COSMOS_TRAIN_LABELS_PATH).exists():
        print(f"Loading COSMOS train from {cfg.COSMOS_TRAIN_FEATURES_PATH.parent}...")
        X = pd.read_csv(cfg.COSMOS_TRAIN_FEATURES_PATH)
        y = pd.read_csv(cfg.COSMOS_TRAIN_LABELS_PATH)
        return X, y

    root = Path(cfg.COSMOS_DATA_DIR)
    feat_files = sorted(root.glob("**/train_features.csv"))
    label_files = sorted(root.glob("**/train_labels.csv"))
    if not feat_files or not label_files:
        raise FileNotFoundError(
            f"Could not find cosmos train csvs under {root}. "
            "Expected train_features.csv/train_labels.csv either at root or in subfolders."
        )

    # Index labels by parent dir so we pair correctly
    labels_by_parent = {p.parent: p for p in label_files}
    pairs = []
    for f in feat_files:
        l = labels_by_parent.get(f.parent)
        if l is None:
            raise FileNotFoundError(f"Missing train_labels.csv for {f.parent}")
        pairs.append((f, l))

    print(f"Loading COSMOS train from {root} ({len(pairs)} shards)...")
    Xs = []
    ys = []
    base_X_cols = None
    base_y_cols = None
    for f, l in pairs:
        Xi = pd.read_csv(f)
        yi = pd.read_csv(l)
        if base_X_cols is None:
            base_X_cols = list(Xi.columns)
            base_y_cols = list(yi.columns)
        else:
            if list(Xi.columns) != base_X_cols:
                raise ValueError(f"train_features schema mismatch in {f}")
            if list(yi.columns) != base_y_cols:
                raise ValueError(f"train_labels schema mismatch in {l}")
        Xs.append(Xi)
        ys.append(yi)

    X = pd.concat(Xs, axis=0, ignore_index=True)
    y = pd.concat(ys, axis=0, ignore_index=True)
    return X, y

def load_data():
    """
    Load training and test data from CSV files.
    
    Returns:
        tuple: (train_features, train_labels, test_features) as pandas DataFrames.
    """
    try:
        dataset_use = str(cfg.DATASET_USE).lower().strip()
        if dataset_use not in {"kaggle", "cosmos", "both"}:
            raise ValueError(f"Unsupported DATASET_USE={cfg.DATASET_USE!r}. Use 'kaggle', 'cosmos', or 'both'.")

        if dataset_use == "kaggle":
            print(f"Loading Kaggle train from {cfg.TRAIN_FEATURES_PATH.parent}...")
            train_features = pd.read_csv(cfg.TRAIN_FEATURES_PATH)
            train_labels = pd.read_csv(cfg.TRAIN_LABELS_PATH)
        elif dataset_use == "cosmos":
            train_features, train_labels = _load_cosmos_dataset()
        else:
            print(f"Loading BOTH train datasets (kaggle + cosmos)...")
            kaggle_features = pd.read_csv(cfg.TRAIN_FEATURES_PATH)
            kaggle_labels = pd.read_csv(cfg.TRAIN_LABELS_PATH)
            cosmos_features, cosmos_labels = _load_cosmos_dataset()

            # Ensure schemas match; if not, fail loudly.
            if list(kaggle_features.columns) != list(cosmos_features.columns):
                raise ValueError("train_features schema mismatch between kaggle and cosmos datasets.")
            if list(kaggle_labels.columns) != list(cosmos_labels.columns):
                raise ValueError("train_labels schema mismatch between kaggle and cosmos datasets.")

            train_features = pd.concat([kaggle_features, cosmos_features], axis=0, ignore_index=True)
            train_labels = pd.concat([kaggle_labels, cosmos_labels], axis=0, ignore_index=True)

        # Test features always come from Kaggle test file (cosmos dataset has no test set here)
        test_features = pd.read_csv(cfg.TEST_FEATURES_PATH)
        
        print(f"Train features shape: {train_features.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Test features shape: {test_features.shape}")
        
        return train_features, train_labels, test_features
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise