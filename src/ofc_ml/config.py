from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data" / "ofc-2026-ml-challenge"
TRAIN_FEATURES_PATH = DATA_DIR / "train_features.txt"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.txt"
TEST_FEATURES_PATH = DATA_DIR / "test_features.txt"
EXAMPLE_SUBMISSION_PATH = DATA_DIR / "example_submission.txt"

# Output Paths
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

# Model Config
RANDOM_STATE = 42
TEST_SIZE = 0.2
