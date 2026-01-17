from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "ofc-2026-ml-challenge"
TRAIN_FEATURES_PATH = DATA_DIR / "train_features.csv"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.csv"
TEST_FEATURES_PATH = DATA_DIR / "test_features.csv"
EXAMPLE_SUBMISSION_PATH = DATA_DIR / "example_submission.csv"

SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

WANDB_PROJECT = "ofc-2026-ml-challenge"
WANDB_ENTITY = None
WANDB_MODE = "online"
