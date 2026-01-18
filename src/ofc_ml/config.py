from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "ofc-2026-ml-challenge"
TRAIN_FEATURES_PATH = DATA_DIR / "train_features_clean.csv"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels_clean.csv"
TEST_FEATURES_PATH = DATA_DIR / "test_features.csv"
EXAMPLE_SUBMISSION_PATH = DATA_DIR / "example_submission.csv"

# Dataset switch:
# - "kaggle": use TRAIN_FEATURES_PATH/TRAIN_LABELS_PATH (clean kaggle train)
# - "cosmos": use COSMOS_TRAIN_* as training data
# - "both": concatenate kaggle train + cosmos train for training
DATASET_USE = "kaggle"  # one of: {"kaggle", "cosmos", "both"}

# External cosmos-as-kaggle dataset (same schema as kaggle train)
# This can either be a directory containing train_features.csv/train_labels.csv directly,
# or a directory containing multiple subfolders each with those two files (e.g. by gain).
COSMOS_DATA_DIR = PROJECT_ROOT / "data" / "cosmos-as-kaggle-by-gain"
COSMOS_TRAIN_FEATURES_PATH = COSMOS_DATA_DIR / "train_features.csv"  # optional (may not exist)
COSMOS_TRAIN_LABELS_PATH = COSMOS_DATA_DIR / "train_labels.csv"      # optional (may not exist)

SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.05

WANDB_PROJECT = "ofc-2026-ml-challenge"
WANDB_ENTITY = None
WANDB_MODE = "online"

# Model selection
# - "mlp": SimpleGainPredictor (MLP)
# - "fourier_kan": FourierKANGainPredictor (FourierKAN blocks), typically deeper
MODEL_TYPE = "fourier_kan"

DROPOUT = 0.2
HIDDEN_DIMS = [128, 64]

# FourierKAN hyperparameters
FOURIER_KAN_DROPOUT = 0.2
# Deeper by default (you can tune)
FOURIER_KAN_HIDDEN_DIMS = [256, 256, 128, 128, 64]
# Number of Fourier frequencies used inside each FourierKANLayer
FOURIER_KAN_N_FREQUENCIES = 4
# Let the model see activation pattern (mask) by concatenating it to input features.
FOURIER_KAN_CONCAT_MASK_INPUT = True

# Backward-compatible aliases (old SimpleKAN naming)
SIMPLE_KAN_DROPOUT = FOURIER_KAN_DROPOUT
SIMPLE_KAN_HIDDEN_DIMS = FOURIER_KAN_HIDDEN_DIMS
SIMPLE_KAN_CONCAT_MASK_INPUT = FOURIER_KAN_CONCAT_MASK_INPUT
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
EARLY_STOPPING_PATIENCE = 60

# Training device: "cpu", "cuda", or "cuda:N" (e.g. "cuda:2")
DEVICE = "cuda:3"
