# OFC 2026 ML Challenge - EDFA Gain Spectrum Prediction

Machine learning pipeline for predicting EDFA gain spectra in the OFC 2026 ML Challenge.

## Quick Start

```bash
# Create conda environment
conda create -n ofc_ml python=3.12
conda activate ofc_ml

# Install dependencies
pip install -r requirements.txt

# Run training + generate a submission
python main.py
```

## Project Structure

```
ofc_ml/
├── src/ofc_ml/              # Source code
│   ├── config.py            # Configuration file
│   ├── network.py           # Neural network architectures
│   ├── model.py             # Training logic
│   ├── features.py          # Feature preprocessing
│   └── data.py              # Data loading
├── data/                     # Dataset directory
├── scripts/                  # Utility scripts
│   └── cosmos_to_kaggle.py   # COSMOS data converter
├── submissions/              # Generated submissions
├── main.py                   # Entry point
└── requirements.txt          # Dependencies
```

## Data Setup

### Kaggle Competition Data

Download the competition files from Kaggle and place them under:

- `data/ofc-2026-ml-challenge/`

Expected files: `train_features.csv`, `train_labels.csv`, `test_features.csv`

### COSMOS-EDFA-Dataset (Optional)

Clone the original dataset repo:

```bash
git clone https://github.com/functions-lab/COSMOS-EDFA-Dataset.git COSMOS-EDFA-Dataset
```

Convert COSMOS JSON to Kaggle-style CSV:

```bash
python scripts/cosmos_to_kaggle.py \
  --cosmos-dataset-dir COSMOS-EDFA-Dataset/dataset \
  --out-dir data/cosmos-as-kaggle \
  --category cosmos \
  --gains 18dB \
  --channel-types fix
```

Note: COSMOS does not contain Kaggle's `aging/shb/unseen` labels; `Category` is filled with a constant.

## Configuration

Edit `src/ofc_ml/config.py` to customize training. Configuration is organized into logical groups:

### FEATURE_CONFIG

Feature preprocessing settings:

```python
FEATURE_CONFIG = {
    "USE_MASK": "concat",  # Options: "none", "concat", "multiply"
}
```

- `"none"`: Do not use mask columns
- `"concat"`: Concatenate mask columns as input features
- `"multiply"`: Multiply mask with spectra features (physical channel switch meaning)

### DATASET_CONFIG

Data paths and dataset selection:

```python
DATA_DIR = PROJECT_ROOT / "data" / "ofc-2026-ml-challenge"
TRAIN_FEATURES_PATH = DATA_DIR / "train_features.csv"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.csv"
TEST_FEATURES_PATH = DATA_DIR / "test_features.csv"
COSMOS_DATA_DIR = PROJECT_ROOT / "data" / "cosmos-as-kaggle"
DATASET_USE = "kaggle"  # Options: "kaggle", "cosmos", "both"
RANDOM_STATE = 42
TEST_SIZE = 0.05
```

### MODEL_CONFIG

HybridFNOKANPredictor hyperparameters:

```python
MODEL_CONFIG = {
    "DROPOUT": 0.2,
    "HIDDEN_DIMS": [256, 256, 128, 128, 128],
    "N_FREQUENCIES": 4,
    "SPECTRAL_FREQ_RATIO": 0.5,
    "USE_SPECTRAL_MIXING": True,
}
```

### TRAINING_CONFIG

General training settings:

```python
TRAINING_CONFIG = {
    "DEVICE": "cuda",  # "cpu" | "cuda" | "cuda:N"
    "LOAD_PRETRAINED_MODEL": True,
    "PRETRAIN_MODEL_PATH": PROJECT_ROOT / "models" / "pretrained_model.pt",
}
```

### PRETRAIN_CONFIG

Stage 1: Pretraining on COSMOS dataset:

```python
PRETRAIN_CONFIG = {
    "LEARNING_RATE": 0.001,
    "WEIGHT_DECAY": 1e-4,
    "BATCH_SIZE": 64,
    "EPOCHS": 500,
    "EARLY_STOPPING_PATIENCE": 50,
}
```

### FINETUNE_CONFIG

Stage 2: Fine-tuning on Kaggle dataset:

```python
FINETUNE_CONFIG = {
    "LEARNING_RATE": 0.0002,
    "WEIGHT_DECAY": 5e-5,
    "BATCH_SIZE": 32,
    "EPOCHS": 1000,
    "EARLY_STOPPING_PATIENCE": 50,
    "DISCRIMINATIVE_LR_DECAY": 0.95,
}
```

## Model Architecture

The project uses **HybridFNOKANPredictor**, which combines:

- **FourierKAN**: Kolmogorov-Arnold Networks with Fourier basis functions
- **Spectral Mixing**: Frequency domain mixing layers to capture global dependencies

### Architecture Structure

```
Input (variable dimension based on USE_MASK mode)
    ↓
[Early] FourierKAN Block × 2
    ↓
SpectralMixingLayer (FNO lightweight version)
    ↓ Frequency domain global mixing
    ↓
[Late] FourierKAN Block × 3
    ↓
Output Layer (95 channels)
```

### Key Components

- **SpectralMixingLayer**: Lightweight frequency domain mixing layer that captures global channel dependencies
- **FourierKANLayer**: Basic KAN layer with Fourier basis
- **FourierKANBlock**: Complete block with normalization and activation

## Two-Stage Training

The framework implements a two-stage transfer learning strategy:

### Stage 1: Pretraining (COSMOS)
- Train on COSMOS dataset
- Larger learning rate (0.001)
- Larger batch size (64)
- Save model to `models/pretrained_model.pt`

### Stage 2: Fine-tuning (Kaggle)
- Load pretrained model
- Fine-tune on Kaggle dataset
- Smaller learning rate (0.0002)
- Smaller batch size (32)
- Discriminative learning rates for different layers

### Training Flow

```
Check LOAD_PRETRAINED_MODEL and pretrained_model.pt existence
    ↓
    ├─→ Load model (skip Stage 1)
    │
    └─→ Stage 1: Pretrain on COSMOS
           ↓
           Save pretrained model
           ↓
           Stage 2: Finetune on Kaggle
              ↓
              Generate predictions
```

### Using Pretrained Model

If `models/pretrained_model.pt` exists and `LOAD_PRETRAINED_MODEL=True`, Stage 1 is skipped:

```bash
# Use existing pretrained model
python main.py

# Force retraining from scratch
python main.py --no-load-pretrained
```

## Dataset Selection

Choose dataset by setting `DATASET_USE`:

- `"kaggle"`: Use only Kaggle competition data
- `"cosmos"`: Use only COSMOS dataset
- `"both"`: Use both datasets (concatenated, recommended for two-stage training)

## Expected Performance

- Validation MSE: ~0.003-0.004
- Validation RMSE: ~0.05-0.06
- Prediction mean: ~17-18 dB

## Output

After training, you'll find:

- `models/pretrained_model.pt` - Pretrained model checkpoint
- `submissions/submission_YYYYMMDD_HHMMSS.csv` - Test predictions

## Kaggle Submission

1. Find latest submission: `ls -lt submissions/`
2. Upload to [Kaggle](https://www.kaggle.com/competitions/ofc-2026-ml-challenge/submissions)
3. View results on leaderboard

## Troubleshooting

**Import error?** Run from project root: `python main.py`

**Out of memory?** Reduce batch size in `config.py`

**Slow training?** Use GPU via `DEVICE="cuda:N"` in `config.py`

**Poor predictions?** Verify masked loss is used and check data preprocessing

## Requirements

- Python 3.12+
- PyTorch, scikit-learn, pandas, numpy

See `requirements.txt` for versions.
