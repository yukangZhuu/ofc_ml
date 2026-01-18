# OFC 2026 ML Challenge - EDFA Gain Spectrum Prediction

Machine learning pipeline for predicting EDFA gain spectra in the OFC 2026 ML Challenge.

## Data setup (from scratch)

### Kaggle competition data

Download the competition files from Kaggle and place them under:

- `data/ofc-2026-ml-challenge/`

(This repo expects the usual files like `train_features.csv`, `train_labels.csv`, `test_features.csv`.)

### COSMOS-EDFA-Dataset (optional, external; not committed)

If you want to train with the COSMOS EDFA measurements, first clone the original dataset repo:

```bash
git clone https://github.com/functions-lab/COSMOS-EDFA-Dataset.git COSMOS-EDFA-Dataset
```

Then convert COSMOS JSON → Kaggle-shaped CSV:

```bash
python3 scripts/cosmos_to_kaggle.py \
  --cosmos-dataset-dir COSMOS-EDFA-Dataset/dataset \
  --out-dir data/cosmos-as-kaggle \
  --category cosmos \
  --gains 18dB \
  --channel-types fix
```

Notes:
- COSMOS does not contain Kaggle's `aging/shb/unseen` labels; `Category` is filled with a constant (default `cosmos`).
- The exporter computes `calculated_gain_spectra_*` as **output spectra − input spectra** per channel, following the COSMOS authors' extraction logic (see [functions-lab/COSMOS-EDFA-Dataset](https://github.com/functions-lab/COSMOS-EDFA-Dataset)).

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

That's it! The script will:

- Load data from `data/ofc-2026-ml-challenge/`
- Train the model
- Generate a timestamped submission in `submissions/`

## Convert COSMOS-EDFA-Dataset to Kaggle-style CSV (optional)

If you want to turn `COSMOS-EDFA-Dataset` (JSON measurements) into Kaggle-shaped CSV tables
(`train_features.csv` + `train_labels.csv`), run:

```bash
python3 /home/shaowen/ofc_ml/scripts/cosmos_to_kaggle.py \
  --cosmos-dataset-dir /home/shaowen/ofc_ml/COSMOS-EDFA-Dataset/dataset \
  --out-dir /home/shaowen/ofc_ml/data/cosmos-as-kaggle \
  --category cosmos \
  --gains 18dB \
  --channel-types fix
```

Notes:
- COSMOS does not contain Kaggle's `aging/shb/unseen` split labels, so `Category` is filled with a constant (default `cosmos`).
- The exporter computes `calculated_gain_spectra_*` as **output spectra − input spectra** per channel, matching the COSMOS authors' feature extraction logic.

## Project Structure

```
ofc_ml/
├── src/ofc_ml/              # Source code
├── data/                     # Dataset (included in repo)
├── submissions/               # Generated submissions
├── main.py                   # Run this to train
├── requirements.txt           # Dependencies
├── NETWORK_ARCHITECTURE.txt  # Network details
└── DATA_FLOW_DIAGRAM.txt     # Data flow
```

## Configuration

Edit `src/ofc_ml/config.py` to customize:

```python
# WandB (optional)
WANDB_MODE = "online"  # "online", "offline", or "disabled"

# Model
# - "mlp": MLP offset predictor
# - "fourier_kan": FourierKAN offset predictor (default)
MODEL_TYPE = "fourier_kan"

# Dataset selection
# - "kaggle": use clean Kaggle train csvs only
# - "cosmos": use external COSMOS-as-kaggle(-by-gain) train csvs only
# - "both": concatenate kaggle + cosmos for training
DATASET_USE = "kaggle"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.05

# Device
DEVICE = "cuda:3"  # "cpu" | "cuda" | "cuda:N"
```

## Model: FourierKAN (new)

This repo includes a **FourierKAN** model implemented in `src/ofc_ml/network.py`:

- The network predicts a **95-dim offset** (per-channel) and adds a baseline computed from
  `target_gain` + `target_gain_tilt` (baseline+offset approach).
- Optionally concatenates the 95-dim `DUT_WSS_activated_channel_index_*` mask into the input
  to help generalize across activation patterns.

Key knobs in `config.py`:

```python
MODEL_TYPE = "fourier_kan"
FOURIER_KAN_HIDDEN_DIMS = [256, 256, 128, 128, 64]
FOURIER_KAN_N_FREQUENCIES = 4
FOURIER_KAN_CONCAT_MASK_INPUT = True
```

## Dataset switch: kaggle / cosmos / both (new)

You can choose which training dataset(s) to use:

```bash
python main.py --dataset-use kaggle
python main.py --dataset-use cosmos
python main.py --dataset-use both
```

Notes:
- `cosmos`/`both` will load and concatenate multiple shards under `data/cosmos-as-kaggle-by-gain/**/train_{features,labels}.csv`
  (e.g. booster/preamp × 15/18/21 dB) if the top-level `train_features.csv` does not exist.
- Test predictions are always generated for Kaggle's `test_features.csv`.

## WandB Setup (Optional)

For training monitoring:

```bash
wandb login
```

Or disable in `config.py`:

```python
WANDB_MODE = "disabled"
```

## Output

After training, you'll find:

- Console output with training progress
- `submissions/submission_YYYYMMDD_HHMMSS.csv` - Your predictions
- WandB dashboard (if enabled)

### Expected Performance

- Validation MSE: ~1.0-1.5
- Validation RMSE: ~1.0-1.2
- Prediction mean: ~17-18 dB

## Kaggle Submission

1. Find latest submission: `ls -lt submissions/`
2. Upload to [Kaggle](https://www.kaggle.com/competitions/ofc-2026-ml-challenge/submissions)
3. View results on leaderboard

## Troubleshooting

**Import error?** Run from project root: `python main.py`

**Out of memory?** Reduce batch size in `src/ofc_ml/model.py`

**Slow training?** Use GPU via `DEVICE="cuda:N"` in `config.py`

**Poor predictions?** Verify masked loss is used (should be by default)

## Advanced

Modify `src/ofc_ml/model.py` for:

- Learning rate, batch size, epochs
- Model architecture in `src/ofc_ml/network.py`

See [NETWORK_ARCHITECTURE.txt](NETWORK_ARCHITECTURE.txt) for details.

## Requirements

- Python 3.12+
- PyTorch, scikit-learn, pandas, numpy, wandb

See `requirements.txt` for versions.
