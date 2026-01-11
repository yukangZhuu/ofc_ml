# OFC 2026 ML Challenge - EDFA Gain Spectrum Prediction

Machine learning pipeline for predicting EDFA gain spectra in the OFC 2026 ML Challenge.

## Quick Start

```bash
# Create conda environment
conda create -n ofc_ml python=3.12
conda activate ofc_ml

# Install dependencies
pip install -r requirements.txt

# Run training
python main.py
```

That's it! The script will:

- Load data from `data/ofc-2026-ml-challenge/`
- Train the model
- Generate a timestamped submission in `submissions/`

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

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2
```

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

**Slow training?** Use GPU (auto-detected if available)

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
