# OFC 2026 ML Challenge

This project contains the machine learning pipeline for the OFC 2026 ML Challenge.

## Structure

*   `src/ofc_ml/`: Source code package.
    *   `data.py`: Data loading.
    *   `features.py`: Feature preprocessing.
    *   `model.py`: Model training.
    *   `utils.py`: Utilities (masking, submission).
    *   `config.py`: Configuration and paths.
*   `data/`: Dataset directory.
*   `main.py`: Main entry point to run the pipeline.
*   `requirements.txt`: Project dependencies.

## Installation

Ensure you have Python 3.12+ installed.

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python main.py
```

This will generate `submission.csv` in the project root.
