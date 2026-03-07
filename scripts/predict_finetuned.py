#!/usr/bin/env python3
import sys
import torch
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from ofc_ml import config as cfg
from ofc_ml.data import load_data_separate
from ofc_ml.features import preprocess_features
from ofc_ml.network import HybridFNOKANPredictor, compute_baseline_gain
from ofc_ml.utils import create_submission

def parse_args():
    parser = argparse.ArgumentParser(description="Predict using a finetuned model.")
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=str(cfg.FINETUNE_MODEL_PATH),
        help="Path to the finetuned model checkpoint (.pt)"
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default=None,
        help="Path to save the submission CSV. Defaults to submissions/submission_{timestamp}_finetuned_prediction.csv"
    )
    return parser.parse_args()

class PyTorchModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def predict(self, X, target_gain, target_gain_tilt, mask=None):
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X).to(self.device)

            if mask is not None:
                tensor_mask = torch.FloatTensor(mask).to(self.device)
                preds_offset = self.model(tensor_X, tensor_mask)
            else:
                preds_offset = self.model(tensor_X)

            preds_offset = preds_offset.cpu().numpy()

            baseline = compute_baseline_gain(target_gain, target_gain_tilt)

            preds = baseline + preds_offset

            if mask is not None:
                preds = preds * mask

            return preds

def main():
    args = parse_args()
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    
    # 1. Data Loading & Preprocessing
    # We need to load all data to correctly fit the preprocessor (scaling, etc.)
    print("Loading datasets to fit preprocessor...")
    cosmos_features, cosmos_labels, kaggle_features, kaggle_labels, test_features = load_data_separate()
    
    combined_features = pd.concat([cosmos_features, kaggle_features], axis=0, ignore_index=True)
    
    print("Fitting preprocessor...")
    _, X_test, mask_cols, preprocessor = preprocess_features(combined_features, test_features)
    
    # 2. Model Initialization
    # Detect Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load Checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get dimensions from checkpoint if available, otherwise infer
    input_dim = checkpoint.get('input_dim', X_test.shape[1])
    # For output dim, we can try to guess or use the saved one. 
    # If not saved, we might default to 95 (standard for this task) but better to be safe.
    # The new saving logic I added includes output_dim. 
    # If using an old model without it, we might crash or need hardcoding.
    # Let's assume the user uses the new saving logic or we infer from config/data.
    # But wait, we don't have y_test, so we can't infer output_dim from data easily unless we look at labels.
    # cosmos_labels has 95 columns usually.
    target_cols = [c for c in cosmos_labels.columns if 'calculated_gain_spectra_' in c]
    output_dim = checkpoint.get('output_dim', len(target_cols))

    print(f"Model Input Dim: {input_dim}, Output Dim: {output_dim}")

    model = HybridFNOKANPredictor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=cfg.HYBRID_FNO_KAN_HIDDEN_DIMS,
        dropout=cfg.HYBRID_FNO_KAN_DROPOUT,
        use_residual=True,
        n_frequencies=cfg.HYBRID_FNO_KAN_N_FREQUENCIES,
        spectral_freq_ratio=cfg.HYBRID_FNO_KAN_SPECTRAL_FREQ_RATIO,
        use_spectral_mixing=cfg.HYBRID_FNO_KAN_USE_SPECTRAL_MIXING,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    wrapper = PyTorchModelWrapper(model, device)
    print("Model loaded successfully.")

    # 3. Prediction
    print("Predicting on test set...")
    test_masks = test_features[mask_cols].values
    test_target_gain = test_features['target_gain'].values
    test_target_gain_tilt = test_features['target_gain_tilt'].values
    
    y_test_pred = wrapper.predict(X_test, test_target_gain, test_target_gain_tilt, mask=test_masks)
    print(f"Prediction shape: {y_test_pred.shape}")

    # 4. Save Submission
    target_cols_names = [f'calculated_gain_spectra_{i:02d}' for i in range(y_test_pred.shape[1])]
    
    if args.output_path:
        out_path = Path(args.output_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = cfg.SUBMISSION_DIR / f"submission_{timestamp}_finetuned_prediction.csv"
    
    create_submission(y_test_pred, test_features, target_cols_names, output_path=out_path)

if __name__ == "__main__":
    main()
