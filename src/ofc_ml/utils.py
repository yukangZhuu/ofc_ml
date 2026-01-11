import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from .config import SUBMISSION_DIR

def apply_mask(predictions, features_df, mask_cols):
    print("Applying mask constraints...")
    
    masked_preds = predictions.copy()
    
    masks = features_df[mask_cols].values
    
    masked_preds = masked_preds * masks
    
    return masked_preds

def create_submission(predictions, test_features_df, target_cols, output_path=None):
    print("Creating submission file...")
    
    submission = pd.DataFrame(predictions, columns=target_cols)
    submission.insert(0, 'ID', test_features_df['ID'])
    
    if output_path is None:
        SUBMISSION_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = SUBMISSION_DIR / f"submission_{timestamp}.csv"
    
    submission.to_csv(output_path, index=False)
    
    print(f"Submission saved to {output_path}")
    
    return output_path
