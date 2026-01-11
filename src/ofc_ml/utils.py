import pandas as pd
import numpy as np

def apply_mask(predictions, features_df, mask_cols):
    """
    Apply channel mask to predictions.
    Sets gain to 0 for inactive channels.
    
    Args:
        predictions: numpy array of shape (n_samples, n_channels)
        features_df: pandas DataFrame containing mask columns
        mask_cols: list of mask column names
        
    Returns:
        numpy array: Masked predictions
    """
    print("Applying mask constraints...")
    
    masked_preds = predictions.copy()
    
    # Extract mask values matching the order of mask_cols
    # Assuming predictions columns correspond 1-to-1 with mask_cols order
    masks = features_df[mask_cols].values
    
    # Multiply element-wise
    masked_preds = masked_preds * masks
    
    return masked_preds

def create_submission(predictions, test_features_df, target_cols, output_path):
    """
    Create and save submission file.
    """
    print("Creating submission file...")
    submission = pd.DataFrame(predictions, columns=target_cols)
    submission.insert(0, 'ID', test_features_df['ID'])
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
