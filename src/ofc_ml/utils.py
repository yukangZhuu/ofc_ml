import pandas as pd
from datetime import datetime
from .config import SUBMISSION_DIR

def create_submission(predictions, test_features_df, target_cols, output_path=None):
    """
    Create and save submission file.
    
    Args:
        predictions: numpy array of shape (n_samples, n_targets)
        test_features_df: pandas DataFrame containing 'ID' column
        target_cols: list of target column names
        output_path: optional path to save the submission file
    """
    print("Creating submission file...")
    
    submission = pd.DataFrame(predictions, columns=target_cols)
    submission.insert(0, 'ID', test_features_df['ID'])
    
    if output_path is None:
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = SUBMISSION_DIR / f"submission_{timestamp}.csv"
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return output_path
