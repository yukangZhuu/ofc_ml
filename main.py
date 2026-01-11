import sys
from pathlib import Path

# Add src to python path to ensure modules can be imported
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ofc_ml.data import load_data
from ofc_ml.features import preprocess_features
from ofc_ml.model import train_model
from ofc_ml.utils import apply_mask, create_submission
from ofc_ml.config import SUBMISSION_PATH

def main():
    # 1. Load Data
    train_features, train_labels, test_features = load_data()
    
    # Verify alignment
    assert len(train_features) == len(train_labels)
    
    # 2. Preprocess
    X_train, X_test, mask_cols, _ = preprocess_features(train_features, test_features)
    
    # 3. Prepare targets
    target_cols = [c for c in train_labels.columns if 'calculated_gain_spectra_' in c]
    target_cols.sort()
    y_train = train_labels[target_cols].values
    
    # 4. Train
    model, metrics = train_model(X_train, y_train)
    
    # 5. Predict on Test
    print("Predicting on test set...")
    y_test_pred = model.predict(X_test)
    
    # 6. Post-process (Masking)
    y_test_pred_masked = apply_mask(y_test_pred, test_features, mask_cols)
    
    # 7. Create Submission
    create_submission(y_test_pred_masked, test_features, target_cols, SUBMISSION_PATH)

if __name__ == "__main__":
    main()
