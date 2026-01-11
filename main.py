import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ofc_ml.data import load_data
from ofc_ml.features import preprocess_features
from ofc_ml.model import train_model
from ofc_ml.utils import apply_mask, create_submission
from ofc_ml.config import SUBMISSION_PATH

def main():
    train_features, train_labels, test_features = load_data()
    
    assert len(train_features) == len(train_labels)
    
    X_train, X_test, mask_cols, preprocessor = preprocess_features(train_features, test_features)
    
    target_cols = [c for c in train_labels.columns if 'calculated_gain_spectra_' in c]
    target_cols.sort()
    y_train = train_labels[target_cols].values
    
    model, metrics = train_model(X_train, y_train, preprocessor)
    
    print("Predicting on test set...")
    y_test_pred = model.predict(X_test)
    
    y_test_pred_masked = apply_mask(y_test_pred, test_features, mask_cols)
    
    create_submission(y_test_pred_masked, test_features, target_cols, SUBMISSION_PATH)

if __name__ == "__main__":
    main()
