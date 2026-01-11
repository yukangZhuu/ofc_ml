import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from ofc_ml.features import preprocess_features
from ofc_ml.utils import apply_mask

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.train_data = pd.DataFrame({
            'Category': ['A', 'B'],
            'EDFA_type': ['T1', 'T2'],
            'edfa_index': [0, 1],
            'target_gain': [10.0, 20.0],
            'target_gain_tilt': [0.5, -0.5],
            'EDFA_input_power_total': [-10, -5],
            'EDFA_output_power_total': [5, 10],
            'EDFA_input_spectra_00': [-20, -25],
            'DUT_WSS_activated_channel_index_00': [1, 0]
        })
        self.test_data = self.train_data.copy()
        self.test_data['ID'] = [1, 2]

    def test_preprocess(self):
        X_train, X_test, mask_cols, _ = preprocess_features(self.train_data, self.test_data)
        self.assertEqual(X_train.shape[0], 2)
        self.assertEqual(X_test.shape[0], 2)
        self.assertIn('DUT_WSS_activated_channel_index_00', mask_cols)

    def test_apply_mask(self):
        # Mock features with mask column
        features = pd.DataFrame({
            'DUT_WSS_activated_channel_index_00': [1, 0, 1]
        })
        mask_cols = ['DUT_WSS_activated_channel_index_00']
        
        # Predictions (all ones)
        preds = np.ones((3, 1))
        
        masked_preds = apply_mask(preds, features, mask_cols)
        
        # Check logic: 1*1=1, 1*0=0, 1*1=1
        expected = np.array([[1], [0], [1]])
        np.testing.assert_array_equal(masked_preds, expected)

if __name__ == '__main__':
    unittest.main()
