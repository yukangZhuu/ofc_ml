import numpy as np
import pandas as pd
from ofc_ml.features import create_preprocessor, get_feature_columns

def test_interleaved():
    # Setup dummy data with 2 channels
    data = {
        'Category': ['A', 'B'],
        'EDFA_type': ['T1', 'T2'],
        'edfa_index': [1, 2],
        'target_gain': [10, 20],
        'target_gain_tilt': [0.5, -0.5],
        'EDFA_input_power_total': [-10, -5],
        'EDFA_output_power_total': [0, 5],
        # Spectra (dBm) -> Linear (uW): 0->1000, 10->10000, 20->100000, 30->1000000
        'EDFA_input_spectra_1': [0, 10], 
        'EDFA_input_spectra_2': [20, 30],
        # Mask
        'DUT_WSS_activated_channel_index_1': [1, 0],
        'DUT_WSS_activated_channel_index_2': [1, 1]
    }
    df = pd.DataFrame(data)
    
    cat_cols, num_cols, spectra_cols, mask_cols = get_feature_columns(df)
    
    print("\n--- Testing Concat (Interleaved) Mode ---")
    prep = create_preprocessor(num_cols, spectra_cols, cat_cols, mask_cols, "concat")
    X = prep.fit_transform(df)
    
    # Structure: [num(4), cat(6?), interleaved(4)]
    # We focus on the last 4 columns which should be [s1, m1, s2, m2]
    interleaved_part = X[:, -4:]
    print("Transformed Interleaved Part:")
    print(interleaved_part)
    
    # Expected Row 0:
    # s1=1000, m1=1, s2=100000, m2=1
    expected_0 = [1000.0, 1.0, 100000.0, 1.0]
    
    # Expected Row 1:
    # s1=10000, m1=0, s2=1000000, m2=1
    expected_1 = [10000.0, 0.0, 1000000.0, 1.0]
    
    assert np.allclose(interleaved_part[0], expected_0), "Row 0 mismatch"
    assert np.allclose(interleaved_part[1], expected_1), "Row 1 mismatch"
    
    print("Interleaved Check: PASSED")

if __name__ == "__main__":
    test_interleaved()
