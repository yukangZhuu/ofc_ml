import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

def get_feature_columns(df):
    cat_cols = ['Category', 'EDFA_type', 'edfa_index']
    
    num_cols = ['target_gain', 'target_gain_tilt', 'EDFA_input_power_total', 'EDFA_output_power_total']
    
    spectra_cols = [c for c in df.columns if 'EDFA_input_spectra_' in c]
    spectra_cols.sort()
    
    mask_cols = [c for c in df.columns if 'DUT_WSS_activated_channel_index_' in c]
    mask_cols.sort()
    
    return cat_cols, num_cols, spectra_cols, mask_cols

def create_preprocessor(num_cols, spectra_cols, cat_cols, apply_mask_to_spectra=False):
    spectra_pipeline = Pipeline([
        # 光谱值先从 dB 转线性刻度：x_dB -> 10^(0.1 * x_dB)
        ("to_linear", FunctionTransformer(lambda x: 1e3*np.power(10.0, 0.1 * x), feature_names_out="one-to-one")),
        # ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('spectra', spectra_pipeline, spectra_cols),
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )
    
    preprocessor.apply_mask_to_spectra = apply_mask_to_spectra
    preprocessor.spectra_cols = spectra_cols
    
    return preprocessor

def preprocess_features(train_df, test_df, apply_mask_to_spectra=False):
    print("Preprocessing features...")
    
    cat_cols, num_cols, spectra_cols, mask_cols = get_feature_columns(train_df)
    
    preprocessor = create_preprocessor(num_cols, spectra_cols, cat_cols, apply_mask_to_spectra)
    
    X_train = preprocessor.fit_transform(train_df)
    
    test_df_processed = test_df.drop(columns=['ID', 'Usage'], errors='ignore')
    X_test = preprocessor.transform(test_df_processed)
    
    if apply_mask_to_spectra:
        print("Applying mask to spectral features before feeding to network...")
        
        # Get the indices of spectral features in the transformed array
        # ColumnTransformer preserves the order of transformers
        spectra_start_idx = 0
        spectra_end_idx = len(spectra_cols)
        
        # Apply mask to spectral features
        train_masks = train_df[mask_cols].values
        test_masks = test_df_processed[mask_cols].values
        
        # Multiply spectral features by mask
        X_train[:, spectra_start_idx:spectra_end_idx] = X_train[:, spectra_start_idx:spectra_end_idx] * train_masks
        X_test[:, spectra_start_idx:spectra_end_idx] = X_test[:, spectra_start_idx:spectra_end_idx] * test_masks
        
        print(f"  Mask applied to {len(spectra_cols)} spectral features")
    
    print(f"Feature shape: {X_train.shape}")
    print(f"  - Spectral features: {len(spectra_cols)}")
    print(f"  - Scalar features (num): {len(num_cols)}")
    print(f"  - Categorical features: {len(cat_cols)}")
    print(f"  - Total: {X_train.shape[1]}")
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    if X_train.shape[1] != X_test.shape[1]:
        print(f"WARNING: Feature dimension mismatch!")
        print(f"  Train: {X_train.shape[1]}, Test: {X_test.shape[1]}")
    
    return X_train, X_test, mask_cols, preprocessor
