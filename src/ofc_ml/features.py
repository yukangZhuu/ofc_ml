import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from . import config as cfg

def get_feature_columns(df):
    cat_cols = ['Category', 'EDFA_type', 'edfa_index']
    
    num_cols = ['target_gain', 'target_gain_tilt', 'EDFA_input_power_total', 'EDFA_output_power_total']
    
    spectra_cols = [c for c in df.columns if 'EDFA_input_spectra_' in c]
    mask_cols = [c for c in df.columns if 'DUT_WSS_activated_channel_index' in c]

    spectra_cols.sort(key=lambda x: int(x.split('_')[-1]))
    mask_cols.sort(key=lambda x: int(x.split('_')[-1]))
    
    return cat_cols, num_cols, spectra_cols, mask_cols

class MaskedSpectraTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle the multiplication of spectra and mask columns.
    Assumes the input contains [spectra_cols, mask_cols] in that order.
    """
    def __init__(self, n_channels):
        self.n_channels = n_channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure input is numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Split into spectra and mask
        # Expecting input shape: [n_samples, 2 * n_channels]
        spectra = X[:, :self.n_channels]
        mask = X[:, self.n_channels:]
        
        # 1. Convert log power (dBm) to linear power (mW)
        # 1e3 * 10^(0.1 * x)
        spectra_linear = 1e3 * np.power(10.0, 0.1 * spectra)
        
        # 2. Multiply by mask
        masked_spectra = spectra_linear * mask
        
        return masked_spectra
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return [f"spectra_masked_{i}" for i in range(self.n_channels)]
        # Return names corresponding to spectra columns
        return input_features[:self.n_channels]

def create_preprocessor(num_cols, spectra_cols, cat_cols, mask_cols, use_mask_mode):
    """
    Create sklearn preprocessor based on USE_MASK mode.
    """
    transformers = []
    
    # 1. Numerical Features: StandardScaler
    transformers.append(('num', StandardScaler(), num_cols))
    
    # 2. Categorical Features: OneHotEncoder
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
    
    # 3. Spectra and Mask Features
    if use_mask_mode == "multiply":
        print(f"[Features] USE_MASK=multiply: spectra * mask")
        # Pass both spectra and mask columns to the custom transformer
        # Note: The order in the list MUST match the split logic in MaskedSpectraTransformer
        combined_cols = spectra_cols + mask_cols
        transformers.append(('spectra_masked', MaskedSpectraTransformer(len(spectra_cols)), combined_cols))
        
    elif use_mask_mode == "concat":
        print(f"[Features] USE_MASK=concat: spectra + mask")
        # Process spectra: Log -> Linear
        transformers.append(('spectra', FunctionTransformer(lambda x: 1e3*np.power(10.0, 0.1 * x)), spectra_cols))
        # Process mask: Passthrough
        transformers.append(('mask', 'passthrough', mask_cols))
        
    else: # none
        print(f"[Features] USE_MASK=none: spectra only")
        # Process spectra: Log -> Linear
        transformers.append(('spectra', FunctionTransformer(lambda x: 1e3*np.power(10.0, 0.1 * x)), spectra_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor

def preprocess_features(train_df, test_df):
    """
    Preprocess features using the configured pipeline.
    """
    print("Preprocessing features...")
    
    cat_cols, num_cols, spectra_cols, mask_cols = get_feature_columns(train_df)
    use_mask_mode = cfg.USE_MASK
    
    preprocessor = create_preprocessor(num_cols, spectra_cols, cat_cols, mask_cols, use_mask_mode)
    
    # Fit and transform training data
    X_train = preprocessor.fit_transform(train_df)
    
    # Transform test data
    test_df_processed = test_df.drop(columns=['ID', 'Usage'], errors='ignore')
    X_test = preprocessor.transform(test_df_processed)
    
    print(f"Feature shape: {X_train.shape}")
    print(f"  - Spectral features: {len(spectra_cols)}")
    print(f"  - Scalar features (num): {len(num_cols)}")
    print(f"  - Categorical features: {len(cat_cols)}")
    print(f"  - Mask features: {len(mask_cols)}")
    print(f"  - Mask Mode: {use_mask_mode}")
    print(f"  - Total output dims: {X_train.shape[1]}")
    
    if X_train.shape[1] != X_test.shape[1]:
        print(f"WARNING: Feature dimension mismatch! Train: {X_train.shape[1]}, Test: {X_test.shape[1]}")
    
    return X_train, X_test, mask_cols, preprocessor
