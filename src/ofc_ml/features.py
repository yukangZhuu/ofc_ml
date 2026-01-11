import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def get_feature_columns(df):
    cat_cols = ['Category', 'EDFA_type', 'edfa_index']
    
    num_cols = ['target_gain', 'target_gain_tilt', 'EDFA_input_power_total', 'EDFA_output_power_total']
    
    spectra_cols = [c for c in df.columns if 'EDFA_input_spectra_' in c]
    spectra_cols.sort()
    
    mask_cols = [c for c in df.columns if 'DUT_WSS_activated_channel_index_' in c]
    mask_cols.sort()
    
    return cat_cols, num_cols, spectra_cols, mask_cols

def create_preprocessor(cat_cols, num_cols, spectra_cols, mask_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ('spectra', StandardScaler(), spectra_cols),
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('mask', 'passthrough', mask_cols)
        ],
        remainder='drop'
    )
    return preprocessor

def preprocess_features(train_df, test_df):
    print("Preprocessing features...")
    
    cat_cols, num_cols, spectra_cols, mask_cols = get_feature_columns(train_df)
    
    preprocessor = create_preprocessor(cat_cols, num_cols, spectra_cols, mask_cols)
    
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    
    print(f"Feature shape: {X_train.shape}")
    print(f"  - Spectral features: {len(spectra_cols)}")
    print(f"  - Scalar features (num): {len(num_cols)}")
    print(f"  - Categorical features: {len(cat_cols)}")
    print(f"  - Mask features: {len(mask_cols)}")
    print(f"  - Total: {X_train.shape[1]}")
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    if X_train.shape[1] != X_test.shape[1]:
        print(f"WARNING: Feature dimension mismatch!")
        print(f"  Train: {X_train.shape[1]}, Test: {X_test.shape[1]}")
    
    return X_train, X_test, mask_cols, preprocessor
