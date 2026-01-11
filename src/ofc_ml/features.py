import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def get_feature_columns(df):
    """
    Identify feature columns by type.
    """
    # Categorical
    cat_cols = ['Category', 'EDFA_type', 'edfa_index']
    
    # Numerical Scalars
    num_cols = ['target_gain', 'target_gain_tilt', 'EDFA_input_power_total', 'EDFA_output_power_total']
    
    # Spectra columns
    spectra_cols = [c for c in df.columns if 'EDFA_input_spectra_' in c]
    spectra_cols.sort()
    
    # Mask columns
    mask_cols = [c for c in df.columns if 'DUT_WSS_activated_channel_index_' in c]
    mask_cols.sort()
    
    return cat_cols, num_cols, spectra_cols, mask_cols

def create_preprocessor(cat_cols, num_cols, spectra_cols, mask_cols):
    """
    Create a ColumnTransformer for preprocessing.
    """
    all_num_cols = num_cols + spectra_cols
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), all_num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('mask', 'passthrough', mask_cols)
        ]
    )
    return preprocessor

def preprocess_features(train_df, test_df):
    """
    Preprocess features for training and testing.
    
    Returns:
        tuple: (X_train, X_test, mask_cols, preprocessor)
    """
    print("Preprocessing features...")
    
    cat_cols, num_cols, spectra_cols, mask_cols = get_feature_columns(train_df)
    
    preprocessor = create_preprocessor(cat_cols, num_cols, spectra_cols, mask_cols)
    
    # Fit on train and transform both
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    
    return X_train, X_test, mask_cols, preprocessor
