import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from . import config as cfg

def get_feature_columns(df):
    cat_cols = ['Category', 'EDFA_type', 'edfa_index']
    
    num_cols = ['target_gain', 'target_gain_tilt', 'EDFA_input_power_total', 'EDFA_output_power_total']
    
    spectra_cols = [c for c in df.columns if 'EDFA_input_spectra_' in c]
    mask_cols = [c for c in df.columns if 'DUT_WSS_activated_channel_index' in c]
    
    spectra_cols.sort(key=lambda x: int(x.split('_')[-1]))
    mask_cols.sort(key=lambda x: int(x.split('_')[-1]))
    
    return cat_cols, num_cols, spectra_cols, mask_cols

def create_preprocessor(num_cols, spectra_cols, cat_cols,mask_cols, use_mask_mode):
    """
    创建预处理器，根据USE_MASK模式决定mask处理方式
    
    Args:
        num_cols: 数值特征列
        spectra_cols: 光谱特征列
        cat_cols: 类别特征列
        use_mask_mode: "none", "concat", 或 "multiply"
    
    Returns:
        preprocessor: sklearn预处理器
        mask_transformer: mask转换器（用于multiply模式）
    """
    transformers = []
    
    # 光谱特征处理
    spectra_pipeline = Pipeline([
        ("to_linear", FunctionTransformer(lambda x: 1e3*np.power(10.0, 0.1 * x), feature_names_out="one-to-one")),
    ])
    transformers.append(('spectra', spectra_pipeline, spectra_cols))
    
    # 数值特征处理
    transformers.append(('num', StandardScaler(), num_cols))
    
    # 类别特征处理
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
    
    # 根据USE_MASK模式处理mask
    if use_mask_mode == "concat":
        print(f"[Features] USE_MASK=concat: mask columns will be concatenated as input features")
        transformers.append(('mask', 'passthrough', mask_cols))
        mask_transformer = None
    elif use_mask_mode == "multiply":
        print(f"[Features] USE_MASK=multiply: mask columns will multiply with spectra features")
        # mask不作为输入，而是作为转换器
        mask_transformer = FunctionTransformer(lambda x: x, feature_names_out="one-to-one")
        # 不添加mask到transformers中
        mask_transformer = None
    else:
        print(f"[Features] USE_MASK=none: mask columns will NOT be used")
        mask_transformer = None
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    return preprocessor, mask_transformer

def preprocess_features(train_df, test_df):
    """
    预处理特征，根据USE_MASK配置决定mask处理方式
    
    Args:
        train_df: 训练数据框
        test_df: 测试数据框
    
    Returns:
        X_train: 训练集特征
        X_test: 测试集特征
        mask_cols: mask列名列表
        preprocessor: 预处理器对象
        mask_transformer: mask转换器（multiply模式用）
    """
    print("Preprocessing features...")
    
    cat_cols, num_cols, spectra_cols, mask_cols = get_feature_columns(train_df)
    
    # 根据USE_MASK配置决定处理模式
    use_mask_mode = cfg.USE_MASK
    preprocessor, mask_transformer = create_preprocessor(num_cols, spectra_cols, cat_cols, mask_cols, use_mask_mode)
    
    X_train = preprocessor.fit_transform(train_df)
    
    test_df_processed = test_df.drop(columns=['ID', 'Usage'], errors='ignore')
    X_test = preprocessor.transform(test_df_processed)
    
    # 如果是multiply模式，需要将mask应用到spectra特征上
    if use_mask_mode == "multiply":
        print(f"[Features] Applying mask multiplication to spectra features")
        
        # 从原始数据中提取mask和spectra
        train_masks = train_df[mask_cols].values
        test_masks = test_df_processed[mask_cols].values
        
        # 找到spectra特征在X_train和X_test中的位置
        # 注意：spectra特征在preprocessor中经过了to_linear转换
        # 我们需要在转换后的数据上应用mask
        # 由于ColumnTransformer的输出顺序是固定的，我们需要找到spectra部分
        
        # 重新构建训练数据，在spectra上应用mask
        # 首先从原始数据中提取spectra
        train_spectra = train_df[spectra_cols].values
        test_spectra = test_df_processed[spectra_cols].values
        
        # 应用to_linear转换
        train_spectra_linear = 1e3 * np.power(10.0, 0.1 * train_spectra)
        test_spectra_linear = 1e3 * np.power(10.0, 0.1 * test_spectra)
        
        # 应用mask
        train_spectra_masked = train_spectra_linear * train_masks
        test_spectra_masked = test_spectra_linear * test_masks
        
        # 重新构建X_train和X_test，将masked的spectra替换原来的spectra
        # 这需要知道spectra在转换后数据中的位置
        # 简化方案：我们重新构建整个特征矩阵
        
        # 获取数值和分类特征（这些不需要mask）
        train_num = train_df[num_cols].values
        test_num = test_df_processed[num_cols].values
        
        # 获取分类特征（one-hot编码后）
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        train_cat_encoded = cat_encoder.fit_transform(train_df[cat_cols])
        test_cat_encoded = cat_encoder.transform(test_df_processed[cat_cols])
        
        # 重新组合：num + cat + masked_spectra
        X_train = np.hstack([train_num, train_cat_encoded, train_spectra_masked])
        X_test = np.hstack([test_num, test_cat_encoded, test_spectra_masked])
        
        # 注意：mask_cols 保持不变，训练代码仍需要原始mask来计算masked loss
    
    print(f"Feature shape: {X_train.shape}")
    print(f"  - Spectral features: {len(spectra_cols)}")
    print(f"  - Scalar features (num): {len(num_cols)}")
    print(f"  - Categorical features: {len(cat_cols)}")

    print(f"  - Mask features: {len(mask_cols)}")
    print(f"  - Mask Mode: {use_mask_mode}")
    print(f"  - Total: {X_train.shape[1]}")
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    if X_train.shape[1] != X_test.shape[1]:
        print(f"WARNING: Feature dimension mismatch!")
        print(f"  Train: {X_train.shape[1]}, Test: {X_test.shape[1]}")
    
    return X_train, X_test, mask_cols, preprocessor
