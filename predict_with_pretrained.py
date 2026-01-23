import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import argparse

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from ofc_ml.data import load_data_separate
from ofc_ml.features import preprocess_features
from ofc_ml.model import PyTorchModelWrapper, apply_mask_to_features
from ofc_ml.network import SimpleGainPredictor, FourierKANGainPredictor, HybridFNOKANPredictor, ResNetPredictor
from ofc_ml.utils import create_submission
from ofc_ml import config as cfg
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser(description="Predict using a pretrained model")
    # Argument is optional. If provided, it's used as path. If not, config default is used.
    p.add_argument("model_path_arg", nargs="?", type=str, default=None,
                   help="Path to the pretrained model checkpoint (.pt file). If omitted, uses path from config.")
    p.add_argument("--mask-strategy", choices=["concat", "multiply"], default=None, 
                   help="Mask strategy used during training (override config)")
    p.add_argument("--model-type", choices=["mlp", "fourier_kan", "hybrid_fno_kan", "resnet_mlp"], default=None,
                   help="Model architecture type (override config)")
    return p.parse_args()

def main():
    args = parse_args()
    
    # 1. Determine model path
    if args.model_path_arg:
        model_path = Path(args.model_path_arg)
        print(f"Using specified model path: {model_path}")
    elif cfg.PRETRAIN_MODEL_PATH.exists():
        model_path = cfg.PRETRAIN_MODEL_PATH
        print(f"Using default pretrained model path from config: {model_path}")
    else:
        print("Error: No model path provided and default path does not exist.")
        print(f"Default path from config: {cfg.PRETRAIN_MODEL_PATH}")
        sys.exit(1)
        
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
        
    # 2. 加载数据
    print("Loading data...")
    cosmos_features, cosmos_labels, kaggle_features, kaggle_labels, test_features = load_data_separate()
    
    # 3. 预处理
    # 为了保证特征转换的一致性，我们需要像训练时一样fit preprocessor
    # 使用所有数据（cosmos + kaggle）来拟合preprocessor
    combined_features = pd.concat([cosmos_features, kaggle_features], axis=0, ignore_index=True)
    
    # 确定Mask策略
    if args.mask_strategy:
        cfg.MASK_STRATEGY = args.mask_strategy
    
    apply_mask_to_spectra = (cfg.MASK_STRATEGY.lower() == "multiply")
    print(f"Mask strategy: {cfg.MASK_STRATEGY}")
    
    # 预处理
    print("Preprocessing features...")
    # 注意：这里我们fit on combined_features，transform on test_features
    # preprocess_features函数内部会返回fit好的scaler和pca等，但我们主要需要X_test和preprocessor
    _, X_test, mask_cols, preprocessor = preprocess_features(combined_features, test_features, apply_mask_to_spectra)
    
    # 4. 加载模型
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu')) # 先加载到CPU
    
    # 确定模型类型和维度
    model_type = args.model_type if args.model_type else checkpoint.get('model_type', cfg.MODEL_TYPE)
    input_dim = checkpoint.get('input_dim', X_test.shape[1])
    output_dim = checkpoint.get('output_dim', 74) # Default 74
    
    # 如果checkpoint里没有input_dim，我们需要小心
    # 如果是concat策略，input_dim可能比X_test.shape[1]大（加上了mask维度）
    if 'input_dim' not in checkpoint:
        # 尝试推断
        if cfg.MASK_STRATEGY == "concat" and X_test.shape[1] < input_dim: 
             # 这是一个简单的启发式，但最好相信checkpoint里的input_dim
             pass
    
    print(f"Model Type: {model_type}")
    print(f"Input Dim: {input_dim}")
    print(f"Output Dim: {output_dim}")
    
    # 初始化模型架构
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if model_type == "mlp":
        model = SimpleGainPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=cfg.HIDDEN_DIMS,
            dropout=cfg.DROPOUT
        )
    elif model_type in {"fourier_kan", "simple_kan"}:
        model = FourierKANGainPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=cfg.FOURIER_KAN_HIDDEN_DIMS,
            dropout=cfg.FOURIER_KAN_DROPOUT,
            use_residual=True,
            n_frequencies=cfg.FOURIER_KAN_N_FREQUENCIES,
        )
    elif model_type == "hybrid_fno_kan":
        model = HybridFNOKANPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=cfg.HYBRID_FNO_KAN_HIDDEN_DIMS,
            dropout=cfg.HYBRID_FNO_KAN_DROPOUT,
            use_residual=True,
            n_frequencies=cfg.HYBRID_FNO_KAN_N_FREQUENCIES,
            spectral_freq_ratio=cfg.HYBRID_FNO_KAN_SPECTRAL_FREQ_RATIO,
            use_spectral_mixing=cfg.HYBRID_FNO_KAN_USE_SPECTRAL_MIXING,
        )
    elif model_type == "resnet_mlp":
        model = ResNetPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=cfg.RESNET_MLP_HIDDEN_DIMS,
            dropout=cfg.RESNET_MLP_DROPOUT
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 5. 预测
    print("Predicting on test set...")
    wrapper = PyTorchModelWrapper(model, device)
    
    test_masks = test_features[mask_cols].values
    test_target_gain = test_features['target_gain'].values
    test_target_gain_tilt = test_features['target_gain_tilt'].values
    
    y_test_pred = wrapper.predict(X_test, test_target_gain, test_target_gain_tilt, mask=test_masks)
    
    print(f"Final prediction shape: {y_test_pred.shape}")
    
    # 6. 生成提交文件
    target_cols = [f'calculated_gain_spectra_{i:02d}' for i in range(y_test_pred.shape[1])]
    
    # Custom submission creation to include _pretrained_ tag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_filename = f"submission_pretrained_{timestamp}.csv"
    submission_path = cfg.SUBMISSION_DIR / submission_filename
    
    # Use standard utility but rename/copy or just manually save if utils doesn't support custom name
    # Looking at utils.py (not visible here but assuming behavior), let's just use pandas directly 
    # to be safe and ensure the name is exactly what we want.
    
    submission_df = pd.DataFrame(y_test_pred, columns=target_cols)
    submission_df.insert(0, 'ID', test_features['ID'].values)
    
    cfg.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nSubmission generated at: {submission_path}")

if __name__ == "__main__":
    main()
