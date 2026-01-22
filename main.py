import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ofc_ml.data import load_data, load_data_separate
from ofc_ml.features import preprocess_features
from ofc_ml.model import train_model, train_model_two_stage
from ofc_ml.utils import create_submission
from ofc_ml.network import compute_baseline_gain
from ofc_ml import config as cfg

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-use", choices=["kaggle", "cosmos", "both"], default=None)
    p.add_argument("--two-stage", action="store_true", help="Use two-stage training (pretrain + finetune)")
    p.add_argument("--load-pretrained", action="store_true", help="Load pretrained model if exists (skip pretraining)")
    p.add_argument("--no-load-pretrained", action="store_true", help="Don't load pretrained model, retrain from scratch")
    return p.parse_args()

def main():
    args = parse_args()
    if args.dataset_use is not None:
        cfg.DATASET_USE = args.dataset_use
    
    # 处理预训练模型加载参数
    if args.load_pretrained:
        cfg.LOAD_PRETRAINED_MODEL = True
    elif args.no_load_pretrained:
        cfg.LOAD_PRETRAINED_MODEL = False
    
    # 根据参数或配置决定是否使用两阶段训练
    use_two_stage = args.two_stage if args.two_stage else cfg.USE_TWO_STAGE_TRAINING
    
    if use_two_stage:
        print("\n" + "="*80)
        print("USING TWO-STAGE TRAINING MODE")
        print("="*80)
        
        # 分别加载数据集
        cosmos_features, cosmos_labels, kaggle_features, kaggle_labels, test_features = load_data_separate()
        
        # 使用所有数据（cosmos + kaggle）来拟合preprocessor
        all_features = np.vstack([cosmos_features, kaggle_features])
        combined_features = pd.concat([cosmos_features, kaggle_features], axis=0, ignore_index=True)
        
        # 特征预处理（只fit一次preprocessor）
        _, X_test, mask_cols, preprocessor = preprocess_features(combined_features, test_features)
        
        # 两阶段训练
        model, metrics = train_model_two_stage(
            cosmos_features, cosmos_labels,
            kaggle_features, kaggle_labels,
            test_features,
            preprocessor,
            mask_cols
        )
        
    else:
        print("\n" + "="*80)
        print("USING SINGLE-STAGE TRAINING MODE")
        print("="*80)
        
        # 原始单阶段训练
        train_features, train_labels, test_features = load_data()
        
        assert len(train_features) == len(train_labels)
        
        X_train, X_test, mask_cols, preprocessor = preprocess_features(train_features, test_features)
        
        target_cols = [c for c in train_labels.columns if 'calculated_gain_spectra_' in c]
        target_cols.sort()
        y_train = train_labels[target_cols].values
        
        model, metrics = train_model(X_train, y_train, preprocessor, train_features, mask_cols)
    
    # 测试集预测
    print("\n" + "="*80)
    print("PREDICTING ON TEST SET")
    print("="*80)
    
    test_masks = test_features[mask_cols].values
    test_target_gain = test_features['target_gain'].values
    test_target_gain_tilt = test_features['target_gain_tilt'].values
    
    y_test_pred = model.predict(X_test, test_target_gain, test_target_gain_tilt, mask=test_masks)
    
    print(f"Final prediction shape: {y_test_pred.shape}")
    
    # 生成提交文件（列名需要前导零，如 calculated_gain_spectra_00）
    target_cols = [f'calculated_gain_spectra_{i:02d}' for i in range(y_test_pred.shape[1])]
    create_submission(y_test_pred, test_features, target_cols)

if __name__ == "__main__":
    main()
