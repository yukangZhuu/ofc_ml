from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "ofc-2026-ml-challenge"
TRAIN_FEATURES_PATH = DATA_DIR / "train_features_clean.csv"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels_clean.csv"
TEST_FEATURES_PATH = DATA_DIR / "test_features.csv"
EXAMPLE_SUBMISSION_PATH = DATA_DIR / "example_submission.csv"

# Dataset switch:
# - "kaggle": use TRAIN_FEATURES_PATH/TRAIN_LABELS_PATH (clean kaggle train)
# - "cosmos": use COSMOS_TRAIN_* as training data
# - "both": concatenate kaggle train + cosmos train for training
DATASET_USE = "both"  # one of: {"kaggle", "cosmos", "both"}

# External cosmos-as-kaggle dataset (same schema as kaggle train)
# This can either be a directory containing train_features.csv/train_labels.csv directly,
# or a directory containing multiple subfolders each with those two files (e.g. by gain).
COSMOS_DATA_DIR = PROJECT_ROOT / "data" / "cosmos-as-kaggle"
COSMOS_TRAIN_FEATURES_PATH = COSMOS_DATA_DIR / "train_features.csv"  # optional (may not exist)
COSMOS_TRAIN_LABELS_PATH = COSMOS_DATA_DIR / "train_labels.csv"      # optional (may not exist)

SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.05

WANDB_PROJECT = "ofc-2026-ml-challenge"
WANDB_ENTITY = None
WANDB_MODE = "online"

# Model selection
# - "mlp": SimpleGainPredictor (MLP)
# - "fourier_kan": FourierKANGainPredictor (FourierKAN blocks), typically deeper
# - "hybrid_fno_kan": HybridFNOKANPredictor (FNO + FourierKAN 混合架构)
# - "resnet_mlp": ResNetPredictor (Robust Deep MLP with Residuals)
MODEL_TYPE = "resnet_mlp"

DROPOUT = 0.1
HIDDEN_DIMS = [512, 512, 512, 512, 256, 256]

# Mask application strategy
# - "concat": concatenate mask as additional input dimension (original approach)
# - "multiply": multiply mask to spectral features before feeding to network (new approach)
MASK_STRATEGY = "multiply"  # one of: {"concat", "multiply"}

# ResNet-MLP hyperparameters
RESNET_MLP_DROPOUT = 0.1
RESNET_MLP_HIDDEN_DIMS = [512, 512, 512, 512, 256, 256]  # 6 layers, wide and deep

# FourierKAN hyperparameters
FOURIER_KAN_DROPOUT = 0.2
# Deeper by default (you can tune)
FOURIER_KAN_HIDDEN_DIMS = [128, 128, 64] 
# Number of Fourier frequencies used inside each FourierKANLayer
FOURIER_KAN_N_FREQUENCIES = 4
# Let the model see activation pattern (mask) by concatenating it to input features.
# Deprecated: Use MASK_STRATEGY instead. This is kept for backward compatibility.
FOURIER_KAN_CONCAT_MASK_INPUT = True

# Hybrid FNO+KAN hyperparameters
HYBRID_FNO_KAN_DROPOUT = 0.2
HYBRID_FNO_KAN_HIDDEN_DIMS = [128, 128, 64]  # 4层，保守维度
HYBRID_FNO_KAN_N_FREQUENCIES = 4
HYBRID_FNO_KAN_SPECTRAL_FREQ_RATIO = 0.5  # 频域保留的频率比例（0-1），0.5表示保留低50%频率，确保各层物理意义一致
HYBRID_FNO_KAN_USE_SPECTRAL_MIXING = True  # 是否使用频域混合层
HYBRID_FNO_KAN_CONCAT_MASK_INPUT = True

# Backward-compatible aliases (old SimpleKAN naming)
SIMPLE_KAN_DROPOUT = FOURIER_KAN_DROPOUT
SIMPLE_KAN_HIDDEN_DIMS = FOURIER_KAN_HIDDEN_DIMS
SIMPLE_KAN_CONCAT_MASK_INPUT = FOURIER_KAN_CONCAT_MASK_INPUT
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
EARLY_STOPPING_PATIENCE = 50

# Training device: "cpu", "cuda", or "cuda:N" (e.g. "cuda:2")
DEVICE = "cuda"
# Two-stage training (pretrain + finetune)
USE_TWO_STAGE_TRAINING = True  # 是否使用两阶段训练

# 是否从已有的预训练模型加载并跳过预训练阶段
# 如果为 True 且 PRETRAIN_MODEL_PATH 存在，则直接加载预训练模型并开始微调
LOAD_PRETRAINED_MODEL = True  # 如果预训练模型存在，直接加载

# Stage 1: Pretrain on COSMOS dataset
PRETRAIN_LEARNING_RATE = 0.001  # 预训练学习率（可以稍大）
PRETRAIN_WEIGHT_DECAY = 1e-4
PRETRAIN_BATCH_SIZE = 256
PRETRAIN_EPOCHS = 500
PRETRAIN_EARLY_STOPPING_PATIENCE = 50

# Stage 2: Finetune on Kaggle dataset
FINETUNE_LEARNING_RATE = 0.0002  # 微调学习率（通常更小）
FINETUNE_WEIGHT_DECAY = 5e-5
FINETUNE_BATCH_SIZE = 32  # 微调可以用更小的batch size
FINETUNE_EPOCHS = 1000
FINETUNE_EARLY_STOPPING_PATIENCE = 50

# ==================== Discriminative Fine-tuning 策略 ====================
# 判别式微调：不同层使用不同学习率（底层小学习率，顶层大学习率）
DISCRIMINATIVE_LR_DECAY = 0.95  # 每往底层走一层，学习率衰减因子（0.95表示每层学习率是上一层的95%）
# 例如：顶层 lr=0.0004, 下一层 lr=0.0004*0.95, 再下一层 lr=0.0004*0.95^2

# ==================== Layer Freezing Strategy ====================
# Fine-tuning时冻结底层参数，只训练顶层
# - True: 冻结除最后一层外的所有层
# - False: 不冻结，全量微调
FREEZE_LAYERS = True
UNFREEZE_LAST_N_LAYERS = 2  # 仅解冻最后N层（如果是ResNet-MLP，则解冻最后N个block和输出层）

# Model checkpoint path
PRETRAIN_MODEL_PATH = PROJECT_ROOT / "models" / "pretrained_model0122_zyk_02.pt"
