"""
计算保守调整和浅层网络结合的参数量
"""

def calculate_linear_params(input_dim, output_dim):
    """计算线性层参数量（包括权重和偏置）"""
    return input_dim * output_dim + output_dim

def calculate_model_params(input_dim, hidden_dims, output_dim):
    """计算整个模型的参数量"""
    total = 0
    prev_dim = input_dim
    for h in hidden_dims:
        total += calculate_linear_params(prev_dim, h)
        prev_dim = h
    total += calculate_linear_params(prev_dim, output_dim)
    return total

# 正确的维度
concat_input_dim = 207  # 95(spectra) + 4(num) + 13(cat) + 95(mask)
multiply_input_dim = 112  # 95(spectra) + 4(num) + 13(cat)
output_dim = 95  # 95个通道

# 原始配置
original_concat_hidden_dims = [256, 256, 128, 128, 128]
original_multiply_hidden_dims = [256, 256, 128, 128, 128]

# 保守调整（5层）
conservative_hidden_dims = [150, 150, 75, 75, 75]

# 浅层网络（4层，原始维度）
shallow_original_hidden_dims = [256, 256, 128, 128]

# 浅层网络（4层，保守维度）- 保守调整和浅层网络的结合
combined_hidden_dims = [150, 150, 75, 75]

# 计算参数量
original_concat_params = calculate_model_params(concat_input_dim, original_concat_hidden_dims, output_dim)
original_multiply_params = calculate_model_params(multiply_input_dim, original_multiply_hidden_dims, output_dim)
conservative_params = calculate_model_params(multiply_input_dim, conservative_hidden_dims, output_dim)
shallow_original_params = calculate_model_params(multiply_input_dim, shallow_original_hidden_dims, output_dim)
combined_params = calculate_model_params(multiply_input_dim, combined_hidden_dims, output_dim)

print("=" * 80)
print("保守调整和浅层网络结合的参数量分析")
print("=" * 80)

print(f"\n输入维度：")
print(f"  concat策略: {concat_input_dim}维")
print(f"  multiply策略: {multiply_input_dim}维")

print(f"\n配置对比：")
print(f"\n1. 原始配置 (concat, 5层):")
print(f"   HYBRID_FNO_KAN_HIDDEN_DIMS = {original_concat_hidden_dims}")
print(f"   参数量: {original_concat_params:,}")

print(f"\n2. 原始配置 (multiply, 5层):")
print(f"   HYBRID_FNO_KAN_HIDDEN_DIMS = {original_multiply_hidden_dims}")
print(f"   参数量: {original_multiply_params:,}")
print(f"   相比concat减少: {original_concat_params - original_multiply_params:,} ({(original_concat_params - original_multiply_params) / original_concat_params * 100:.1f}%)")

print(f"\n3. 保守调整 (multiply, 5层):")
print(f"   HYBRID_FNO_KAN_HIDDEN_DIMS = {conservative_hidden_dims}")
print(f"   参数量: {conservative_params:,}")
print(f"   相比原始multiply减少: {original_multiply_params - conservative_params:,} ({(original_multiply_params - conservative_params) / original_multiply_params * 100:.1f}%)")
print(f"   相比原始concat减少: {original_concat_params - conservative_params:,} ({(original_concat_params - conservative_params) / original_concat_params * 100:.1f}%)")

print(f"\n4. 浅层网络 (multiply, 4层, 原始维度):")
print(f"   HYBRID_FNO_KAN_HIDDEN_DIMS = {shallow_original_hidden_dims}")
print(f"   参数量: {shallow_original_params:,}")
print(f"   相比原始multiply减少: {original_multiply_params - shallow_original_params:,} ({(original_multiply_params - shallow_original_params) / original_multiply_params * 100:.1f}%)")
print(f"   相比原始concat减少: {original_concat_params - shallow_original_params:,} ({(original_concat_params - shallow_original_params) / original_concat_params * 100:.1f}%)")

print(f"\n5. 保守调整 + 浅层网络 (multiply, 4层, 保守维度) ⭐:")
print(f"   HYBRID_FNO_KAN_HIDDEN_DIMS = {combined_hidden_dims}")
print(f"   参数量: {combined_params:,}")
print(f"   相比原始multiply减少: {original_multiply_params - combined_params:,} ({(original_multiply_params - combined_params) / original_multiply_params * 100:.1f}%)")
print(f"   相比原始concat减少: {original_concat_params - combined_params:,} ({(original_concat_params - combined_params) / original_concat_params * 100:.1f}%)")

print("\n" + "=" * 80)
print("方案5的优势")
print("=" * 80)
print(f"""
配置: HYBRID_FNO_KAN_HIDDEN_DIMS = {combined_hidden_dims}

特点：
1. 层数减少：从5层减少到4层
   - 降低模型复杂度
   - 减少过拟合风险
   - 提升训练速度

2. 维度保守：使用保守的隐藏层维度
   - 第一层从256降到150（减少41%）
   - 第二层从256降到150（减少41%）
   - 第三层从128降到75（减少41%）
   - 第四层从128降到75（减少41%）
   - 与输入维度减少比例（45.9%）大致匹配

3. 参数量大幅减少：
   - 相比原始concat配置减少 {(original_concat_params - combined_params) / original_concat_params * 100:.1f}%
   - 相比原始multiply配置减少 {(original_multiply_params - combined_params) / original_multiply_params * 100:.1f}%
   - 训练速度显著提升

4. 平衡性好：
   - 既有浅层网络的简洁性
   - 又有保守调整的参数效率
   - 适合中等规模数据集

推荐使用此配置！
""")

print("\n" + "=" * 80)
print("配置建议")
print("=" * 80)
print(f"""
在 config.py 中修改：

# Hybrid FNO+KAN hyperparameters
HYBRID_FNO_KAN_HIDDEN_DIMS = {combined_hidden_dims}

# FourierKAN hyperparameters (建议保持5层或也改为4层）
FOURIER_KAN_HIDDEN_DIMS = [150, 150, 75, 75, 64]  # 或 [150, 150, 75, 75]

其他可调整的参数：
- HYBRID_FNO_KAN_N_FREQUENCIES = 3  # 从4减少到3
- HYBRID_FNO_KAN_SPECTRAL_FREQ_RATIO = 0.4  # 从0.5减少到0.4
- HYBRID_FNO_KAN_DROPOUT = 0.25  # 从0.2增加到0.25（防止过拟合）
""")
