"""
计算模型参数量，用于对比不同配置的参数规模
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

# 当前配置
concat_input_dim = 303  # 208特征 + 95 mask
multiply_input_dim = 208  # 208特征
output_dim = 95  # 95个通道

# Hybrid FNO+KAN 配置
hybrid_hidden_dims = [256, 256, 128, 128, 128]

# 计算参数量
concat_params = calculate_model_params(concat_input_dim, hybrid_hidden_dims, output_dim)
multiply_params = calculate_model_params(multiply_input_dim, hybrid_hidden_dims, output_dim)

print("=" * 80)
print("参数量对比分析")
print("=" * 80)
print(f"\n输入维度：")
print(f"  concat策略: {concat_input_dim}维")
print(f"  multiply策略: {multiply_input_dim}维")
print(f"  减少: {concat_input_dim - multiply_input_dim}维 ({(concat_input_dim - multiply_input_dim) / concat_input_dim * 100:.1f}%)")

print(f"\n当前配置 (HYBRID_FNO_KAN_HIDDEN_DIMS = {hybrid_hidden_dims})：")
print(f"  concat策略参数量: {concat_params:,}")
print(f"  multiply策略参数量: {multiply_params:,}")
print(f"  减少: {concat_params - multiply_params:,} ({(concat_params - multiply_params) / concat_params * 100:.1f}%)")

# 建议的配置
suggested_hidden_dims = [192, 192, 96, 96, 96]
suggested_params = calculate_model_params(multiply_input_dim, suggested_hidden_dims, output_dim)

print(f"\n建议配置 (HYBRID_FNO_KAN_HIDDEN_DIMS = {suggested_hidden_dims})：")
print(f"  multiply策略参数量: {suggested_params:,}")
print(f"  相比当前multiply配置减少: {multiply_params - suggested_params:,} ({(multiply_params - suggested_params) / multiply_params * 100:.1f}%)")
print(f"  相比原始concat配置减少: {concat_params - suggested_params:,} ({(concat_params - suggested_params) / concat_params * 100:.1f}%)")

# 更激进的配置
aggressive_hidden_dims = [128, 128, 64, 64, 64]
aggressive_params = calculate_model_params(multiply_input_dim, aggressive_hidden_dims, output_dim)

print(f"\n激进配置 (HYBRID_FNO_KAN_HIDDEN_DIMS = {aggressive_hidden_dims})：")
print(f"  multiply策略参数量: {aggressive_params:,}")
print(f"  相比原始concat配置减少: {concat_params - aggressive_params:,} ({(concat_params - aggressive_params) / concat_params * 100:.1f}%)")

# 浅层配置
shallow_hidden_dims = [256, 256, 128, 128]  # 4层
shallow_params = calculate_model_params(multiply_input_dim, shallow_hidden_dims, output_dim)

print(f"\n浅层配置 (HYBRID_FNO_KAN_HIDDEN_DIMS = {shallow_hidden_dims})：")
print(f"  multiply策略参数量: {shallow_params:,}")
print(f"  相比当前multiply配置减少: {multiply_params - shallow_params:,} ({(multiply_params - shallow_params) / multiply_params * 100:.1f}%)")

print("\n" + "=" * 80)
print("总结建议")
print("=" * 80)
print("""
1. 保守调整（推荐）：
   - 使用建议配置: [192, 192, 96, 96, 96]
   - 参数量减少约25%，与输入维度减少比例匹配
   - 适合大多数场景，风险较低

2. 激进调整：
   - 使用激进配置: [128, 128, 64, 64, 64]
   - 参数量减少约50%
   - 适合数据量大、训练时间受限的场景

3. 浅层网络：
   - 使用浅层配置: [256, 256, 128, 128]
   - 减少一层，参数量减少约20%
   - 适合防止过拟合的场景

建议先尝试保守调整，如果性能可以接受，再考虑更激进的调整。
""")
