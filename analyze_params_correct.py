"""
正确计算模型参数量
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

# 当前配置
hybrid_hidden_dims = [256, 256, 128, 128, 128]

# 计算参数量
concat_params = calculate_model_params(concat_input_dim, hybrid_hidden_dims, output_dim)
multiply_params = calculate_model_params(multiply_input_dim, hybrid_hidden_dims, output_dim)

print("=" * 80)
print("正确的参数量对比分析")
print("=" * 80)
print(f"\n输入维度：")
print(f"  concat策略: {concat_input_dim}维")
print(f"    = 95(spectra) + 4(num) + 13(cat) + 95(mask)")
print(f"  multiply策略: {multiply_input_dim}维")
print(f"    = 95(spectra) + 4(num) + 13(cat)")
print(f"    mask已应用到光谱特征，不需要拼接")
print(f"  减少: {concat_input_dim - multiply_input_dim}维 ({(concat_input_dim - multiply_input_dim) / concat_input_dim * 100:.1f}%)")

print(f"\n当前配置 (HYBRID_FNO_KAN_HIDDEN_DIMS = {hybrid_hidden_dims})：")
print(f"  concat策略参数量: {concat_params:,}")
print(f"  multiply策略参数量: {multiply_params:,}")
print(f"  减少: {concat_params - multiply_params:,} ({(concat_params - multiply_params) / concat_params * 100:.1f}%)")

# 建议的配置（按比例缩小）
suggested_hidden_dims = [150, 150, 75, 75, 75]
suggested_params = calculate_model_params(multiply_input_dim, suggested_hidden_dims, output_dim)

print(f"\n建议配置 (HYBRID_FNO_KAN_HIDDEN_DIMS = {suggested_hidden_dims})：")
print(f"  multiply策略参数量: {suggested_params:,}")
print(f"  相比当前multiply配置减少: {multiply_params - suggested_params:,} ({(multiply_params - suggested_params) / multiply_params * 100:.1f}%)")
print(f"  相比原始concat配置减少: {concat_params - suggested_params:,} ({(concat_params - suggested_params) / concat_params * 100:.1f}%)")

# 更激进的配置
aggressive_hidden_dims = [112, 112, 56, 56, 56]
aggressive_params = calculate_model_params(multiply_input_dim, aggressive_hidden_dims, output_dim)

print(f"\n激进配置 (HYBRID_FNO_KAN_HIDDEN_DIMS = {aggressive_hidden_dims})：")
print(f"  multiply策略参数量: {aggressive_params:,}")
print(f"  相比原始concat配置减少: {concat_params - aggressive_params:,} ({(concat_params - aggressive_params) / concat_params * 100:.1f}%)")

# 浅层配置
shallow_hidden_dims = [200, 200, 100, 100]  # 4层
shallow_params = calculate_model_params(multiply_input_dim, shallow_hidden_dims, output_dim)

print(f"\n浅层配置 (HYBRID_FNO_KAN_HIDDEN_DIMS = {shallow_hidden_dims})：")
print(f"  multiply策略参数量: {shallow_params:,}")
print(f"  相比当前multiply配置减少: {multiply_params - shallow_params:,} ({(multiply_params - shallow_params) / multiply_params * 100:.1f}%)")

print("\n" + "=" * 80)
print("总结建议")
print("=" * 80)
print(f"""
输入维度从 {concat_input_dim} 减少到 {multiply_input_dim}，减少了 {(concat_input_dim - multiply_input_dim) / concat_input_dim * 100:.1f}%。

当前配置使用 {hybrid_hidden_dims}，参数量从 {concat_params:,} 减少到 {multiply_params:,}，
只减少了 {(concat_params - multiply_params) / concat_params * 100:.1f}%，远小于输入维度的减少比例。

建议调整：

1. 保守调整（推荐）：
   - 使用建议配置: [150, 150, 75, 75, 75]
   - 参数量: {suggested_params:,} (相比concat减少 {(concat_params - suggested_params) / concat_params * 100:.1f}%)
   - 隐藏层维度约为原来的60%，与输入维度减少比例匹配
   - 适合大多数场景，风险较低

2. 激进调整：
   - 使用激进配置: [112, 112, 56, 56, 56]
   - 参数量: {aggressive_params:,} (相比concat减少 {(concat_params - aggressive_params) / concat_params * 100:.1f}%)
   - 隐藏层维度约为输入维度，参数量大幅减少
   - 适合数据量大、训练时间受限的场景

3. 浅层网络：
   - 使用浅层配置: [200, 200, 100, 100]
   - 参数量: {shallow_params:,} (相比concat减少 {(concat_params - shallow_params) / concat_params * 100:.1f}%)
   - 减少一层，降低模型复杂度
   - 适合防止过拟合的场景

建议先尝试保守调整，如果性能可以接受，再考虑更激进的调整。
""")
