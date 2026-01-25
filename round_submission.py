"""
将 submission CSV 文件中的数值保留一位小数
"""
import pandas as pd
import sys

def round_submission(input_file, output_file, decimals=1):
    """
    读取 submission 文件并将数值保留指定小数位
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        decimals: 保留的小数位数，默认为1
    """
    print(f"读取文件: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"文件形状: {df.shape}")
    print(f"列数: {len(df.columns)}")
    
    # ID 列保持不变，其他列保留指定小数位
    # 获取所有数值列（除了 ID）
    numeric_columns = df.columns[1:]  # 跳过第一列 ID
    
    print(f"处理 {len(numeric_columns)} 个数值列...")
    
    # 将数值列转换为 float 并四舍五入
    for col in numeric_columns:
        df[col] = df[col].astype(float).round(decimals)
    
    print(f"保存到: {output_file}")
    df.to_csv(output_file, index=False)
    
    print("完成!")

if __name__ == "__main__":
    from pathlib import Path
    
    # Get the project root directory (assuming this script is in the root or scripts/ folder)
    PROJECT_ROOT = Path(__file__).resolve().parent
    
    input_file = PROJECT_ROOT / "submissions" / "submission_20260125_171724_finetuned.csv"
    output_file = PROJECT_ROOT / "submissions" / "submission_20260125_171724_finetuned_rounded.csv"
    
    round_submission(input_file, output_file, decimals=1)
