"""
修复提交文件的列名格式，将 calculated_gain_spectra_0 改为 calculated_gain_spectra_00
"""
import pandas as pd
import sys
from pathlib import Path

def fix_submission_columns(input_path, output_path=None):
    """
    修复提交文件的列名格式
    
    Args:
        input_path: 输入的提交文件路径
        output_path: 输出的提交文件路径（可选，默认覆盖原文件）
    """
    print(f"读取文件: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"原始列数: {len(df.columns)}")
    print(f"前5列: {list(df.columns[:5])}")
    
    # 修复列名
    new_columns = []
    for col in df.columns:
        if col == 'ID':
            new_columns.append(col)
        elif col.startswith('calculated_gain_spectra_'):
            # 提取数字部分
            num_str = col.replace('calculated_gain_spectra_', '')
            try:
                num = int(num_str)
                # 重新格式化为两位数字（带前导零）
                new_col = f'calculated_gain_spectra_{num:02d}'
                new_columns.append(new_col)
            except ValueError:
                # 如果已经是正确格式，保持不变
                new_columns.append(col)
        else:
            new_columns.append(col)
    
    df.columns = new_columns
    
    print(f"修复后前5列: {list(df.columns[:5])}")
    
    # 保存文件
    if output_path is None:
        output_path = input_path
    
    df.to_csv(output_path, index=False)
    print(f"已保存到: {output_path}")
    print(f"行数: {len(df)}, 列数: {len(df.columns)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python fix_submission_columns.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    fix_submission_columns(input_file, output_file)
