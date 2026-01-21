"""
对比单阶段训练和两阶段训练的效果

运行此脚本会：
1. 运行单阶段训练（使用 both 数据集）
2. 运行两阶段训练（COSMOS 预训练 + Kaggle 微调）
3. 对比两种方法的验证集性能
"""

import sys
import subprocess
from pathlib import Path
import time

def run_training(mode, description):
    """运行训练并记录结果"""
    print("\n" + "="*80)
    print(f"开始 {description}")
    print("="*80)
    
    start_time = time.time()
    
    if mode == "single":
        # 单阶段训练：使用 both 数据集
        cmd = ["python", "main.py"]
    elif mode == "two_stage":
        # 两阶段训练
        cmd = ["python", "main.py", "--two-stage"]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        print(result.stdout)
        
        # 提取性能指标
        metrics = {}
        for line in result.stdout.split('\n'):
            if "Final Validation MSE:" in line:
                metrics['mse'] = float(line.split(':')[1].strip())
            elif "Final Validation RMSE:" in line:
                metrics['rmse'] = float(line.split(':')[1].strip())
            elif "Final Validation MAE:" in line:
                metrics['mae'] = float(line.split(':')[1].strip())
        
        return {
            'success': True,
            'metrics': metrics,
            'time': elapsed_time
        }
        
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        print(e.stderr)
        return {
            'success': False,
            'error': str(e)
        }

def main():
    results = {}
    
    print("="*80)
    print("训练模式对比实验")
    print("="*80)
    print("\n本脚本将依次运行：")
    print("1. 单阶段训练（baseline）")
    print("2. 两阶段训练（预训练 + 微调）")
    print("\n注意：完整运行可能需要较长时间\n")
    
    input("按 Enter 键开始...")
    
    # 运行单阶段训练
    print("\n" + "🔵 " * 20)
    results['single'] = run_training('single', '单阶段训练（Baseline）')
    
    # 运行两阶段训练
    print("\n" + "🟢 " * 20)
    results['two_stage'] = run_training('two_stage', '两阶段训练（预训练 + 微调）')
    
    # 输出对比结果
    print("\n" + "="*80)
    print("训练结果对比")
    print("="*80)
    
    if results['single']['success'] and results['two_stage']['success']:
        print("\n性能指标对比：")
        print("-" * 60)
        print(f"{'指标':<15} {'单阶段':<20} {'两阶段':<20} {'改进':<15}")
        print("-" * 60)
        
        for metric in ['mse', 'rmse', 'mae']:
            single_val = results['single']['metrics'].get(metric, 0)
            two_stage_val = results['two_stage']['metrics'].get(metric, 0)
            
            if single_val > 0:
                improvement = ((single_val - two_stage_val) / single_val) * 100
                improvement_str = f"{improvement:+.2f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{metric.upper():<15} {single_val:<20.6f} {two_stage_val:<20.6f} {improvement_str:<15}")
        
        print("-" * 60)
        print(f"\n训练时间对比：")
        print(f"  单阶段: {results['single']['time']/60:.2f} 分钟")
        print(f"  两阶段: {results['two_stage']['time']/60:.2f} 分钟")
        
        # 推荐
        if results['two_stage']['metrics'].get('rmse', float('inf')) < results['single']['metrics'].get('rmse', float('inf')):
            print("\n✅ 推荐：两阶段训练效果更好！")
        else:
            print("\n⚠️  单阶段训练效果更好，可能需要调整两阶段训练的超参数")
    else:
        print("\n⚠️  部分训练失败，无法进行对比")
        if not results['single']['success']:
            print(f"  单阶段训练失败: {results['single'].get('error', 'Unknown error')}")
        if not results['two_stage']['success']:
            print(f"  两阶段训练失败: {results['two_stage'].get('error', 'Unknown error')}")
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)

if __name__ == "__main__":
    main()
