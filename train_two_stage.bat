@echo off
REM 两阶段训练启动脚本（Windows）

echo ========================================
echo 两阶段训练（预训练 + 微调）
echo ========================================
echo.
echo 阶段 1: COSMOS 数据集预训练
echo 阶段 2: Kaggle 数据集微调
echo.

REM 运行两阶段训练
python main.py --two-stage

echo.
echo ========================================
echo 训练完成！
echo ========================================
echo.
echo 输出文件：
echo - 预训练模型: models\pretrained_model.pt
echo - 提交文件: submissions\submission_*.csv
echo.
pause
