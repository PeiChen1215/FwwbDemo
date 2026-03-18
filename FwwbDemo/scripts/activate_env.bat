@echo off
REM 激活A14项目虚拟环境
call "%~dp0fwwb_env\Scripts\activate.bat"
echo [A14虚拟环境已激活]
echo Python版本:
python --version
echo.
echo 可用命令:
echo   python run_baseline_simple.py    - 运行基线模型
echo   python prepare_tabulars3l_inputs.py - 准备数据
echo   python train_tabulars3l_dae_cluster_transformer.py - 训练SSL模型
echo.
cmd /k
