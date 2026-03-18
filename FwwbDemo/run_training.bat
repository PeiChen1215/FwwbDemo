@echo off
chcp 65001 >nul
echo ============================================================
echo           Student Behavior Analysis - Training
echo ============================================================
echo.

set PYTHON=fwwb_env\Scripts\python.exe

if not exist %PYTHON% (
    echo Error: Virtual environment not found!
    echo Please run: python -m venv fwwb_env
    exit /b 1
)

echo [1/3] Testing environment...
%PYTHON% -c "import torch; print('PyTorch:', torch.__version__)"

echo.
echo [2/3] Running model comparison...
%PYTHON% scripts\evaluate\compare_models.py

echo.
echo [3/3] Choose training mode:
echo   1. Quick test (10 epochs, ~5 min)
echo   2. Full training (200 epochs, ~60 min)
echo   3. Exit

echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Running quick SSL training (10 epochs)...
    %PYTHON% -c "import sys; sys.path.insert(0, '.'); exec(open('scripts/train/train_ssl_transformer.py').read().replace('PRETRAIN_EPOCHS = 200', 'PRETRAIN_EPOCHS = 10'))"
) else if "%choice%"=="2" (
    echo.
    echo Running full SSL training (200 epochs)...
    %PYTHON% scripts\train\train_ssl_transformer.py
) else (
    echo Exiting...
)

echo.
echo ============================================================
echo Training complete! Check outputs/results/ for results.
echo ============================================================
pause
