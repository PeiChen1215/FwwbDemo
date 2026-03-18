@echo off
chcp 65001 >nul
cls

echo ============================================================
echo      Student Behavior Analysis - Training Launcher
echo ============================================================
echo.

REM Check virtual environment
if not exist "fwwb_env\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please create it first:
    echo   python -m venv fwwb_env
    echo   fwwb_env\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

set PYTHON=fwwb_env\Scripts\python.exe

echo [INFO] Python: %PYTHON%
echo.

REM Menu
echo Choose training mode:
echo.
echo   [1] Quick Test (~5 min)     - Baseline + Comparison only
echo   [2] Quick Training (~15 min) - Full pipeline, 10 epochs
echo   [3] Full Training (~60 min)  - Complete SSL, 200 epochs
echo   [4] Custom (manual)
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" goto quick_test
if "%choice%"=="2" goto quick_train
if "%choice%"=="3" goto full_train
if "%choice%"=="4" goto custom
goto invalid

:quick_test
cls
echo ============================================================
echo                     Quick Test Mode
echo ============================================================
echo.
%PYTHON% run_quick_test.py
goto done

:quick_train
cls
echo ============================================================
echo                   Quick Training Mode
echo ============================================================
echo.
echo This will run:
echo   1. Baseline models (LR/RF)
echo   2. Model comparison (Raw vs SSL)
echo   3. Clustering analysis
echo.
echo Note: SSL training with 10 epochs for quick validation.
echo For production, use Full Training (200 epochs).
echo.
pause
%PYTHON% train_all_quick.py
goto done

:full_train
cls
echo ============================================================
echo                    Full Training Mode
echo ============================================================
echo.
echo This will run the COMPLETE training pipeline:
echo   1. Baseline models
echo   2. SSL Pre-training (200 epochs, ~30-60 min)
echo   3. Model comparison
echo   4. Clustering analysis
echo   5. Risk prediction model
echo.
echo Estimated time: 60-90 minutes
echo Press Ctrl+C to cancel at any time.
echo.
pause
%PYTHON% run_full_pipeline.py
goto done

:custom
cls
echo ============================================================
echo                     Custom Mode
echo ============================================================
echo.
echo Available scripts:
echo.
echo   [a] Baseline models              - scripts\train\train_baseline.py
echo   [b] SSL Transformer (full)       - scripts\train\train_ssl_transformer.py
echo   [c] SSL DAE                      - scripts\train\train_ssl_dae.py
echo   [d] Model comparison             - scripts\evaluate\compare_models.py
echo   [e] Clustering analysis          - scripts\evaluate\analyze_clusters.py
echo   [f] Risk prediction              - scripts\train\train_risk_model.py
echo.
set /p custom_choice="Enter choice (a-f): "

if "%custom_choice%"=="a" %PYTHON% scripts\train\train_baseline.py
if "%custom_choice%"=="b" %PYTHON% scripts\train\train_ssl_transformer.py
if "%custom_choice%"=="c" %PYTHON% scripts\train\train_ssl_dae.py
if "%custom_choice%"=="d" %PYTHON% scripts\evaluate\compare_models.py
if "%custom_choice%"=="e" %PYTHON% scripts\evaluate\analyze_clusters.py
if "%custom_choice%"=="f" %PYTHON% scripts\train\train_risk_model.py
goto done

:invalid
echo [ERROR] Invalid choice!
pause
exit /b 1

:done
echo.
echo ============================================================
echo                      Training Complete!
echo ============================================================
echo.
echo Results are saved in:
echo   - outputs\results\
echo   - model_comparison_results.csv
echo.
pause
