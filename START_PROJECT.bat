@echo off
echo ========================================
echo LSTM Anomaly Detection System
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo Starting project...
python run_project.py

pause