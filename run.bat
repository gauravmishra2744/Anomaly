#!/bin/bash
# Quick Setup and Run Script for LSTM Anomaly Detection Project

echo "=================================="
echo "LSTM Anomaly Detection Project"
echo "Quick Setup & Run"
echo "=================================="
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Install dependencies
echo Installing required packages...
pip install numpy pandas scikit-learn --quiet
echo Dependencies installed successfully!
echo.

REM Run analysis
echo Running anomaly detection analysis...
echo.
python run_anomaly_detection.py

echo.
echo =================================="
echo "Analysis Complete!"
echo "=================================="
echo.
echo Next steps:
echo 1. Review the analysis output above
echo 2. Check the README.md for detailed documentation
echo 3. To retrain the model, run: python lstm_autoencoder_train.py
echo 4. To preprocess raw data, run: python data_preprocessing.py
echo.
pause
