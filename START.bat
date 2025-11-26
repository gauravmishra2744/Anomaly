@echo off
cls
echo ========================================
echo Anomaly Detection System
echo ========================================
echo.
echo Installing dependencies...
pip install -r requirements.txt -q
echo.
echo Starting application...
echo.
echo Opening http://localhost:5000
echo.
timeout /t 2
start http://localhost:5000
python app.py
