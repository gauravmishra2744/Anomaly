@echo off
cls
echo ========================================
echo ANOMALY DETECTION SYSTEM
echo Production Deployment
echo ========================================
echo.
echo Installing dependencies...
pip install -r requirements.txt -q
echo.
echo Starting production server...
echo.
echo Server will be available at:
echo   http://localhost:8080
echo.
echo Press Ctrl+C to stop the server
echo.
timeout /t 2
start http://localhost:8080
python deploy.py
