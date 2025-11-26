@echo off
cls
echo ========================================
echo ANOMALY DETECTION SYSTEM
echo Cybersecurity Theme
echo ========================================
echo.
echo Backend Status: WORKING
echo Frontend Status: READY
echo.
echo Starting server on port 5000...
echo.
echo Open browser: http://localhost:5000
echo.
timeout /t 2
start http://localhost:5000
python app.py
