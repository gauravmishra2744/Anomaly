@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting Anomaly Detection System...
echo.
echo Open browser and go to: http://localhost:5000
echo.

python app.py
pause
