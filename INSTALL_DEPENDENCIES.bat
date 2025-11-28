@echo off
echo ========================================
echo Installing Dependencies
echo ========================================
echo.

echo Installing Python packages...
pip install -r requirements.txt

echo.
echo Setting up environment...
python setup_environment.py

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To start the project, run: START_PROJECT.bat
pause