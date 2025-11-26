@echo off
echo ================================================================================
echo   ANOMALY DETECTION SYSTEM WITH ENHANCED XAI
echo ================================================================================
echo.
echo [1] Testing XAI Module...
python test_xai.py
echo.
echo ================================================================================
echo [2] Starting Flask Application...
echo ================================================================================
echo.
echo Application will start on: http://localhost:5000
echo.
echo XAI Features Available:
echo   - Multi-method feature importance
echo   - Severity classification (CRITICAL/HIGH/MEDIUM/LOW/NORMAL)
echo   - Pattern analysis
echo   - Contributing factors identification
echo   - Top important timesteps
echo   - Detailed explanations
echo.
echo Press Ctrl+C to stop the server
echo.
python app.py
