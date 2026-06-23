@echo off
setlocal

cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo Starting TravelAgent Streamlit UI...
echo Project: %CD%
echo URL: http://localhost:8502
echo.

python -m streamlit run app/streamlit_app.py --server.port 8502

if errorlevel 1 (
    echo.
    echo TravelAgent failed to start. Check that Python dependencies are installed:
    echo python -m pip install -e .
    echo.
    pause
)
