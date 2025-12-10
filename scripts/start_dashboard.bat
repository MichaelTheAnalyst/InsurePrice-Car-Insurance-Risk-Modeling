@echo off
REM ========================================================
REM 🚗 InsurePrice Dashboard Auto-Launcher (Windows)
REM ========================================================
REM
REM This batch file automatically starts the InsurePrice dashboard
REM with all necessary checks and error handling.
REM
REM Usage: Double-click this file or run from command prompt
REM
REM Author: Masood Nazari
REM Date: December 2025
REM ========================================================

echo 🚗 Starting InsurePrice Dashboard...
echo ========================================================
echo 🎨 AI-Powered Car Insurance Risk Modeling
echo 🎯 Interactive Dashboard with Color Psychology
echo ========================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if dashboard file exists
if not exist "insureprice_dashboard.py" (
    echo ❌ ERROR: insureprice_dashboard.py not found!
    echo Please ensure you're running this from the correct directory.
    pause
    exit /b 1
)

REM Check if data file exists
if not exist "Enhanced_Synthetic_Car_Insurance_Claims.csv" (
    echo ❌ ERROR: Required data file not found!
    echo Please ensure Enhanced_Synthetic_Car_Insurance_Claims.csv exists.
    pause
    exit /b 1
)

REM Check if actuarial engine exists
if not exist "actuarial_pricing_engine.py" (
    echo ❌ ERROR: Required actuarial engine not found!
    echo Please ensure actuarial_pricing_engine.py exists.
    pause
    exit /b 1
)

echo ✅ All required files verified
echo.

REM Try to activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
) else if exist ".venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
) else (
    echo ℹ️  No virtual environment found, using system Python
)

echo.
echo 🚀 Launching dashboard on http://localhost:8501
echo Press Ctrl+C to stop the dashboard
echo.

REM Launch the dashboard
python -m streamlit run insureprice_dashboard.py --server.port 8501 --server.address localhost

REM If we get here, the dashboard was stopped
echo.
echo 👋 Dashboard stopped successfully
echo Press any key to exit...
pause >nul
