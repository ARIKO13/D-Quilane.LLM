@echo off
REM ============================================================
REM  D'QUILANE Local Web UI (Windows)
REM ============================================================
REM  Starts local web server + opens browser to test model
REM ============================================================

echo ============================================================
echo   D'QUILANE Local Web UI
echo ============================================================
echo.

REM Cek venv
if not exist venv (
    echo [ERROR] venv belum ada. Run setup.bat dulu!
    pause
    exit /b 1
)

REM Cek model
if not exist web\dquilane_model.json (
    echo [ERROR] Model belum ada. Run train.bat dulu!
    pause
    exit /b 1
)

REM Activate venv (untuk python)
call venv\Scripts\activate.bat

echo Starting local server di http://localhost:8000
echo Tekan Ctrl+C untuk stop server.
echo.

REM Start server di background, buka browser
start "" http://localhost:8000
python -m http.server 8000 --directory web

REM Kalau sampai sini, server dihentikan
echo.
echo Server stopped.
pause
