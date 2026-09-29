@echo off
REM ============================================================
REM  D'QUILANE Local Trainer - One-time Setup (Windows)
REM ============================================================
REM  What this does:
REM    1. Checks Python is installed
REM    2. Creates virtual environment (venv)
REM    3. Installs PyTorch with CUDA support (untuk GPU kamu)
REM    4. Installs numpy
REM ============================================================

echo ============================================================
echo   D'QUILANE Local Trainer - Setup
echo ============================================================
echo.

REM Cek Python
echo [1/4] Cek Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python belum ter-install atau belum di PATH.
    echo.
    echo Cara fix:
    echo   1. Download Python dari https://www.python.org/downloads/
    echo   2. Saat install, CENTANG "Add Python to PATH"
    echo   3. Setelah install selesai, run setup.bat lagi
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYVER=%%i
echo   OK: %PYVER%
echo.

REM Bikin venv
echo [2/4] Bikin virtual environment...
if exist venv (
    echo   venv sudah ada, skip bikin baru.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo   [ERROR] Gagal bikin venv.
        pause
        exit /b 1
    )
    echo   OK: venv dibuat.
)
echo.

REM Activate venv
call venv\Scripts\activate.bat

REM Install PyTorch dengan CUDA
echo [3/4] Install PyTorch dengan CUDA support...
echo   (Ini download besar ~2.5 GB, butuh waktu 5-10 menit)
echo.
pip install torch --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo.
    echo   [WARNING] CUDA install gagal, coba CPU fallback...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
)
echo.

REM Install numpy
echo [4/4] Install numpy...
pip install numpy
if errorlevel 1 (
    echo   [ERROR] Gagal install numpy.
    pause
    exit /b 1
)
echo.

REM Cek CUDA available
echo ============================================================
echo   Setup Selesai! Verifikasi:
echo ============================================================
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

echo.
echo ============================================================
echo   NEXT STEPS:
echo ============================================================
echo   1. Run train.bat   (train model di GPU)
echo   2. Run serve.bat   (start local web UI)
echo.
echo Kalau ada masalah, baca README di:
echo   https://ariko13.github.io/D-Quilane.LLM/train.html
echo.
pause
