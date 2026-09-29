@echo off
REM ============================================================
REM  D'QUILANE Local Trainer - Train di GPU (Windows)
REM ============================================================
REM  What this does:
REM    1. Activate venv (harus sudah setup.bat dulu)
REM    2. Run training dengan GPU (kalau ada)
REM    3. GPU cuma kepake pas training. Setelah selesai, GPU bebas.
REM ============================================================

echo ============================================================
echo   D'QUILANE Local Trainer - Training
echo ============================================================
echo.

REM Cek venv
if not exist venv (
    echo [ERROR] venv belum ada. Run setup.bat dulu!
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate.bat

REM Cek Python file
if not exist dquilane_local.py (
    echo [ERROR] dquilane_local.py tidak ditemukan di folder ini.
    echo Pastikan semua file ada di folder yang sama.
    pause
    exit /b 1
)

REM Cek corpus file
if not exist corpus_v010.py (
    echo [ERROR] corpus_v010.py tidak ditemukan.
    pause
    exit /b 1
)

echo Starting training...
echo   - GPU akan kepake untuk training
echo   - Setelah training selesai, GPU akan balik idle
echo   - Model baru disave ke web\dquilane_model.json
echo.

REM Run training
python dquilane_local.py
if errorlevel 1 (
    echo.
    echo [ERROR] Training gagal. Cek error message di atas.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Training Selesai!
echo ============================================================
echo.
echo Model baru ada di: web\dquilane_model.json
echo.
echo NEXT:
echo   1. Run serve.bat buat test local
echo   2. Atau copy web\dquilane_model.json ke docs\ di repo GitHub
echo      untuk update production
echo.
pause
