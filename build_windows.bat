@echo off
REM ══════════════════════════════════════════════════════════════
REM   Build Video Karaoke Maker for Windows (Universal)
REM
REM   Single build that uses NVIDIA GPU when available,
REM   automatically falls back to CPU on machines without one.
REM
REM   Bundle size: ~4 GB (includes CUDA libraries)
REM   Works on: any Windows 10/11 PC (GPU optional, accelerates if present)
REM ══════════════════════════════════════════════════════════════

cd /d "%~dp0"

echo.
echo ========================================
echo   Building Video Karaoke Maker
echo   Universal build (GPU + CPU fallback)
echo ========================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.10-3.12 first.
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
if exist build_env rmdir /s /q build_env
python -m venv build_env
call build_env\Scripts\activate

echo [2/5] Installing dependencies (CUDA 12.1 PyTorch, ~2 GB download)...
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install demucs soundfile pyinstaller yt-dlp

REM Always grab the latest yt-dlp to bypass YouTube bot detection
pip install -U yt-dlp

pip uninstall torchcodec -y 2>nul

echo [3/5] Setting up FFmpeg...
if not exist ffmpeg_build mkdir ffmpeg_build
cd ffmpeg_build
if not exist ffmpeg.exe (
    echo Downloading FFmpeg...
    curl -L -o ffmpeg.zip https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
    tar -xf ffmpeg.zip
    copy ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe . >nul
    copy ffmpeg-master-latest-win64-gpl\bin\ffprobe.exe . >nul
)
cd ..

echo [4/5] Building application with PyInstaller...
pyinstaller ^
    --name "KaraokeMaker" ^
    --onedir ^
    --windowed ^
    --noconfirm ^
    --clean ^
    --add-binary "ffmpeg_build\ffmpeg.exe;." ^
    --add-binary "ffmpeg_build\ffprobe.exe;." ^
    --hidden-import "soundfile" ^
    --hidden-import "numpy" ^
    --hidden-import "numpy.core.multiarray" ^
    --hidden-import "numpy.core._multiarray_umath" ^
    --hidden-import "demucs" ^
    --hidden-import "demucs.pretrained" ^
    --hidden-import "demucs.apply" ^
    --hidden-import "demucs.audio" ^
    --hidden-import "demucs.states" ^
    --hidden-import "demucs.hdemucs" ^
    --hidden-import "demucs.htdemucs" ^
    --hidden-import "demucs.repo" ^
    --hidden-import "demucs.utils" ^
    --hidden-import "diffq" ^
    --hidden-import "openunmix" ^
    --hidden-import "torch" ^
    --hidden-import "torchaudio" ^
    --hidden-import "torchaudio.functional" ^
    --hidden-import "yt_dlp" ^
    --hidden-import "yt_dlp.extractor" ^
    --hidden-import "tqdm" ^
    --collect-all "demucs" ^
    --collect-all "diffq" ^
    --collect-all "openunmix" ^
    --collect-all "numpy" ^
    --collect-all "yt_dlp" ^
    --collect-all "torch" ^
    karaoke_maker.py

echo [5/5] Cleaning up...
call deactivate

echo.
echo ========================================
echo   BUILD COMPLETE!
echo   App location: dist\KaraokeMaker\
echo   Run: dist\KaraokeMaker\KaraokeMaker.exe
echo ========================================
echo.
echo To distribute: zip the entire dist\KaraokeMaker folder
echo and share the zip file (^~4 GB).
echo.
echo The app auto-detects GPU at runtime:
echo   - NVIDIA GPU present : uses CUDA (5-10x faster)
echo   - No GPU              : falls back to CPU automatically
echo.
pause
