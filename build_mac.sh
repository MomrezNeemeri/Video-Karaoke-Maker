#!/bin/bash
# ══════════════════════════════════════════════════════
#   Build Video Karaoke Maker for macOS
#   Creates: dist/KaraokeMaker.app
# ══════════════════════════════════════════════════════

set -e

# Always run from the directory where this script lives
cd "$(dirname "$0")"

echo ""
echo "========================================"
echo "  Building Video Karaoke Maker (macOS)"
echo "========================================"
echo ""

# Step 1: Find Python 3.12 (Homebrew) or fall back to python3
PYTHON=""
if [ -x "/opt/homebrew/opt/python@3.12/bin/python3.12" ]; then
    PYTHON="/opt/homebrew/opt/python@3.12/bin/python3.12"
elif command -v python3.12 &> /dev/null; then
    PYTHON="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    echo "ERROR: Python not found. Install via: brew install python@3.12 python-tk@3.12"
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"

# Verify tkinter works
if ! $PYTHON -c "import tkinter" 2>/dev/null; then
    echo "ERROR: tkinter not found. Install via: brew install python-tk@3.12"
    exit 1
fi
echo "tkinter: OK"

# Step 2: Create virtual environment
echo "[1/5] Creating virtual environment..."
rm -rf build_env
$PYTHON -m venv build_env
source build_env/bin/activate

# Step 3: Install dependencies
echo "[2/5] Installing dependencies (this takes a few minutes)..."
pip install --upgrade pip
pip install torch==2.5.1 torchaudio==2.5.1
pip install demucs soundfile pyinstaller yt-dlp

# Uninstall torchcodec if present
pip uninstall torchcodec -y 2>/dev/null || true

# Step 4: Get FFmpeg
echo "[3/5] Setting up FFmpeg..."
mkdir -p ffmpeg_build
if [ ! -f ffmpeg_build/ffmpeg ]; then
    # Try copying from Homebrew first
    BREW_FFMPEG="$(brew --prefix ffmpeg 2>/dev/null)/bin/ffmpeg"
    BREW_FFPROBE="$(brew --prefix ffmpeg 2>/dev/null)/bin/ffprobe"
    if [ -f "$BREW_FFMPEG" ]; then
        cp "$BREW_FFMPEG" ffmpeg_build/ffmpeg
        cp "$BREW_FFPROBE" ffmpeg_build/ffprobe
        echo "Copied FFmpeg from Homebrew."
    else
        echo "ERROR: FFmpeg not found. Install via: brew install ffmpeg"
        exit 1
    fi
fi
chmod +x ffmpeg_build/ffmpeg ffmpeg_build/ffprobe

# Step 5: Build with PyInstaller
echo "[4/5] Building application with PyInstaller..."
pyinstaller \
    --name "KaraokeMaker" \
    --onedir \
    --windowed \
    --noconfirm \
    --clean \
    --add-binary "ffmpeg_build/ffmpeg:." \
    --add-binary "ffmpeg_build/ffprobe:." \
    --hidden-import "soundfile" \
    --hidden-import "numpy" \
    --hidden-import "numpy.core.multiarray" \
    --hidden-import "numpy.core._multiarray_umath" \
    --hidden-import "demucs" \
    --hidden-import "demucs.pretrained" \
    --hidden-import "demucs.apply" \
    --hidden-import "demucs.audio" \
    --hidden-import "demucs.states" \
    --hidden-import "demucs.hdemucs" \
    --hidden-import "demucs.htdemucs" \
    --hidden-import "demucs.repo" \
    --hidden-import "demucs.utils" \
    --hidden-import "diffq" \
    --hidden-import "openunmix" \
    --hidden-import "torch" \
    --hidden-import "torchaudio" \
    --hidden-import "torchaudio.functional" \
    --hidden-import "yt_dlp" \
    --hidden-import "yt_dlp.extractor" \
    --hidden-import "tqdm" \
    --collect-all "demucs" \
    --collect-all "diffq" \
    --collect-all "openunmix" \
    --collect-all "numpy" \
    --collect-all "yt_dlp" \
    --osx-bundle-identifier "com.karaokemaker.app" \
    karaoke_maker.py

echo "[5/5] Done!"

deactivate

echo ""
echo "========================================"
echo "  BUILD COMPLETE!"
echo "  App: dist/KaraokeMaker.app"
echo "========================================"
echo ""
echo "To create a DMG for distribution:"
echo "  hdiutil create -volname KaraokeMaker -srcfolder dist/KaraokeMaker.app -ov -format UDZO KaraokeMaker.dmg"
echo ""
