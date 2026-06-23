#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#   Build Video Karaoke Maker for macOS (.app)
#
#   Produces a self-contained app bundle. END USERS install NOTHING —
#   FFmpeg and libmpv are bundled inside the .app.
#
#   You (the builder) need Homebrew + ffmpeg + mpv installed once,
#   so this script can copy their binaries/dylibs into the bundle.
# ══════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"

# everything builds inside this one folder to keep the project tidy
WORK="_build_workspace"
FINAL="KaraokeMaker_App"
mkdir -p "$WORK"

echo
echo "========================================"
echo "   Building Video Karaoke Maker (macOS)"
echo "========================================"
echo

# ---- prerequisites check ----
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found. Install Python 3.10-3.12 (brew install python@3.12)."
    exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
    echo "ERROR: Homebrew not found. Install from https://brew.sh then re-run."
    exit 1
fi

# ffmpeg + mpv provide the binaries/dylibs we bundle. install if missing.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Installing ffmpeg (one-time, for bundling)..."
    brew install ffmpeg
fi
if ! brew list mpv >/dev/null 2>&1; then
    echo "Installing mpv (one-time, provides libmpv for bundling)..."
    brew install mpv
fi

echo "[1/6] Creating virtual environment..."
rm -rf "$WORK/build_env"
python3 -m venv "$WORK/build_env"
source "$WORK/build_env/bin/activate"

echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip
# CPU PyTorch is fine on Mac (Apple Silicon CPU is fast; mps isn't used by demucs here)
pip install torch torchaudio
pip install demucs soundfile pyinstaller yt-dlp
pip install python-mpv
pip install -U yt-dlp
pip uninstall torchcodec -y 2>/dev/null || true

echo "[3/6] Locating FFmpeg binaries to bundle..."
mkdir -p "$WORK/ffmpeg_build"
FFMPEG_BIN="$(command -v ffmpeg)"
FFPROBE_BIN="$(command -v ffprobe)"
cp -f "$FFMPEG_BIN"  "$WORK/ffmpeg_build/ffmpeg"
cp -f "$FFPROBE_BIN" "$WORK/ffmpeg_build/ffprobe"
echo "  ffmpeg:  $FFMPEG_BIN"
echo "  ffprobe: $FFPROBE_BIN"

echo "[4/6] Locating libmpv to bundle..."
mkdir -p "$WORK/mpv_build"
# Homebrew installs libmpv as a versioned dylib. Find the real file.
MPV_PREFIX="$(brew --prefix mpv 2>/dev/null || true)"
LIBMPV=""
for cand in \
    "$MPV_PREFIX/lib/libmpv.dylib" \
    "$MPV_PREFIX/lib/libmpv.2.dylib" \
    "$(brew --prefix)/lib/libmpv.dylib" \
    "$(brew --prefix)/lib/libmpv.2.dylib"; do
    if [ -e "$cand" ]; then
        # resolve symlink to the actual file
        LIBMPV="$(python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$cand")"
        break
    fi
done

if [ -z "$LIBMPV" ] || [ ! -e "$LIBMPV" ]; then
    echo "WARNING: could not locate libmpv.dylib via Homebrew."
    echo "The Playback tab will be disabled. Try: brew reinstall mpv"
else
    cp -f "$LIBMPV" "$WORK/mpv_build/libmpv.2.dylib"
    echo "  libmpv:  $LIBMPV"
fi

echo "[4b/6] Pre-downloading the AI model (so users never have to)..."
MODELDIR="$WORK/model_cache"
mkdir -p "$MODELDIR/checkpoints"
python3 -c "import torch; torch.hub.set_dir('$MODELDIR'); from demucs.pretrained import get_model; get_model('htdemucs'); print('Model ready.')" \
    || echo "WARNING: model pre-download failed; users will download on first run."

echo "[5/6] Building application with PyInstaller..."
ADD_BINARIES=(
    --add-binary "$WORK/ffmpeg_build/ffmpeg:."
    --add-binary "$WORK/ffmpeg_build/ffprobe:."
)
if [ -e "$WORK/mpv_build/libmpv.2.dylib" ]; then
    ADD_BINARIES+=( --add-binary "$WORK/mpv_build/libmpv.2.dylib:." )
fi

pyinstaller \
    --name "KaraokeMaker" \
    --onedir \
    --windowed \
    --noconfirm \
    --clean \
    --workpath "$WORK/build" \
    --distpath "$WORK/dist" \
    --specpath "$WORK" \
    --add-data "$MODELDIR:model_cache" \
    "${ADD_BINARIES[@]}" \
    --hidden-import "mpv" \
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
    --collect-all "yt_dlp" \
    --collect-all "torch" \
    --exclude-module "torch.utils.tensorboard" \
    --exclude-module "tensorboard" \
    --exclude-module "torch.testing" \
    --exclude-module "torchaudio.prototype" \
    --exclude-module "numpy.f2py" \
    --exclude-module "numpy.distutils" \
    --exclude-module "numpy.testing" \
    --exclude-module "matplotlib" \
    --exclude-module "pandas" \
    --exclude-module "IPython" \
    --exclude-module "pytest" \
    --exclude-module "PIL" \
    --exclude-module "tkinter.test" \
    karaoke_maker.py

echo "[6/6] Cleaning up..."
deactivate || true

# move the finished .app out to a clean, easy-to-find folder
rm -rf "$FINAL"
mkdir -p "$FINAL"
if [ -d "$WORK/dist/KaraokeMaker.app" ]; then
    cp -R "$WORK/dist/KaraokeMaker.app" "$FINAL/KaraokeMaker.app"
fi

echo
echo "========================================"
echo "   BUILD COMPLETE!"
echo "   App: $FINAL/KaraokeMaker.app"
echo "   (Build scratch files are in $WORK/ — delete anytime.)"
echo "========================================"
echo
echo "To make a shareable DMG:"
echo "  hdiutil create -volname KaraokeMaker \\"
echo "    -srcfolder $FINAL/KaraokeMaker.app -ov -format UDZO KaraokeMaker.dmg"
echo
echo "Users just drag the .app to Applications and run it — no installs needed."
echo "(First launch: right-click -> Open, since the app isn't Apple-signed.)"
echo