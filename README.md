# Video Karaoke Maker — Build Guide

Create installable apps for Windows (.exe) and macOS (.app).

---

## Project Structure

```
karaoke-installer/
├── karaoke_maker.py      ← Main app (packagable version)
├── build_windows.bat     ← Windows build script
├── build_mac.sh          ← macOS build script
└── README.md             ← This file
```

---

## Windows — Build .EXE Installer

### Prerequisites
- Python 3.10, 3.11, or 3.12 ( avoid 3.13 — PyInstaller + PyTorch work best on 3.10-3.12)
- Internet connection (to download dependencies)

### Build Steps

1. Open Command Prompt **as Administrator**
2. Navigate to the project folder:
   ```
   cd path\to\karaoke-installer
   ```
3. Run the build script:
   ```
   build_windows.bat
   ```
4. Wait ~10-15 minutes (downloads PyTorch, builds the app)
5. Your app is in: `dist\KaraokeMaker\`

### Distribute

Zip the entire `dist\KaraokeMaker\` folder and share it. Users just unzip and double-click `KaraokeMaker.exe`.

The app is self-contained — **no Python or FFmpeg install needed** for the person running it.

> **Size warning:** The final folder will be ~2-3 GB because it includes the entire PyTorch AI engine. This is normal for AI-powered desktop apps.

---

## macOS — Build .APP Bundle

### Prerequisites
- Python 3.10, 3.11, or 3.12 (install via `brew install python@3.12`)
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew (recommended): [brew.sh](https://brew.sh)
- FFmpeg: `brew install ffmpeg`

### Build Steps

1. Open Terminal
2. Navigate to the project folder:
   ```
   cd path/to/karaoke-installer
   ```
3. Make the script executable and run it:
   ```
   chmod +x build_mac.sh
   ./build_mac.sh
   ```
4. Wait ~10-15 minutes
5. Your app is: `dist/KaraokeMaker.app`

### Create a DMG for Distribution

```bash
hdiutil create -volname "KaraokeMaker" \
  -srcfolder dist/KaraokeMaker.app \
  -ov -format UDZO \
  KaraokeMaker.dmg
```

This creates a `KaraokeMaker.dmg` file users can download and drag to Applications.

### macOS Gatekeeper Note

Since the app isn't signed with an Apple Developer certificate, users will need to:
1. Right-click the app → "Open" (first time only)
2. Click "Open" in the security dialog

To sign the app properly (optional, requires $99/year Apple Developer account):
```bash
codesign --deep --force --sign "Developer ID Application: YOUR NAME" dist/KaraokeMaker.app
```

---

## Troubleshooting

### Build fails with "ModuleNotFoundError"
Add the missing module to the `--hidden-import` list in the build script and rebuild.

### App is too large
To reduce size (~500MB savings), use CPU-only PyTorch. The build scripts already do this for Windows. For macOS, change the pip install line to:
```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### "Model not found" on first run
The AI model (~200MB) downloads automatically on first use. The user needs internet for the first run. To pre-bundle the model:
```python
# Run this once, then copy the model cache into the build
python -c "from demucs.pretrained import get_model; get_model('htdemucs_ft')"
```
The model cache is at:
- Windows: `%USERPROFILE%\.cache\torch\hub\checkpoints\`
- macOS: `~/.cache/torch/hub/checkpoints/`

### PyInstaller + Python 3.13 issues
Use Python 3.10-3.12 for building. Python 3.13 has compatibility issues with PyInstaller and PyTorch.

---

## How It Works Under the Hood

```
┌─────────────┐     ┌──────────┐     ┌───────────────┐     ┌──────────┐
│ Input Video  │────►│  FFmpeg  │────►│  Demucs AI    │────►│  FFmpeg  │────► Karaoke Video
│ (MP4/MKV/..)│     │ Extract  │     │ Remove Vocals │     │ Merge    │     (no vocals)
└─────────────┘     │ Audio    │     │ (PyTorch)     │     │ Back     │
                    └──────────┘     └───────────────┘     └──────────┘
```

**Demucs** by Meta Research is a state-of-the-art music source separation model. It uses a hybrid transformer + U-Net architecture trained on thousands of songs to separate vocals, drums, bass, and other instruments.

The `htdemucs_ft` model (fine-tuned version) produces the best results and is used by default.
