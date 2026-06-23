@echo off
REM ==============================================================
REM   Build Video Karaoke Maker for Windows -- CPU-ONLY edition
REM
REM   No NVIDIA CUDA libraries. Smaller download (~1 GB vs ~4 GB).
REM   Runs on ANY Windows 10/11 PC, GPU or not -- but vocal
REM   separation uses the CPU, so it is slower than the GPU build.
REM
REM   Works with Python 3.10 - 3.13 (CPU torch has 3.13 wheels).
REM   Build files go in "_build_workspace_cpu"; final app in
REM   "KaraokeMaker_App_CPU" so it does not clash with the GPU build.
REM ==============================================================

setlocal enableextensions

cd /d "%~dp0"

set "WORK=_build_workspace_cpu"
set "FINAL=KaraokeMaker_App_CPU"
set "WORKABS=%~dp0%WORK%"

echo.
echo ========================================
echo   Building Video Karaoke Maker (CPU-only)
echo   Work folder: %WORK%\
echo ========================================
echo.

REM ---- Pick any Python 3.10-3.13 (CPU torch supports all of them) ----
set "PYEXE="

py -3.12 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYEXE=py -3.12"
    goto have_python
)

py -3.11 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYEXE=py -3.11"
    goto have_python
)

py -3.13 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYEXE=py -3.13"
    goto have_python
)

py -3.10 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYEXE=py -3.10"
    goto have_python
)

REM fall back to whatever 'python' is on PATH
python --version >nul 2>&1
if %errorlevel%==0 (
    set "PYEXE=python"
    goto have_python
)

echo.
echo ============================================================
echo  ERROR: No Python found. Install Python 3.10-3.13 from:
echo    https://www.python.org/downloads/windows/
echo  (tick "Add python.exe to PATH"), then re-run this script.
echo ============================================================
echo.
pause
exit /b 1

:have_python
echo Using Python: %PYEXE%
%PYEXE% --version
echo.

if not exist "%WORK%" mkdir "%WORK%"

echo [1/5] Creating virtual environment...
if exist "%WORK%\build_env" rmdir /s /q "%WORK%\build_env"
%PYEXE% -m venv "%WORK%\build_env"
call "%WORK%\build_env\Scripts\activate.bat"

echo.
echo [2/5] Installing dependencies (CPU PyTorch, much smaller download)...
python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Stop early if torch did not install, instead of building an empty dist.
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  ERROR: PyTorch CPU build failed to install. Check your
    echo  internet connection and Python version, then try again.
    echo ============================================================
    echo.
    pause
    exit /b 1
)

python -m pip install demucs soundfile pyinstaller yt-dlp
python -m pip install python-mpv
python -m pip install -U yt-dlp
python -m pip uninstall torchcodec -y 2>nul

echo.
echo [3/5] Setting up FFmpeg...
if not exist "%WORK%\ffmpeg_build" mkdir "%WORK%\ffmpeg_build"
pushd "%WORK%\ffmpeg_build"
if not exist ffmpeg.exe (
    echo Downloading FFmpeg...
    curl -L -o ffmpeg.zip https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
    tar -xf ffmpeg.zip
    copy ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe . >nul
    copy ffmpeg-master-latest-win64-gpl\bin\ffprobe.exe . >nul
)
popd

echo.
echo [3b/5] Setting up libmpv (for the Playback tab)...
if not exist "%WORK%\mpv_build" mkdir "%WORK%\mpv_build"

if exist libmpv-2.dll (
    copy /y libmpv-2.dll "%WORK%\mpv_build\libmpv-2.dll" >nul
    echo Using existing libmpv-2.dll from project folder.
    goto mpv_done
)

if exist "%WORK%\mpv_build\libmpv-2.dll" goto mpv_done

echo libmpv-2.dll not found locally. Attempting download...
pushd "%WORK%\mpv_build"
REM NOTE: this URL ages out. If it 404s, download the latest
REM   mpv-dev-x86_64-*.7z  from
REM   https://github.com/shinchiro/mpv-winbuild-cmake/releases
REM   extract libmpv-2.dll, and place it next to this script.
curl -L -o mpv.7z https://github.com/shinchiro/mpv-winbuild-cmake/releases/download/20260610/mpv-dev-x86_64-20260610-git-304426c.7z
where 7z >nul 2>&1
if %errorlevel%==0 (
    7z x mpv.7z -y >nul
) else (
    echo WARNING: 7-Zip not found, cannot auto-extract libmpv.
    echo Place libmpv-2.dll next to this script and re-run.
)
popd

:mpv_done
if not exist "%WORK%\mpv_build\libmpv-2.dll" (
    echo.
    echo ============================================================
    echo  WARNING: libmpv-2.dll is missing. The app will still build,
    echo  but the Playback tab will be disabled. Place libmpv-2.dll
    echo  next to this script and re-run to include it.
    echo ============================================================
    echo.
    pause
)

echo.
echo [3c/5] Pre-downloading the AI model (so users never have to)...
set "MODELDIR=%WORKABS%\model_cache"
if not exist "%MODELDIR%\checkpoints" mkdir "%MODELDIR%\checkpoints"
python -c "import torch, os; torch.hub.set_dir(r'%MODELDIR%'); from demucs.pretrained import get_model; get_model('htdemucs'); print('Model ready.')"
if errorlevel 1 (
    echo WARNING: model pre-download failed. The app will still build, but
    echo users will download the model on first run instead. Check internet.
)

echo.
echo [4/5] Building application with PyInstaller...

pyinstaller ^
    --name "KaraokeMaker" ^
    --onedir ^
    --windowed ^
    --noconfirm ^
    --clean ^
    --workpath "%WORK%\build" ^
    --distpath "%WORK%\dist" ^
    --specpath "%WORK%" ^
    --add-data "%WORKABS%\model_cache;model_cache" ^
    --add-binary "%WORKABS%\ffmpeg_build\ffmpeg.exe;." ^
    --add-binary "%WORKABS%\ffmpeg_build\ffprobe.exe;." ^
    --add-binary "%WORKABS%\mpv_build\libmpv-2.dll;." ^
    --hidden-import "mpv" ^
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
    --collect-all "yt_dlp" ^
    --collect-all "torch" ^
    --exclude-module "torch.utils.tensorboard" ^
    --exclude-module "tensorboard" ^
    --exclude-module "torch.testing" ^
    --exclude-module "torchaudio.prototype" ^
    --exclude-module "numpy.f2py" ^
    --exclude-module "numpy.distutils" ^
    --exclude-module "numpy.testing" ^
    --exclude-module "matplotlib" ^
    --exclude-module "pandas" ^
    --exclude-module "IPython" ^
    --exclude-module "pytest" ^
    --exclude-module "PIL" ^
    --exclude-module "tkinter.test" ^
    karaoke_maker.py

echo.
echo [5/5] Finishing up...
call deactivate

if exist "%FINAL%" rmdir /s /q "%FINAL%"
if exist "%WORK%\dist\KaraokeMaker" (
    xcopy "%WORK%\dist\KaraokeMaker" "%FINAL%\" /e /i /q >nul
)

echo.
echo ========================================
echo   BUILD COMPLETE! (CPU-only)
echo   App location: %FINAL%\
echo   Run: %FINAL%\KaraokeMaker.exe
echo.
echo   (Build scratch files are in %WORK%\ - delete anytime.)
echo ========================================
echo.
echo To distribute: zip the entire %FINAL% folder and share it.
echo This CPU build runs on any PC but separates vocals slower
echo than the GPU build. No NVIDIA card required.
echo.
pause
endlocal