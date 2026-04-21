# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = [('ffmpeg_build/ffmpeg', '.'), ('ffmpeg_build/ffprobe', '.')]
hiddenimports = ['soundfile', 'numpy', 'numpy.core.multiarray', 'numpy.core._multiarray_umath', 'demucs', 'demucs.pretrained', 'demucs.apply', 'demucs.audio', 'demucs.states', 'demucs.hdemucs', 'demucs.htdemucs', 'demucs.repo', 'demucs.utils', 'diffq', 'openunmix', 'torch', 'torchaudio', 'torchaudio.functional', 'yt_dlp', 'yt_dlp.extractor', 'tqdm']
tmp_ret = collect_all('demucs')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('diffq')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('openunmix')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('yt_dlp')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['karaoke_maker.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KaraokeMaker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='KaraokeMaker',
)
app = BUNDLE(
    coll,
    name='KaraokeMaker.app',
    icon=None,
    bundle_identifier='com.karaokemaker.app',
)
