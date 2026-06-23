# Video Karaoke Maker

Strips the vocals out of a song so you can sing over it, with a built-in player to change the key and speed live. Feed it a video file or a YouTube link and you get a karaoke version back.

Built with Python/Tkinter on top of [Demucs](https://github.com/facebookresearch/demucs) (vocal removal), FFmpeg, and mpv.

## Two tabs

- **Convert Karaoke** — drop in a video/audio file or paste a YouTube URL, and it removes the vocals. Seconds on an NVIDIA GPU, a few minutes on CPU.
- **Playback** — open any video (including one you just made) and tweak **key** and **tempo** with sliders while you sing. They're independent: change the key without changing the speed, and vice-versa. Has fullscreen and a "save this version" export.

## Just want to use it?

No Python or installs needed. Download from [Releases](../../releases), unzip, and run `KaraokeMaker.exe` (Windows) or open `KaraokeMaker.app` (Mac). Everything's bundled. On Mac, right-click → Open the first time (it's unsigned).

## How it works

**Convert tab** — pull the vocals out and rebuild the video:

```
Input Video ──► FFmpeg (extract audio) ──► Demucs (remove vocals) ──► FFmpeg (merge back) ──► Karaoke Video
```

The model (~80 MB) ships inside the app, so it works offline.

**Playback tab** — play it back, adjust live, optionally save a copy:

```
Any Video ──► mpv player ──► adjust key/tempo live ──► (Save) FFmpeg re-render ──► New Video
```

Key and tempo are independent — change one without touching the other.

## Building from source

Pick one:

- **`build_windows.bat`** — GPU build (CUDA). Fastest, ~4 GB. Needs Python 3.10–3.12 (no CUDA wheels for 3.13).
- **`build_windows_cpu.bat`** — CPU build. ~1 GB, runs on any PC, works on Python 3.10–3.13, but separation is slower.
- **`build_mac.sh`** — Mac build (CPU). Needs [Homebrew](https://brew.sh) + Python 3.10–3.12; it pulls in FFmpeg/mpv for you.

The Playback tab needs `libmpv-2.dll` (Windows) — the script bundles it, but drop one next to the script if it's missing. Grab it from the [shinchiro builds](https://github.com/shinchiro/mpv-winbuild-cmake/releases) (`mpv-dev-x86_64-*.7z`). Without it, only the Playback tab is disabled.

### Run from source (dev)

```
python -m venv venv && venv\Scripts\activate
pip install torch torchaudio        # add --index-url .../whl/cu121 for GPU
pip install demucs soundfile yt-dlp python-mpv
python karaoke_maker.py
```

FFmpeg, ffprobe, and libmpv need to be on PATH or in the project folder.

## Notes

- Vocal separation is RAM/CPU heavy — rough on 4 GB laptops. Playback is light.
- Tempo exports re-encode the video (a few minutes); pitch-only exports are instant.

Built on [Demucs](https://github.com/facebookresearch/demucs), [FFmpeg](https://ffmpeg.org/), [mpv](https://mpv.io/), and [yt-dlp](https://github.com/yt-dlp/yt-dlp).