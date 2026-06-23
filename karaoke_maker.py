"""
karaoke maker - strips vocals out of videos using demucs.
also downloads from youtube. mostly.
"""

# ok so, this whole top bit is gross but it has to be first.
# pyinstaller + pytorch loves spawning child processes that re-run
# the whole script which means a second tk window pops up out of nowhere.
# ask me how i found out
import sys
import os

import multiprocessing
multiprocessing.freeze_support()

# bail out fast if we're a spawned child
if multiprocessing.current_process().name != "MainProcess":
    sys.exit(0)

# extra paranoia, sometimes the above isn't enough
if any('multiprocessing' in str(a) and 'spawn' in str(a) for a in sys.argv):
    sys.exit(0)

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# torchcodec was a nightmare on windows + py3.13 - dlls just refuse to load.
# trick: shove fake empty modules into sys.modules BEFORE torch tries to import
# them. torch checks sys.modules first so it finds these stubs and gives up
# gracefully instead of exploding.
import types

for _name in [
    "torchcodec",
    "torchcodec.decoders",
    "torchcodec.decoders._core",
    "torchcodec._internally_replaced_utils",
]:
    _mod = types.ModuleType(_name)
    _mod.__path__ = []
    _mod.__version__ = "0.0.0"
    sys.modules[_name] = _mod

os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"

# let torch use all the cores. spent way too long with these set to 1 trying
# to debug something else, then forgot to change them back. don't be me
import os as _os_for_cpu
_cpu_count = _os_for_cpu.cpu_count() or 4
os.environ["OMP_NUM_THREADS"] = str(_cpu_count)
os.environ["MKL_NUM_THREADS"] = str(_cpu_count)
os.environ["OPENBLAS_NUM_THREADS"] = str(_cpu_count)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(_cpu_count)
os.environ["NUMEXPR_NUM_THREADS"] = str(_cpu_count)

# windowed pyinstaller apps on windows have stdout=None and tqdm CRASHES on that.
# this null-stream class just absorbs writes and shrugs
class _NullStream:
    def write(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass
    def isatty(self): return False
    def fileno(self): raise OSError("no fileno")
    def close(self): pass
    def writable(self): return True
    def readable(self): return False
    def seekable(self): return False

if sys.stdout is None:
    sys.stdout = _NullStream()
if sys.stderr is None:
    sys.stderr = _NullStream()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
import shutil
import tempfile
import re
import traceback
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import numpy as np

# env vars don't always stick, force it directly
try:
    torch.set_num_threads(_cpu_count)
    torch.set_num_interop_threads(max(1, _cpu_count // 2))
except Exception:
    pass


# replacing torchaudio's save/load with soundfile because torchaudio's default
# backend on windows is sox which we DEFINITELY don't have, and the alternative
# is ffmpeg-based which has its own can of worms. soundfile just works.
def _sf_save(filepath, src, sample_rate, **kwargs):
    filepath = str(filepath)
    data = src.cpu().numpy().T
    ext = os.path.splitext(filepath)[1].lower()
    subtype = "FLOAT" if ext == ".wav" else "PCM_16"
    sf.write(filepath, data, sample_rate, subtype=subtype)

def _sf_load(filepath, **kwargs):
    filepath = str(filepath)
    data, sr = sf.read(filepath, dtype="float32", always_2d=True)
    tensor = torch.from_numpy(data.T)
    return tensor, sr

torchaudio.save = _sf_save
torchaudio.load = _sf_load

# demucs has its own save_audio that goes through torchaudio, gotta patch that too
import demucs.audio
def _patched_save_audio(wav, path, samplerate, **kwargs):
    data = wav.cpu().numpy().T
    sf.write(str(path), data, samplerate, subtype="FLOAT")
demucs.audio.save_audio = _patched_save_audio

from demucs.pretrained import get_model
from demucs.apply import apply_model

# python-mpv powers the Playback tab. it's optional - if libmpv isn't around
# the tab just shows a message instead of taking the whole app down.
#
# annoying gotcha: python-mpv looks up libmpv through the OS loader and flat
# out refuses relative paths in %PATH%. so grab every dir the dll could be in,
# make them absolute, and register them before the import. handles both the
# raw script (dll sitting next to this file) and the frozen exe.
if sys.platform == "win32":
    try:
        _dll_dirs = []
        # where this script lives
        try:
            _dll_dirs.append(os.path.dirname(os.path.abspath(__file__)))
        except Exception:
            pass
        # cwd as a backup
        _dll_dirs.append(os.path.abspath(os.getcwd()))
        # and the usual frozen-exe spots
        if getattr(sys, "frozen", False):
            if hasattr(sys, "_MEIPASS"):
                _dll_dirs.append(sys._MEIPASS)
            _exe_dir = os.path.dirname(sys.executable)
            _dll_dirs.append(_exe_dir)
            _dll_dirs.append(os.path.join(_exe_dir, "_internal"))

        _seen = set()
        for _d in _dll_dirs:
            _d = os.path.abspath(_d) if _d else _d
            if not _d or _d in _seen or not os.path.isdir(_d):
                continue
            _seen.add(_d)
            try:
                os.add_dll_directory(_d)   # the right way to do it on 3.8+
            except Exception:
                pass
            # absolute path here, otherwise python-mpv's find_library throws a fit
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

elif sys.platform == "darwin":
    # same idea on mac but with dylibs - point the loader at the dirs where
    # libmpv could be hiding (next to the script, or inside the .app bundle).
    try:
        _dylib_dirs = []
        try:
            _dylib_dirs.append(os.path.dirname(os.path.abspath(__file__)))
        except Exception:
            pass
        _dylib_dirs.append(os.path.abspath(os.getcwd()))
        if getattr(sys, "frozen", False):
            if hasattr(sys, "_MEIPASS"):
                _dylib_dirs.append(sys._MEIPASS)
            _exe_dir = os.path.dirname(sys.executable)
            _dylib_dirs.append(_exe_dir)
            # .app puts stuff in Contents/Frameworks and Contents/Resources
            _contents = os.path.dirname(_exe_dir)
            _dylib_dirs.append(os.path.join(_contents, "Frameworks"))
            _dylib_dirs.append(os.path.join(_contents, "Resources"))

        _seen = set()
        _abs_dirs = []
        for _d in _dylib_dirs:
            _d = os.path.abspath(_d) if _d else _d
            if not _d or _d in _seen or not os.path.isdir(_d):
                continue
            _seen.add(_d)
            _abs_dirs.append(_d)
        if _abs_dirs:
            _existing = os.environ.get("DYLD_LIBRARY_PATH", "")
            os.environ["DYLD_LIBRARY_PATH"] = os.pathsep.join(_abs_dirs + ([_existing] if _existing else []))
            # python-mpv also reads this one if we hand it the exact file
            for _d in _abs_dirs:
                _cand = os.path.join(_d, "libmpv.2.dylib")
                if os.path.exists(_cand):
                    os.environ["MPV_DYLIB_PATH"] = _cand
                    break
    except Exception:
        pass

try:
    import mpv as _mpv
    MPV_AVAILABLE = True
    MPV_IMPORT_ERROR = None
except Exception as _e:
    _mpv = None
    MPV_AVAILABLE = False
    MPV_IMPORT_ERROR = str(_e)


# htdemucs > htdemucs_ft for our purposes. ft is slightly better but 4x slower
# and tbh the difference is barely audible
DEMUCS_MODEL = "htdemucs"
SUPPORTED_VIDEO = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v")
SUPPORTED_AUDIO = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma")
HIDE_CONSOLE = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if sys.platform == "win32" else 0

URL_REGEX = re.compile(
    r'^(https?://)?(www\.)?'
    r'(youtube\.com|youtu\.be|vimeo\.com|dailymotion\.com|twitch\.tv)'
    r'/.+',
    re.IGNORECASE
)


def get_downloads_folder():
    home = Path.home()
    downloads = home / "Downloads"
    return str(downloads) if downloads.is_dir() else str(home)


def get_ffmpeg_path():
    # ffmpeg might be bundled with the app (pyinstaller) or on the system path.
    # mac .app bundles are weird - the binary ends up in different spots
    # depending on how pyinstaller felt that day, so check everything
    ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    if getattr(sys, 'frozen', False):
        search_dirs = []
        if hasattr(sys, '_MEIPASS'):
            search_dirs.append(sys._MEIPASS)
        exe_dir = os.path.dirname(sys.executable)
        search_dirs.append(exe_dir)
        if sys.platform == "darwin":
            contents_dir = os.path.dirname(exe_dir)
            search_dirs.append(os.path.join(contents_dir, "Frameworks"))
            search_dirs.append(os.path.join(contents_dir, "Resources"))
            app_dir = os.path.dirname(contents_dir)
            search_dirs.append(os.path.dirname(app_dir))
        search_dirs.append(os.path.join(exe_dir, "_internal"))

        for d in search_dirs:
            candidate = os.path.join(d, ffmpeg_name)
            if os.path.isfile(candidate):
                return candidate

    return shutil.which("ffmpeg") or "ffmpeg"


def run_cmd(cmd, error_msg="Command failed"):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=HIDE_CONSOLE)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[-800:]
        out = result.stdout.decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"{error_msg}\n\n{err}\n{out}")
    return result


def is_url(text):
    return bool(URL_REGEX.match(text.strip()))


def sanitize_filename(name, max_length=100):
    # windows hates these chars in filenames, also some of them are just illegal everywhere
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    name = name.strip(' .')
    if len(name) > max_length:
        name = name[:max_length].strip()
    return name or "video"


# special exception so we can show a nice dialog instead of a wall of error text
# when youtube decides we look like a bot
class BotDetectionError(Exception):
    pass


def check_ytdlp():
    try:
        import yt_dlp
        return True, yt_dlp.version.__version__
    except ImportError:
        return False, None


def download_video(url, output_dir, ffmpeg_path, on_progress=None, on_status=None, browser_cookies=None):
    """grab a video at 720p (or whatever's best if 720p doesn't exist).
    returns (path_on_disk, video_title)."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError(
            "yt-dlp is not installed.\n\n"
            "This app was built without YouTube support.\n"
            "Please reinstall the app or install yt-dlp manually."
        )

    on_status = on_status or (lambda s: None)
    on_progress = on_progress or (lambda v: None)

    # ugly format string but it tries 720p mp4 first then degrades gracefully
    format_selector = (
        'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/'
        'bestvideo[height<=720]+bestaudio/'
        'best[height<=720]/'
        'bestvideo+bestaudio/'
        'best'
    )

    # yt-dlp downloads video and audio as 2 separate streams then merges them.
    # this state dict tracks which one we're on so the progress bar doesn't
    # bounce around like crazy (0-50% video, 50-100% audio)
    state = {
        'current_stream': 0,
        'total_streams': 2,
        'last_pct': 0,
    }

    def progress_hook(d):
        status = d.get('status')

        if status == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            if total:
                stream_pct = (d['downloaded_bytes'] / total) * 100
                per_stream = 100 / state['total_streams']
                overall_pct = (state['current_stream'] * per_stream) + (stream_pct * per_stream / 100)
                overall_pct = min(overall_pct, 99)

                state['last_pct'] = overall_pct
                on_progress(overall_pct)

                speed = d.get('speed', 0) or 0
                speed_mb = speed / 1_000_000
                stream_label = "video" if state['current_stream'] == 0 else "audio"
                on_status(f"Downloading {stream_label}... {overall_pct:.0f}% ({speed_mb:.1f} MB/s)")

        elif status == 'finished':
            state['current_stream'] += 1
            if state['current_stream'] < state['total_streams']:
                on_status("Downloading next stream...")
            else:
                on_status("Merging video and audio...")
                on_progress(100)

    ydl_opts = {
        'format': format_selector,
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'ffmpeg_location': ffmpeg_path,
        'progress_hooks': [progress_hook],
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'restrictfilenames': False,
        # learned this the hard way: noplaylist alone isn't enough, also need
        # the playlistend / playlist_items combo or it'll happily download all
        # 200 videos in someone's playlist (RIP my ssd)
        'noplaylist': True,
        'playlistend': 1,
        'playlist_items': '1',
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        },
        # ios and mweb clients have less aggressive bot checks than the desktop one.
        # try them first before falling back to web
        'extractor_args': {
            'youtube': {
                'player_client': ['ios', 'mweb', 'android', 'web'],
                'player_skip': ['configs'],
            }
        },
    }

    if browser_cookies:
        ydl_opts['cookiesfrombrowser'] = (browser_cookies,)
        on_status(f"Using cookies from {browser_cookies}...")

    def _attempt_download():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # if it somehow returns a playlist anyway, just take the first video
            if info.get('_type') == 'playlist' and info.get('entries'):
                info = info['entries'][0]

            downloaded_path = ydl.prepare_filename(info)

            # yt-dlp sometimes returns the wrong extension after merging.
            # check for the actual file
            if not os.path.exists(downloaded_path):
                base = os.path.splitext(downloaded_path)[0]
                for ext in ['.mp4', '.mkv', '.webm']:
                    candidate = base + ext
                    if os.path.exists(candidate):
                        downloaded_path = candidate
                        break

            if not os.path.exists(downloaded_path):
                raise RuntimeError("Download succeeded but output file not found.")

            return downloaded_path, info.get('title', 'video')

    try:
        return _attempt_download()

    except yt_dlp.utils.DownloadError as e:
        err_msg = str(e)
        err_lower = err_msg.lower()

        # the dreaded "Sign in to confirm you're not a bot" message
        if ('sign in to confirm' in err_lower or
            'not a bot' in err_lower or
            'confirm you' in err_lower):

            if browser_cookies:
                # we already tried cookies once and it still failed, give up
                raise BotDetectionError(
                    "YouTube is blocking the download even with browser cookies."
                )

            # try cookies from each browser the user might have
            on_status("⚠️ YouTube needs verification — retrying with browser cookies...")
            on_progress(0)

            default_browsers = {
                'darwin': ['safari', 'chrome', 'firefox', 'brave', 'edge'],
                'win32':  ['chrome', 'edge', 'firefox', 'brave'],
                'linux':  ['firefox', 'chrome', 'chromium', 'brave'],
            }
            candidates = default_browsers.get(sys.platform, ['chrome', 'firefox'])

            last_err = err_msg
            tried = []
            for browser in candidates:
                try:
                    on_status(f"Retrying with {browser} cookies...")
                    ydl_opts['cookiesfrombrowser'] = (browser,)
                    return _attempt_download()
                except Exception as retry_err:
                    err_str = str(retry_err)
                    last_err = err_str
                    # browser not even installed - skip without spamming the user
                    if 'could not find' in err_str.lower() and 'cookies database' in err_str.lower():
                        continue
                    tried.append(browser)
                    continue

            raise BotDetectionError(
                "YouTube is blocking the download from your IP address."
            )

        # yt-dlp version is stale - youtube changed something
        elif any(sig in err_lower for sig in [
            'signature extraction failed',
            'unable to extract',
            'player response',
            'sig cipher',
            'js player',
            'nsig extraction failed',
        ]):
            raise RuntimeError(
                "⚠️ YouTube download failed — yt-dlp appears to be out of date.\n\n"
                "YouTube has likely updated their site. To fix this:\n\n"
                "1. Open Terminal (Mac) or Command Prompt (Windows)\n"
                "2. Run:  pip install -U yt-dlp\n"
                "3. Restart this app\n\n"
                f"Technical error:\n{err_msg[:300]}"
            )
        elif 'private' in err_lower or 'members-only' in err_lower:
            raise RuntimeError("This video is private or members-only and cannot be downloaded.")
        elif 'unavailable' in err_lower or 'removed' in err_lower:
            raise RuntimeError("This video is unavailable or has been removed.")
        elif 'age' in err_lower and 'restricted' in err_lower:
            raise RuntimeError("This video is age-restricted and cannot be downloaded without authentication.")
        else:
            raise RuntimeError(f"Download failed:\n\n{err_msg[:500]}")

    except BotDetectionError:
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error during download:\n\n{str(e)[:500]}")


class KaraokeProcessor:
    """does the actual work. runs in a worker thread so the gui doesn't freeze.
    callbacks are how it talks back to the gui."""

    def __init__(self, input_source, output_path=None, model_name=DEMUCS_MODEL,
                 on_progress=None, on_status=None, on_done=None, on_error=None,
                 keep_vocals=False, is_url=False, delete_source_after=True):
        self.input_source = input_source
        self.output_path = output_path
        self.model_name = model_name
        self.on_progress = on_progress or (lambda v: None)
        self.on_status = on_status or (lambda s: None)
        self.on_done = on_done or (lambda: None)
        self.on_error = on_error or (lambda e: None)
        self.keep_vocals = keep_vocals
        self.is_url = is_url
        self.delete_source_after = delete_source_after
        self.cancelled = False
        self.temp_dir = None
        self.downloaded_video_path = None
        self.ffmpeg = get_ffmpeg_path()

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="karaoke_")

            if self.is_url:
                self._pipeline_url()
            else:
                ext = Path(self.input_source).suffix.lower()
                if ext in SUPPORTED_AUDIO:
                    self._pipeline_audio(self.input_source, self.output_path)
                else:
                    self._pipeline_video(self.input_source, self.output_path)

            if not self.cancelled:
                self.on_done()
        except BotDetectionError as e:
            if not self.cancelled:
                # tag this so the gui knows to show the special dialog
                self.on_error(("__BOT_DETECTION__", str(e)))
        except Exception as e:
            if not self.cancelled:
                self.on_error(f"{e}\n\n{traceback.format_exc()[-600:]}")
        finally:
            # always clean up temp files even if something blew up
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception:
                    pass

    def _pipeline_url(self):
        # download (0-25%) → process video (25-100%)
        self._update("Downloading video from URL...", 0)

        download_dir = get_downloads_folder()
        self.downloaded_video_path, title = download_video(
            self.input_source,
            download_dir,
            self.ffmpeg,
            on_progress=lambda p: self.on_progress(int(p * 0.25)),
            on_status=self.on_status,
        )

        if self.cancelled: return

        # auto-name the output. if "X [karaoke].mp4" already exists, add (1), (2), etc
        if not self.output_path:
            safe_title = sanitize_filename(title)
            self.output_path = os.path.join(download_dir, f"{safe_title} [karaoke].mp4")
            counter = 1
            base_out = self.output_path
            while os.path.exists(self.output_path):
                stem = Path(base_out).stem
                self.output_path = os.path.join(download_dir, f"{stem} ({counter}).mp4")
                counter += 1

        self._pipeline_video(
            self.downloaded_video_path,
            self.output_path,
            progress_offset=25,
            progress_scale=0.75
        )

        # delete the original download unless user said to keep it
        if self.delete_source_after and self.downloaded_video_path and os.path.exists(self.downloaded_video_path):
            try:
                os.remove(self.downloaded_video_path)
                self.on_status("Cleaned up original video.")
            except Exception:
                pass

    def _pipeline_video(self, video_path, output_path, progress_offset=0, progress_scale=1.0):
        # progress_offset/scale lets us reuse this from _pipeline_url where the
        # download already ate the first 25%
        def scaled_progress(v):
            self.on_progress(progress_offset + int(v * progress_scale))

        wav_path = os.path.join(self.temp_dir, "audio.wav")
        inst_path = os.path.join(self.temp_dir, "instrumental.wav")

        # 1. yank the audio out as wav
        self.on_status("Step 1/3 — Extracting audio...")
        scaled_progress(5)
        run_cmd([self.ffmpeg, "-y", "-i", video_path,
                 "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", wav_path],
                "FFmpeg failed to extract audio")
        if self.cancelled: return

        # 2. demucs does its magic
        self.on_status("Step 2/3 — AI removing vocals (takes a while)...")
        scaled_progress(10)
        self._run_demucs(wav_path, inst_path, progress_cb=scaled_progress)
        if self.cancelled: return

        # 3. swap original audio for the vocal-less version, keep original video stream
        self.on_status("Step 3/3 — Building karaoke video...")
        scaled_progress(90)
        run_cmd([self.ffmpeg, "-y", "-i", video_path, "-i", inst_path,
                 "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                 "-shortest", output_path],
                "FFmpeg failed to merge video")
        scaled_progress(100)

    def _pipeline_audio(self, input_path, output_path):
        # same as video pipeline but no video involved, simpler
        wav_path = os.path.join(self.temp_dir, "audio.wav")
        inst_path = os.path.join(self.temp_dir, "instrumental.wav")

        self._update("Step 1/2 — Preparing audio...", 5)
        if input_path.lower().endswith(".wav"):
            shutil.copy2(input_path, wav_path)
        else:
            run_cmd([self.ffmpeg, "-y", "-i", input_path,
                     "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", wav_path],
                    "FFmpeg audio conversion failed")
        if self.cancelled: return

        self._update("Step 2/2 — AI removing vocals (takes a while)...", 10)
        self._run_demucs(wav_path, inst_path)
        if self.cancelled: return

        self._update("Saving...", 92)
        if output_path.lower().endswith(".wav"):
            shutil.copy2(inst_path, output_path)
        else:
            run_cmd([self.ffmpeg, "-y", "-i", inst_path, "-b:a", "320k", output_path],
                    "FFmpeg encode failed")
        self.on_progress(100)

    def _find_bundled_model_dir(self):
        # if we shipped the model with the app there's a model_cache folder
        # with checkpoints/<weights>.th in it. check the usual places.
        candidates = []
        try:
            if getattr(sys, "frozen", False):
                if hasattr(sys, "_MEIPASS"):
                    candidates.append(os.path.join(sys._MEIPASS, "model_cache"))
                exe_dir = os.path.dirname(sys.executable)
                candidates.append(os.path.join(exe_dir, "model_cache"))
                candidates.append(os.path.join(exe_dir, "_internal", "model_cache"))
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "model_cache"))
        except Exception:
            pass
        for c in candidates:
            # only counts if checkpoints/ actually has something in it
            ckpt = os.path.join(c, "checkpoints")
            if os.path.isdir(ckpt) and os.listdir(ckpt):
                return c
        return None

    def _run_demucs(self, wav_path, output_instrumental_path, progress_cb=None):
        progress_cb = progress_cb or self.on_progress

        # if we bundled the model, point torch hub at it so there's no download
        # on first run and the thing works offline. get_model() reads from
        # <hub_dir>/checkpoints/. no bundled model? torch just downloads it
        # the first time like normal.
        bundled = self._find_bundled_model_dir()
        if bundled:
            try:
                torch.hub.set_dir(bundled)
                self.on_status("Loading AI model (bundled, offline)...")
            except Exception:
                self.on_status("Loading AI model...")
        else:
            self.on_status("Loading AI model (first run downloads ~80MB)...")
        progress_cb(12)

        model = get_model(self.model_name)
        model.eval()

        # cuda for nvidia, cpu for everyone else.
        # tried mps (apple silicon gpu) - demucs has conv layers with > 65536
        # output channels and apple's mps just refuses, hard limit. even with
        # the fallback flag it shuffles tensors between cpu and gpu so much
        # that it ends up SLOWER than just using cpu directly. so we don't.
        # M-series cpus are plenty fast anyway, ~1-2 min for a 4 min song
        cuda_ok = False
        try:
            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False

        if cuda_ok:
            device = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "NVIDIA GPU"
            device_name = f"GPU — {gpu_name} (CUDA)"
        else:
            device = "cpu"
            if sys.platform == "darwin":
                # just so the message reads nicer on mac
                import platform as _plat
                is_arm = _plat.machine().lower() in ("arm64", "aarch64")
                device_name = "Apple Silicon CPU" if is_arm else "Intel Mac CPU"
            else:
                # spell out why we're on cpu - usually it's because someone
                # installed the cpu-only torch wheel by accident
                built_with_cuda = getattr(torch.version, "cuda", None)
                if built_with_cuda:
                    device_name = "CPU (CUDA build present, but no GPU detected)"
                else:
                    device_name = "CPU (this PyTorch is CPU-only - no CUDA support)"

        self.on_status(f"Using {device_name}...")
        model.to(device)

        self.on_status("Loading audio...")
        progress_cb(18)
        wav, sr = _sf_load(wav_path)

        # demucs expects a specific sample rate
        if sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
            sr = model.samplerate

        wav = wav.unsqueeze(0).to(device)

        self.on_status("AI is separating vocals — please wait...")
        progress_cb(25)

        # the actual heavy lifting. num_workers=0 means do it in this process
        # because workers would spawn more processes which would re-init torch which... nope
        with torch.no_grad():
            sources = apply_model(model, wav, device=device, progress=False, num_workers=0)

        progress_cb(85)
        if self.cancelled: return

        # demucs splits audio into 4 stems: vocals, drums, bass, other.
        # for karaoke we want everything-but-vocals so just sum the non-vocal stems
        source_names = model.sources
        self.on_status("Saving instrumental track...")

        if "vocals" in source_names:
            vocals_idx = source_names.index("vocals")
            instrumental = torch.zeros_like(sources[0, 0])
            for i, name in enumerate(source_names):
                if name != "vocals":
                    instrumental += sources[0, i]
        else:
            # fallback for weird models that don't have a "vocals" stem
            instrumental = sources[0, 0]

        _sf_save(output_instrumental_path, instrumental.cpu(), sr)

        # if user wants the isolated vocals saved too (e.g. for remixing)
        if self.keep_vocals and "vocals" in source_names:
            vocals = sources[0, vocals_idx]
            vocals_path = str(Path(self.output_path).with_suffix("")) + "_vocals.wav"
            _sf_save(vocals_path, vocals.cpu(), sr)

        progress_cb(88)

    def _update(self, status, progress):
        self.on_status(status)
        self.on_progress(progress)


class KaraokeApp:
    # color palette - dark blue/red theme. picked these out of vibes
    BG        = "#1a1a2e"
    BG2       = "#16213e"
    CARD      = "#0f3460"
    ACCENT    = "#e94560"
    ACCENT2   = "#ff6b81"
    TEXT      = "#eaeaea"
    DIM       = "#8892a4"
    GREEN     = "#2ecc71"
    ENTRY_BG  = "#162447"
    BORDER    = "#1a3a6a"

    MODE_FILE = "file"
    MODE_URL  = "url"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Karaoke Maker")
        self.root.geometry("820x1040")
        self.root.minsize(720, 920)
        self.root.configure(bg=self.BG)

        # without this the app keeps running in the background after window closes
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.processor = None
        self.worker_thread = None
        self.mode = tk.StringVar(value=self.MODE_FILE)
        self.input_path = tk.StringVar()
        self.url_input = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_var = tk.StringVar(value=DEMUCS_MODEL)
        self.keep_vocals = tk.BooleanVar(value=False)
        self.keep_original_video = tk.BooleanVar(value=False)

        # --- Playback tab state ---
        self.player = None                 # the mpv instance (created lazily)
        self.pb_file = tk.StringVar()      # path of the loaded video
        self.pb_pitch = tk.DoubleVar(value=0.0)   # semitones, -12..+12
        self.pb_tempo = tk.DoubleVar(value=1.0)   # speed multiplier, 0.5..1.5
        self._pb_seeking = False           # suppress slider feedback loops
        self._pb_duration = 0.0
        self._pb_fullscreen = False        # Tk-level fullscreen state
        self._closing = False              # set true during shutdown

        self._styles()
        self._build()
        self._update_mode()

    def _device_badge_text(self):
        # little label so you can tell at a glance if the GPU's actually in use
        try:
            if torch.cuda.is_available():
                try:
                    name = torch.cuda.get_device_name(0)
                except Exception:
                    name = "NVIDIA GPU"
                return f"⚡ GPU acceleration ON — {name}"
        except Exception:
            pass
        if sys.platform == "darwin":
            return "🖥  Running on CPU (Mac) — typical for Apple machines"
        built = getattr(torch.version, "cuda", None)
        if built:
            return "🖥  Running on CPU — no NVIDIA GPU detected"
        return "🖥  Running on CPU — this PyTorch has no CUDA support"

    def _styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("App.TFrame", background=self.BG)
        s.configure("Card.TFrame", background=self.CARD)
        s.configure("Title.TLabel", background=self.BG, foreground=self.ACCENT,
                     font=("Segoe UI", 22, "bold"))
        s.configure("Sub.TLabel", background=self.BG, foreground=self.DIM,
                     font=("Segoe UI", 10))
        s.configure("H.TLabel", background=self.CARD, foreground=self.TEXT,
                     font=("Segoe UI", 11, "bold"))
        s.configure("B.TLabel", background=self.CARD, foreground=self.DIM,
                     font=("Segoe UI", 9))
        s.configure("Status.TLabel", background=self.BG, foreground=self.DIM,
                     font=("Segoe UI", 10))
        s.configure("Ok.TLabel", background=self.BG, foreground=self.GREEN,
                     font=("Segoe UI", 11, "bold"))
        s.configure("Big.TButton", background=self.ACCENT, foreground="white",
                     font=("Segoe UI", 12, "bold"), padding=(20, 12))
        s.map("Big.TButton", background=[("active", self.ACCENT2), ("disabled", "#555")])
        s.configure("Sm.TButton", background=self.CARD, foreground=self.TEXT,
                     font=("Segoe UI", 9), padding=(10, 5))
        s.map("Sm.TButton", background=[("active", self.BORDER)])
        s.configure("Bar.Horizontal.TProgressbar",
                     troughcolor=self.BG2, background=self.ACCENT, thickness=18)
        s.configure("Card.TCheckbutton", background=self.CARD, foreground=self.DIM,
                     font=("Segoe UI", 9))
        s.configure("Mode.TRadiobutton", background=self.BG, foreground=self.TEXT,
                     font=("Segoe UI", 10, "bold"))

        # notebook (the tab header)
        s.configure("TNotebook", background=self.BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=self.BG2, foreground=self.DIM,
                     font=("Segoe UI", 11, "bold"), padding=(22, 10), borderwidth=0)
        s.map("TNotebook.Tab",
              background=[("selected", self.CARD)],
              foreground=[("selected", self.TEXT)])
        # playback sliders
        s.configure("Pb.Horizontal.TScale", background=self.BG, troughcolor=self.BG2)

    def _build(self):
        # tab header across the top
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.convert_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.playback_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.convert_tab, text="  🎵  Convert Karaoke  ")
        self.notebook.add(self.playback_tab, text="  ▶  Playback  ")

        self._build_convert_tab(self.convert_tab)
        self._build_playback_tab(self.playback_tab)

    def _build_convert_tab(self, parent):
        m = ttk.Frame(parent, style="App.TFrame", padding=30)
        m.pack(fill=tk.BOTH, expand=True)

        ttk.Label(m, text="Video Karaoke Maker", style="Title.TLabel").pack(pady=(0, 2))
        ttk.Label(m, text="Remove vocals from any video using AI  •  Powered by Meta Demucs",
                  style="Sub.TLabel").pack(pady=(0, 4))
        # tell them up front whether the GPU is doing the work
        ttk.Label(m, text=self._device_badge_text(), style="Sub.TLabel").pack(pady=(0, 16))

        # file vs URL toggle
        mode_frame = ttk.Frame(m, style="App.TFrame")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(mode_frame, text="📁  Local File",
                         variable=self.mode, value=self.MODE_FILE,
                         command=self._update_mode,
                         style="Mode.TRadiobutton").pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(mode_frame, text="🔗  YouTube URL",
                         variable=self.mode, value=self.MODE_URL,
                         command=self._update_mode,
                         style="Mode.TRadiobutton").pack(side=tk.LEFT)

        # file/url cards swap in/out of this container based on mode
        self.input_container = ttk.Frame(m, style="App.TFrame")
        self.input_container.pack(fill=tk.X)

        # local file input
        self.file_card = ttk.Frame(self.input_container, style="Card.TFrame", padding=15)
        ttk.Label(self.file_card, text="INPUT FILE", style="H.TLabel").pack(anchor=tk.W)
        ttk.Label(self.file_card, text="Select a video or audio file to remove vocals from",
                  style="B.TLabel").pack(anchor=tk.W, pady=(2, 8))
        r1 = ttk.Frame(self.file_card, style="Card.TFrame")
        r1.pack(fill=tk.X)
        self._entry(r1, self.input_path)
        ttk.Button(r1, text="Browse...", style="Sm.TButton",
                   command=self._pick_input).pack(side=tk.RIGHT, padx=(8, 0))

        # youtube url input
        self.url_card = ttk.Frame(self.input_container, style="Card.TFrame", padding=15)
        ttk.Label(self.url_card, text="YOUTUBE URL", style="H.TLabel").pack(anchor=tk.W)
        ttk.Label(self.url_card,
                  text="Paste a YouTube link. Video downloads in 720p (or highest available) to Downloads folder.",
                  style="B.TLabel").pack(anchor=tk.W, pady=(2, 8))
        ur = ttk.Frame(self.url_card, style="Card.TFrame")
        ur.pack(fill=tk.X)
        self._entry(ur, self.url_input)

        # output path picker - only shows in file mode (url mode auto-generates)
        self.output_card = ttk.Frame(m, style="Card.TFrame", padding=15)
        ttk.Label(self.output_card, text="OUTPUT FILE", style="H.TLabel").pack(anchor=tk.W)
        ttk.Label(self.output_card, text="Where to save the karaoke version",
                  style="B.TLabel").pack(anchor=tk.W, pady=(2, 8))
        out_row = ttk.Frame(self.output_card, style="Card.TFrame")
        out_row.pack(fill=tk.X)
        self._entry(out_row, self.output_path)
        ttk.Button(out_row, text="Browse...", style="Sm.TButton",
                   command=self._pick_output).pack(side=tk.RIGHT, padx=(8, 0))

        # info card for url mode
        self.url_output_card = ttk.Frame(m, style="Card.TFrame", padding=15)
        ttk.Label(self.url_output_card, text="OUTPUT", style="H.TLabel").pack(anchor=tk.W)
        ttk.Label(self.url_output_card,
                  text="📁 Karaoke video saves automatically to your Downloads folder\n"
                       "(named after the video title, with [karaoke] suffix)",
                  style="B.TLabel").pack(anchor=tk.W, pady=(2, 0))

        # options
        c3 = ttk.Frame(m, style="Card.TFrame", padding=15)
        c3.pack(fill=tk.X, pady=(10, 10))
        ttk.Label(c3, text="OPTIONS", style="H.TLabel").pack(anchor=tk.W)
        ro = ttk.Frame(c3, style="Card.TFrame")
        ro.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(ro, text="AI Model:", style="B.TLabel").pack(side=tk.LEFT)
        ttk.Combobox(ro, textvariable=self.model_var, width=18,
                      values=["htdemucs_ft", "htdemucs", "mdx_extra"],
                      state="readonly").pack(side=tk.LEFT, padx=(8, 20))
        ttk.Checkbutton(ro, text="Also save isolated vocals",
                         variable=self.keep_vocals,
                         style="Card.TCheckbutton").pack(side=tk.LEFT)

        # this checkbox only shows in url mode
        self.keep_orig_check = ttk.Checkbutton(c3, text="Keep original downloaded video (don't auto-delete)",
                                                variable=self.keep_original_video,
                                                style="Card.TCheckbutton")

        ttk.Label(c3,
            text="htdemucs_ft = Best quality (slower)  •  htdemucs = Faster  •  mdx_extra = Alternative",
            style="B.TLabel").pack(anchor=tk.W, pady=(6, 0))

        bf = ttk.Frame(m, style="App.TFrame")
        bf.pack(fill=tk.X, pady=(10, 10))
        self.start_btn = ttk.Button(bf, text="🎵  MAKE KARAOKE", style="Big.TButton",
                                     command=self._start)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.cancel_btn = ttk.Button(bf, text="Cancel", style="Sm.TButton",
                                      command=self._cancel, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.RIGHT, padx=(10, 0))

        self.progress = ttk.Progressbar(m, style="Bar.Horizontal.TProgressbar",
                                         mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(5, 5))
        self.status_lbl = ttk.Label(m, text="Ready — select a file or paste a URL to get started",
                                     style="Status.TLabel")
        self.status_lbl.pack(anchor=tk.W)

    # ----------------------------------------------------------------
    #  PLAYBACK TAB - mpv embedded in a tk frame with live pitch/tempo.
    #  pitch goes through the rubberband filter (tweaked live with
    #  af-command); tempo is just mpv's 'speed' property with
    #  audio-pitch-correction on so speeding up doesn't chipmunk the audio.
    # ----------------------------------------------------------------
    def _build_playback_tab(self, parent):
        m = ttk.Frame(parent, style="App.TFrame", padding=20)
        m.pack(fill=tk.BOTH, expand=True)

        self._pb_title = ttk.Label(m, text="▶ Playback", style="Title.TLabel")
        self._pb_title.pack(pady=(0, 2))
        self._pb_subtitle = ttk.Label(m, text="Play  •  Control Tempo / Pitch",
                                      style="Sub.TLabel")
        self._pb_subtitle.pack(pady=(0, 12))

        if not MPV_AVAILABLE:
            self._build_playback_unavailable(m)
            return

        # file picker row
        file_row = ttk.Frame(m, style="Card.TFrame", padding=12)
        self._pb_file_row = file_row
        file_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(file_row, text="VIDEO FILE", style="H.TLabel").pack(anchor=tk.W)
        pr = ttk.Frame(file_row, style="Card.TFrame")
        pr.pack(fill=tk.X, pady=(6, 0))
        self._entry(pr, self.pb_file)
        ttk.Button(pr, text="Open...", style="Sm.TButton",
                   command=self._pb_pick_file).pack(side=tk.RIGHT, padx=(8, 0))

        # the black box mpv draws video into. expands to fill, with a sane min.
        self._pb_video_parent = m   # remembered so fullscreen can restore it
        self.video_frame = tk.Frame(m, bg="black", height=260)
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.video_frame.pack_propagate(False)

        # transport row: play/pause + seek bar + time + fullscreen
        transport = ttk.Frame(m, style="App.TFrame")
        self._pb_transport = transport   # video is re-packed before this on exit
        transport.pack(fill=tk.X, pady=(0, 6))
        self.pb_play_btn = ttk.Button(transport, text="▶  Play", style="Sm.TButton",
                                      command=self._pb_toggle_play)
        self.pb_play_btn.pack(side=tk.LEFT)
        self.pb_fs_btn = ttk.Button(transport, text="⛶  Fullscreen", style="Sm.TButton",
                                    command=self._pb_toggle_fullscreen)
        self.pb_fs_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.pb_time_lbl = ttk.Label(transport, text="0:00 / 0:00", style="Status.TLabel")
        self.pb_time_lbl.pack(side=tk.RIGHT)
        self.pb_seek = ttk.Scale(transport, from_=0, to=1000, orient=tk.HORIZONTAL,
                                 command=self._pb_on_seek_drag)
        self.pb_seek.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 12))
        # distinguish a user drag from programmatic updates
        self.pb_seek.bind("<ButtonPress-1>", lambda e: setattr(self, "_pb_seeking", True))
        self.pb_seek.bind("<ButtonRelease-1>", self._pb_seek_commit)

        # ── PITCH control ──
        pitch_card = ttk.Frame(m, style="Card.TFrame", padding=14)
        self._pb_pitch_card = pitch_card
        pitch_card.pack(fill=tk.X, pady=(4, 6))
        ph = ttk.Frame(pitch_card, style="Card.TFrame")
        ph.pack(fill=tk.X)
        ttk.Label(ph, text="🎚  KEY / PITCH", style="H.TLabel").pack(side=tk.LEFT)
        self.pb_pitch_lbl = ttk.Label(ph, text="0 semitones", style="B.TLabel")
        self.pb_pitch_lbl.pack(side=tk.RIGHT)
        ttk.Scale(pitch_card, from_=-12, to=12, orient=tk.HORIZONTAL,
                  variable=self.pb_pitch, command=self._pb_on_pitch).pack(fill=tk.X, pady=(8, 0))
        prow = ttk.Frame(pitch_card, style="Card.TFrame")
        prow.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(prow, text="–1", style="Sm.TButton",
                   command=lambda: self._pb_nudge_pitch(-1)).pack(side=tk.LEFT)
        ttk.Button(prow, text="Reset key", style="Sm.TButton",
                   command=lambda: self._pb_set_pitch(0)).pack(side=tk.LEFT, padx=6)
        ttk.Button(prow, text="+1", style="Sm.TButton",
                   command=lambda: self._pb_nudge_pitch(1)).pack(side=tk.LEFT)

        # ── TEMPO control ──
        tempo_card = ttk.Frame(m, style="Card.TFrame", padding=14)
        self._pb_tempo_card = tempo_card
        tempo_card.pack(fill=tk.X, pady=(0, 6))
        th = ttk.Frame(tempo_card, style="Card.TFrame")
        th.pack(fill=tk.X)
        ttk.Label(th, text="⏩  TEMPO / SPEED", style="H.TLabel").pack(side=tk.LEFT)
        self.pb_tempo_lbl = ttk.Label(th, text="1.00×", style="B.TLabel")
        self.pb_tempo_lbl.pack(side=tk.RIGHT)
        ttk.Scale(tempo_card, from_=0.5, to=1.5, orient=tk.HORIZONTAL,
                  variable=self.pb_tempo, command=self._pb_on_tempo).pack(fill=tk.X, pady=(8, 0))
        trow = ttk.Frame(tempo_card, style="Card.TFrame")
        trow.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(trow, text="Reset speed", style="Sm.TButton",
                   command=lambda: self._pb_set_tempo(1.0)).pack(side=tk.LEFT)
        ttk.Label(tempo_card,
                  text="Tempo keeps pitch fixed (and vice-versa) — adjust each independently.",
                  style="B.TLabel").pack(anchor=tk.W, pady=(6, 0))

        # ── EXPORT: save a new video with the current pitch/tempo baked in ──
        export_card = ttk.Frame(m, style="Card.TFrame", padding=14)
        export_card.pack(fill=tk.X, pady=(0, 6))
        self._pb_export_card = export_card
        ttk.Label(export_card, text="💾  SAVE THIS VERSION", style="H.TLabel").pack(anchor=tk.W)
        ttk.Label(export_card,
                  text="Render a new video file with the current key and tempo applied.",
                  style="B.TLabel").pack(anchor=tk.W, pady=(2, 8))
        self.pb_export_btn = ttk.Button(export_card, text="Save adjusted video...",
                                        style="Sm.TButton", command=self._pb_export)
        self.pb_export_btn.pack(anchor=tk.W)

        self.pb_status = ttk.Label(m, text="Open a video to start.", style="Status.TLabel")
        self.pb_status.pack(anchor=tk.W, pady=(2, 0))

    def _build_playback_unavailable(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=20)
        card.pack(fill=tk.X, pady=10)
        ttk.Label(card, text="⚠  Playback engine (mpv) not available",
                  style="H.TLabel").pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(card,
                  text="The live playback tab needs libmpv installed on this machine.\n\n"
                       "• Windows: the libmpv DLL must be bundled or on PATH\n"
                       "• macOS:   brew install mpv\n"
                       "• Linux:   install the libmpv package\n\n"
                       "You can still use the Convert Karaoke tab normally.",
                  style="B.TLabel", justify=tk.LEFT).pack(anchor=tk.W)
        if MPV_IMPORT_ERROR:
            ttk.Label(card, text=f"Details: {MPV_IMPORT_ERROR}",
                      style="B.TLabel", wraplength=600, justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 0))

    # ---- player lifecycle ----
    def _pb_ensure_player(self):
        """create the mpv instance once, bound to the video frame."""
        if self.player is not None:
            return self.player
        wid = str(int(self.video_frame.winfo_id()))
        # pitch-correction on means changing speed won't shift the pitch.
        # also throw in a rubberband filter now so we have something to poke
        # at later when the pitch slider moves.
        self.player = _mpv.MPV(
            wid=wid,
            input_default_bindings=True,
            input_vo_keyboard=True,
            osc=True,   # mpv's own seekbar - nice to have in fullscreen
            keep_open="yes",
            audio_pitch_correction=True,
        )
        # start the rubberband filter at 1.0 (no change) so af-command works
        try:
            self.player["af"] = "rubberband=pitch-scale=1.0"
        except Exception:
            pass

        # keep the seek bar + time label in sync.
        # these fire from mpv's own thread; we marshal back to tk via after().
        # guard against firing while the app is closing (root destroyed) -
        # a stray callback then would raise and could wedge shutdown.
        def _safe_after(fn):
            if getattr(self, "_closing", False) or self.player is None:
                return
            try:
                self.root.after(0, fn)
            except (tk.TclError, RuntimeError):
                pass

        @self.player.property_observer("time-pos")
        def _on_time(_name, value):
            _safe_after(lambda: self._pb_update_time(value))

        @self.player.property_observer("duration")
        def _on_dur(_name, value):
            if value:
                self._pb_duration = value

        @self.player.property_observer("pause")
        def _on_pause(_name, value):
            txt = "▶  Play" if value else "⏸  Pause"
            _safe_after(lambda: self._pb_set_playbtn(txt))

        return self.player

    def _pb_pick_file(self):
        all_ext = " ".join(f"*{e}" for e in SUPPORTED_VIDEO + SUPPORTED_AUDIO)
        p = filedialog.askopenfilename(
            title="Open Video or Audio for Playback",
            filetypes=[("All Supported", all_ext), ("All Files", "*.*")]
        )
        if not p:
            return
        self.pb_file.set(p)
        self._pb_load(p)

    def _pb_load(self, path):
        try:
            player = self._pb_ensure_player()
            player.play(path)
            player.pause = False
            # re-apply current slider values to the freshly loaded file
            self.root.after(300, self._pb_apply_pitch)
            self.root.after(300, self._pb_apply_tempo)
            self._pb_set_status("Playing — drag the sliders to change key and tempo.")
        except Exception as e:
            messagebox.showerror("Playback error", f"Could not play this file:\n\n{e}")

    def _pb_toggle_play(self):
        if self.player is None:
            if self.pb_file.get().strip():
                self._pb_load(self.pb_file.get().strip())
            return
        try:
            self.player.pause = not self.player.pause
        except Exception:
            pass

    def _pb_toggle_fullscreen(self):
        # whatever you do, do NOT reparent the video frame. mpv draws into that
        # frame's window handle and moving it to a new parent kills the handle -
        # you get a black screen. learned that the hard way. so we just toggle
        # the OS fullscreen state and let the frame (which already expands) fill it.
        if self.player is None:
            if self.pb_file.get().strip():
                self._pb_load(self.pb_file.get().strip())
            else:
                messagebox.showinfo("No video", "Open a video first, then go fullscreen.")
            return

        if not getattr(self, "_pb_fullscreen", False):
            self._enter_fullscreen()
        else:
            self._exit_fullscreen()

    def _enter_fullscreen(self):
        self._pb_fullscreen = True
        try:
            self.notebook.select(self.playback_tab)
        except Exception:
            pass
        # hide everything except the video so it fills the screen. note we
        # only pack_forget the siblings - never the video frame itself, see above.
        for w in (self._pb_title, self._pb_subtitle, self._pb_file_row,
                  self._pb_pitch_card, self._pb_tempo_card, self._pb_export_card,
                  self.pb_status):
            try:
                w.pack_forget()
            except Exception:
                pass
        try:
            self.root.attributes("-fullscreen", True)
        except Exception:
            pass
        # bind the exit keys and grab focus so esc/f actually register
        self.root.bind("<Escape>", self._fs_key_exit)
        self.root.bind("<f>", self._fs_key_exit)
        self.root.bind("<F11>", self._fs_key_exit)
        try:
            self.root.focus_force()
        except Exception:
            pass
        try:
            self.pb_fs_btn.config(text="⛶  Exit (Esc)")
        except Exception:
            pass

    def _fs_key_exit(self, _evt=None):
        self._exit_fullscreen()

    def _exit_fullscreen(self):
        if not getattr(self, "_pb_fullscreen", False):
            return
        self._pb_fullscreen = False
        try:
            self.root.attributes("-fullscreen", False)
        except Exception:
            pass
        for seq in ("<Escape>", "<f>", "<F11>"):
            try:
                self.root.unbind(seq)
            except Exception:
                pass
        # put everything back. the video frame and transport never moved, so
        # we just re-pack the hidden widgets around them with before=/after=.
        try:
            # order matters here - file_row first since the others anchor to it
            self._pb_file_row.pack(fill=tk.X, pady=(0, 10), before=self.video_frame)
            self._pb_subtitle.pack(pady=(0, 12), before=self._pb_file_row)
            self._pb_title.pack(pady=(0, 2), before=self._pb_subtitle)
            self._pb_pitch_card.pack(fill=tk.X, pady=(4, 6), after=self._pb_transport)
            self._pb_tempo_card.pack(fill=tk.X, pady=(0, 6), after=self._pb_pitch_card)
            self._pb_export_card.pack(fill=tk.X, pady=(0, 6), after=self._pb_tempo_card)
            self.pb_status.pack(anchor=tk.W, pady=(2, 0), after=self._pb_export_card)
        except Exception:
            pass
        try:
            self.pb_fs_btn.config(text="⛶  Fullscreen")
        except Exception:
            pass

    # ---- pitch ----
    def _pb_on_pitch(self, _val):
        semis = round(self.pb_pitch.get())
        self.pb_pitch_lbl.config(text=f"{semis:+d} semitones" if semis else "0 semitones")
        self._pb_apply_pitch()

    def _pb_nudge_pitch(self, delta):
        self._pb_set_pitch(round(self.pb_pitch.get()) + delta)

    def _pb_set_pitch(self, semis):
        semis = max(-12, min(12, semis))
        self.pb_pitch.set(semis)
        self.pb_pitch_lbl.config(text=f"{semis:+d} semitones" if semis else "0 semitones")
        self._pb_apply_pitch()

    def _pb_apply_pitch(self):
        if self.player is None:
            return
        semis = self.pb_pitch.get()
        scale = 2 ** (semis / 12.0)   # semitones -> frequency multiplier
        try:
            # change the live rubberband filter without restarting playback
            self.player.command("af-command", "rubberband", "set-pitch-scale", f"{scale:.6f}")
        except Exception:
            # fallback: rebuild the filter chain
            try:
                self.player["af"] = f"rubberband=pitch-scale={scale:.6f}"
            except Exception:
                pass

    # ---- tempo ----
    def _pb_on_tempo(self, _val):
        spd = self.pb_tempo.get()
        self.pb_tempo_lbl.config(text=f"{spd:.2f}×")
        self._pb_apply_tempo()

    def _pb_set_tempo(self, spd):
        spd = max(0.5, min(1.5, spd))
        self.pb_tempo.set(spd)
        self.pb_tempo_lbl.config(text=f"{spd:.2f}×")
        self._pb_apply_tempo()

    def _pb_apply_tempo(self):
        if self.player is None:
            return
        try:
            # pitch correction is on, so bumping speed leaves the pitch alone
            self.player.speed = self.pb_tempo.get()
        except Exception:
            pass

    # ---- export adjusted video ----
    def _pb_export(self):
        src = self.pb_file.get().strip()
        if not src or not os.path.isfile(src):
            messagebox.showinfo("No video", "Open a video first, then save an adjusted version.")
            return

        semis = round(self.pb_pitch.get())
        tempo = round(self.pb_tempo.get(), 2)
        if semis == 0 and abs(tempo - 1.0) < 0.001:
            if not messagebox.askyesno(
                "No changes",
                "Pitch and tempo are both at default, so the saved file will "
                "match the original. Save anyway?"):
                return

        # warn that tempo changes re-encode the video and take longer
        if abs(tempo - 1.0) >= 0.001:
            if not messagebox.askyesno(
                "Tempo change re-encodes video",
                "Changing the tempo means the video has to be re-encoded, which "
                "can take a few minutes for a full song. Pitch-only changes are "
                "much faster.\n\nContinue?"):
                return

        base, ext = os.path.splitext(os.path.basename(src))
        tag = []
        if semis:
            tag.append(f"key{semis:+d}")
        if abs(tempo - 1.0) >= 0.001:
            tag.append(f"{tempo:.2f}x")
        suffix = ("_" + "_".join(tag)) if tag else "_adjusted"
        out = filedialog.asksaveasfilename(
            title="Save Adjusted Video",
            defaultextension=ext or ".mp4",
            initialfile=f"{base}{suffix}{ext or '.mp4'}",
            filetypes=[("Video", "*.mp4 *.mkv *.mov"), ("All Files", "*.*")])
        if not out:
            return

        self._pb_export_cancel = False
        self.pb_export_btn.config(state="disabled", text="Saving... (click Cancel)",
                                  command=self._pb_export_cancel_now)
        self._pb_set_status("Rendering... 0%")
        t = threading.Thread(target=self._pb_export_worker,
                             args=(src, out, semis, tempo), daemon=True)
        t.start()

    def _pb_export_cancel_now(self):
        self._pb_export_cancel = True
        self._pb_set_status("Cancelling...")
        proc = getattr(self, "_pb_export_proc", None)
        if proc:
            try:
                proc.kill()
            except Exception:
                pass

    def _pb_export_worker(self, src, out, semis, tempo):
        try:
            ffmpeg = get_ffmpeg_path()
            duration = self._probe_duration(src) or 0

            # two ways to do the pitch shift: rubberband (sounds better) or the
            # old asetrate+atempo trick (uglier but works in any ffmpeg). try
            # rubberband first, fall back if ffmpeg chokes. saves us from having
            # to reliably detect whether rubberband is even compiled in.
            def chain(use_rb):
                pitch_scale = 2 ** (semis / 12.0)
                parts = []
                if abs(tempo - 1.0) >= 0.001:
                    parts.append(f"atempo={tempo:.4f}")
                if semis != 0:
                    if use_rb:
                        parts.append(f"rubberband=pitch-scale={pitch_scale:.6f}")
                    else:
                        # speed the audio up to raise pitch, then atempo it back
                        # to the right length
                        sr = 44100
                        parts.append(f"asetrate={int(sr*pitch_scale)}")
                        parts.append(f"aresample={sr}")
                        parts.append(f"atempo={1.0/pitch_scale:.6f}")
                return ",".join(parts)

            def build_cmd(afilter):
                c = [ffmpeg, "-y", "-i", src]
                if abs(tempo - 1.0) >= 0.001:
                    # changing tempo means re-timing the video too (setpts) and
                    # re-encoding it - no way around that
                    vfactor = 1.0 / tempo
                    c += ["-filter_complex",
                          f"[0:v]setpts={vfactor:.6f}*PTS[v];[0:a]{afilter}[a]",
                          "-map", "[v]", "-map", "[a]",
                          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                          "-pix_fmt", "yuv420p"]
                elif afilter:
                    # pitch only - leave the video stream untouched, way faster
                    c += ["-af", afilter, "-c:v", "copy"]
                else:
                    c += ["-c", "copy"]
                # we touched the audio so it has to be re-encoded; aac is safe
                if afilter:
                    c += ["-c:a", "aac", "-b:a", "320k"]
                c += ["-progress", "pipe:1", "-nostats", out]
                return c

            # dump everything to a log so failures aren't a mystery
            log_path = self._export_log_path()

            attempts = [chain(True)]
            # only worth a second try if pitch is in play (that's the rubberband risk)
            if semis != 0:
                attempts.append(chain(False))

            last_err = ""
            for idx, afilter in enumerate(attempts):
                if self._pb_export_cancel:
                    break
                cmd = build_cmd(afilter)
                rc, errtail = self._run_export_cmd(cmd, duration, log_path, idx)
                if rc == 0 and not self._pb_export_cancel:
                    self.root.after(0, lambda: self._pb_export_done(out, None))
                    return
                last_err = errtail

            if self._pb_export_cancel:
                try:
                    if os.path.exists(out):
                        os.remove(out)
                except Exception:
                    pass
                self.root.after(0, lambda: self._pb_export_done(out, "cancelled"))
            else:
                msg = ("FFmpeg could not export this file.\n\n"
                       + last_err +
                       f"\n\nA full log was saved to:\n{log_path}")
                self.root.after(0, lambda m=msg: self._pb_export_done(out, m))
        except Exception as e:
            self.root.after(0, lambda: self._pb_export_done(out, str(e)))

    def _run_export_cmd(self, cmd, duration, log_path, attempt_idx):
        # log the exact command first so we can see what we ran
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== Export attempt {attempt_idx+1} ===\n")
                f.write(" ".join(f'"{x}"' if " " in x else x for x in cmd) + "\n\n")
        except Exception:
            pass

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=HIDE_CONSOLE, text=True)
        self._pb_export_proc = proc

        # ffmpeg's real errors go to stderr - drain it on its own thread (so it
        # can't fill the pipe and deadlock) and keep the tail + write to the log
        tail = []
        def _drain_err():
            try:
                for el in proc.stderr:
                    el = el.rstrip()
                    tail.append(el)
                    if len(tail) > 80:
                        tail.pop(0)
                    try:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(el + "\n")
                    except Exception:
                        pass
            except Exception:
                pass
        et = threading.Thread(target=_drain_err, daemon=True)
        et.start()

        # stdout carries the -progress output; parse out_time_ms for a percent
        for line in proc.stdout:
            if self._pb_export_cancel:
                break
            line = line.strip()
            if line.startswith("out_time_ms=") and duration > 0:
                try:
                    ms = int(line.split("=", 1)[1])
                    pct = min(99, int((ms / 1_000_000) / duration * 100))
                    self.root.after(0, lambda p=pct:
                                    self._pb_set_status(f"Rendering... {p}%"))
                except Exception:
                    pass
        proc.wait()
        et.join(timeout=1.0)
        self._pb_export_proc = None
        # if it failed, the useful bit is the last few stderr lines
        return proc.returncode, "\n".join(tail[-10:])

    @staticmethod
    def _export_log_path():
        # drop the log on the Desktop if we can find it, else just use cwd
        try:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            if os.path.isdir(desktop):
                return os.path.join(desktop, "karaoke_export_log.txt")
        except Exception:
            pass
        return os.path.join(os.getcwd(), "karaoke_export_log.txt")

    @staticmethod
    def _probe_duration(path):
        # ask ffprobe how long the file is, so we can show a real percentage
        try:
            ffmpeg = get_ffmpeg_path()
            ffprobe = os.path.join(os.path.dirname(ffmpeg),
                                   "ffprobe.exe" if sys.platform == "win32" else "ffprobe")
            if not os.path.isfile(ffprobe):
                ffprobe = "ffprobe"
            r = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                                "format=duration", "-of", "csv=p=0", path],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=HIDE_CONSOLE, text=True)
            return float(r.stdout.strip())
        except Exception:
            return 0

    def _pb_export_done(self, out, error):
        try:
            self.pb_export_btn.config(state="normal", text="Save adjusted video...",
                                      command=self._pb_export)
        except Exception:
            pass
        if error == "cancelled":
            self._pb_set_status("Export cancelled.")
        elif error:
            self._pb_set_status("Export failed.")
            messagebox.showerror("Export failed",
                                 f"Could not save the adjusted video:\n\n{error[:600]}")
        else:
            self._pb_set_status("Saved adjusted video.")
            messagebox.showinfo("Saved", f"Adjusted video saved to:\n{out}")

    # ---- seek + time ----
    def _pb_update_time(self, pos):
        if self.player is None or pos is None:
            return
        dur = self._pb_duration or 0
        self.pb_time_lbl.config(text=f"{self._fmt_time(pos)} / {self._fmt_time(dur)}")
        if dur > 0 and not self._pb_seeking:
            try:
                self.pb_seek.set((pos / dur) * 1000)
            except Exception:
                pass

    def _pb_on_seek_drag(self, _val):
        # while dragging we just update the time label preview; commit on release
        if self._pb_seeking and self._pb_duration:
            frac = float(self.pb_seek.get()) / 1000.0
            self.pb_time_lbl.config(
                text=f"{self._fmt_time(frac * self._pb_duration)} / {self._fmt_time(self._pb_duration)}")

    def _pb_seek_commit(self, _evt):
        if self.player is not None and self._pb_duration:
            frac = float(self.pb_seek.get()) / 1000.0
            try:
                self.player.command("seek", frac * self._pb_duration, "absolute")
            except Exception:
                pass
        self._pb_seeking = False

    @staticmethod
    def _fmt_time(secs):
        secs = int(secs or 0)
        return f"{secs // 60}:{secs % 60:02d}"

    def _pb_set_status(self, s):
        try:
            self.pb_status.config(text=s)
        except (tk.TclError, RuntimeError):
            pass

    def _pb_set_playbtn(self, txt):
        try:
            self.pb_play_btn.config(text=txt)
        except (tk.TclError, RuntimeError):
            pass

    def _entry(self, parent, var):
        # standard text entry, just styled to match the theme
        e = tk.Entry(parent, textvariable=var,
                     bg=self.ENTRY_BG, fg=self.TEXT, insertbackground=self.TEXT,
                     font=("Segoe UI", 10), bd=0,
                     highlightthickness=1, highlightcolor=self.ACCENT,
                     highlightbackground=self.BORDER)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        self._attach_entry_menu(e)
        return e

    def _attach_entry_menu(self, entry):
        # tk entries don't come with a right-click menu, which trips people up
        # (they expect paste to be there). so build one ourselves.
        menu = tk.Menu(entry, tearoff=0)
        menu.add_command(label="Cut",
                         command=lambda: entry.event_generate("<<Cut>>"))
        menu.add_command(label="Copy",
                         command=lambda: entry.event_generate("<<Copy>>"))
        menu.add_command(label="Paste",
                         command=lambda: entry.event_generate("<<Paste>>"))
        menu.add_separator()
        menu.add_command(label="Select All",
                         command=lambda: entry.select_range(0, tk.END))

        def show(ev):
            try:
                entry.focus_set()
                menu.tk_popup(ev.x_root, ev.y_root)
            finally:
                menu.grab_release()

        # right-click is button-3 on windows/linux, button-2 on some macs
        entry.bind("<Button-3>", show)
        entry.bind("<Button-2>", show)

    def _update_mode(self):
        # show/hide cards based on whether we're in file or url mode
        if self.mode.get() == self.MODE_FILE:
            self.url_card.pack_forget()
            self.file_card.pack(fill=tk.X, pady=(0, 10))
            self.url_output_card.pack_forget()
            self.output_card.pack(fill=tk.X, pady=(0, 0))
            self.keep_orig_check.pack_forget()
        else:
            self.file_card.pack_forget()
            self.url_card.pack(fill=tk.X, pady=(0, 10))
            self.output_card.pack_forget()
            self.url_output_card.pack(fill=tk.X, pady=(0, 0))
            self.keep_orig_check.pack(anchor=tk.W, pady=(6, 0))

    def _pick_input(self):
        all_ext = " ".join(f"*{e}" for e in SUPPORTED_VIDEO + SUPPORTED_AUDIO)
        p = filedialog.askopenfilename(
            title="Select Video or Audio",
            filetypes=[("All Supported", all_ext), ("All Files", "*.*")]
        )
        if p:
            self.input_path.set(p)
            # auto-fill output with same name + _karaoke suffix
            out = Path(p)
            self.output_path.set(str(out.with_stem(out.stem + "_karaoke")))

    def _pick_output(self):
        p = filedialog.asksaveasfilename(
            title="Save As", defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("MKV", "*.mkv"),
                       ("WAV", "*.wav"), ("MP3", "*.mp3"), ("All", "*.*")]
        )
        if p:
            self.output_path.set(p)

    def _start(self):
        if self.mode.get() == self.MODE_URL:
            self._start_url_mode()
        else:
            self._start_file_mode()

    def _start_file_mode(self):
        inp = self.input_path.get().strip()
        out = self.output_path.get().strip()
        if not inp:
            messagebox.showwarning("No Input", "Select an input file first.")
            return
        if not os.path.isfile(inp):
            messagebox.showerror("Not Found", f"File not found:\n{inp}")
            return
        if not out:
            messagebox.showwarning("No Output", "Choose where to save.")
            return
        if os.path.exists(out):
            if not messagebox.askyesno("Overwrite?", f"File exists:\n{out}\n\nOverwrite?"):
                return

        self._launch_processor(input_source=inp, output_path=out, is_url=False)

    def _start_url_mode(self):
        url = self.url_input.get().strip()
        if not url:
            messagebox.showwarning("No URL", "Paste a YouTube URL first.")
            return
        if not is_url(url):
            messagebox.showwarning("Invalid URL",
                "That doesn't look like a valid video URL.\n\n"
                "Supported: YouTube, Vimeo, Dailymotion, Twitch.")
            return

        has_ytdlp, _ = check_ytdlp()
        if not has_ytdlp:
            messagebox.showerror("yt-dlp Not Installed",
                "YouTube support requires yt-dlp.\n\n"
                "This app was built without it. Please reinstall or run:\n"
                "  pip install yt-dlp")
            return

        self._launch_processor(input_source=url, output_path=None, is_url=True)

    def _launch_processor(self, input_source, output_path, is_url):
        self.start_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress["value"] = 0

        # if user closes the window mid-processing, root is gone but the
        # worker thread doesn't know that and tries to call gui methods.
        # safe_after just swallows TclError when that happens
        def safe_after(fn):
            try:
                self.root.after(0, fn)
            except (tk.TclError, RuntimeError):
                pass

        def on_progress(v):
            safe_after(lambda: self._set_progress(v))

        def on_status(s):
            safe_after(lambda: self._set_status(s))

        def on_done():
            safe_after(self._done)

        def on_error(e):
            safe_after(lambda: self._fail(e))

        self.processor = KaraokeProcessor(
            input_source=input_source,
            output_path=output_path,
            is_url=is_url,
            model_name=self.model_var.get(),
            keep_vocals=self.keep_vocals.get(),
            delete_source_after=(not self.keep_original_video.get()),
            on_progress=on_progress,
            on_status=on_status,
            on_done=on_done,
            on_error=on_error,
        )
        self.worker_thread = threading.Thread(target=self.processor.run, daemon=True)
        self.worker_thread.start()

    def _set_progress(self, v):
        try:
            self.progress.configure(value=v)
        except (tk.TclError, RuntimeError):
            pass

    def _set_status(self, s):
        try:
            self.status_lbl.config(text=s)
        except (tk.TclError, RuntimeError):
            pass

    def _cancel(self):
        if self.processor:
            self.processor.cancel()
        self._reset("Cancelled")

    def _done(self):
        self._reset("")
        self.status_lbl.config(text="✅ Done! Karaoke file saved.", style="Ok.TLabel")
        output_path = self.processor.output_path if self.processor else "?"
        self.status_lbl.config(style="Status.TLabel")
        # offer to jump straight into the Playback tab with the result loaded,
        # but only for video output (mpv plays audio too, but the singer wants video)
        if MPV_AVAILABLE and output_path and os.path.isfile(output_path):
            if messagebox.askyesno(
                "Success",
                f"Saved to:\n{output_path}\n\nOpen it in the Playback tab to sing "
                "and adjust key/tempo?"):
                self.pb_file.set(output_path)
                self.notebook.select(self.playback_tab)
                self.root.after(200, lambda: self._pb_load(output_path))
        else:
            messagebox.showinfo("Success", f"Saved to:\n{output_path}")

    def _fail(self, msg):
        # if it's the special bot-detection signal, show the nice dialog instead
        if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__BOT_DETECTION__":
            self._reset("YouTube blocked the download")
            self._show_bot_detection_dialog()
            return

        self._reset("Error occurred")
        messagebox.showerror("Something went wrong", msg)

    def _show_bot_detection_dialog(self):
        # custom dialog instead of a generic error popup. points users to
        # cobalt.tools which is way more reliable for grabbing youtube videos
        # than fighting yt-dlp's bot checks
        import webbrowser

        dialog = tk.Toplevel(self.root)
        dialog.title("YouTube blocked the download")
        dialog.configure(bg=self.BG)
        dialog.geometry("560x440")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # center on the parent window
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 560) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 440) // 2
        dialog.geometry(f"+{x}+{y}")

        container = tk.Frame(dialog, bg=self.BG, padx=24, pady=20)
        container.pack(fill=tk.BOTH, expand=True)

        tk.Label(container, text="⚠️  YouTube Blocked the Download",
                 bg=self.BG, fg=self.ACCENT,
                 font=("Segoe UI", 14, "bold")).pack(anchor=tk.W, pady=(0, 12))

        explanation = (
            "YouTube's bot-detection is blocking direct downloads from your "
            "computer. This is increasingly common and not something this app "
            "can fully fix on its own.\n\n"
            "Easy workaround — use a free web tool to grab the video, then drop "
            "it back into this app:"
        )
        tk.Label(container, text=explanation,
                 bg=self.BG, fg=self.TEXT,
                 font=("Segoe UI", 10),
                 wraplength=510, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 14))

        steps = tk.Frame(container, bg=self.CARD, padx=16, pady=14)
        steps.pack(fill=tk.X, pady=(0, 16))

        step_text = (
            "1.  Click \"Open cobalt.tools\" below\n"
            "2.  Paste your YouTube URL there and download the video\n"
            "3.  Come back here and switch to \"Local File\" mode\n"
            "4.  Select the downloaded video and click MAKE KARAOKE"
        )
        tk.Label(steps, text=step_text,
                 bg=self.CARD, fg=self.TEXT,
                 font=("Segoe UI", 10),
                 justify=tk.LEFT).pack(anchor=tk.W)

        btn_row = tk.Frame(container, bg=self.BG)
        btn_row.pack(fill=tk.X, pady=(4, 0))

        def open_cobalt():
            webbrowser.open("https://cobalt.tools")

        def open_yt_dlp_help():
            webbrowser.open("https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies")

        def close_dialog():
            dialog.grab_release()
            dialog.destroy()

        # using regular tk buttons (not ttk) because ttk styling on Toplevel
        # is inconsistent across platforms and i gave up
        cobalt_btn = tk.Button(btn_row, text="🔗  Open cobalt.tools",
                                command=open_cobalt,
                                bg=self.ACCENT, fg="white",
                                activebackground=self.ACCENT2, activeforeground="white",
                                font=("Segoe UI", 10, "bold"),
                                bd=0, padx=18, pady=10, cursor="hand2")
        cobalt_btn.pack(side=tk.LEFT)

        help_btn = tk.Button(btn_row, text="Cookie Help",
                              command=open_yt_dlp_help,
                              bg=self.CARD, fg=self.TEXT,
                              activebackground=self.BORDER, activeforeground=self.TEXT,
                              font=("Segoe UI", 9),
                              bd=0, padx=12, pady=8, cursor="hand2")
        help_btn.pack(side=tk.LEFT, padx=(8, 0))

        ok_btn = tk.Button(btn_row, text="OK",
                            command=close_dialog,
                            bg=self.CARD, fg=self.TEXT,
                            activebackground=self.BORDER, activeforeground=self.TEXT,
                            font=("Segoe UI", 9),
                            bd=0, padx=20, pady=8, cursor="hand2")
        ok_btn.pack(side=tk.RIGHT)

        dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    def _reset(self, status):
        self.start_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        if status:
            self.status_lbl.config(text=status)

    def _on_close(self):
        self._closing = True
        # stop any conversion that's running
        if self.processor:
            try:
                self.processor.cancel()
            except Exception:
                pass

        # shutting mpv down is fiddly: python-mpv's terminate() deadlocks if
        # property observers are still attached (it waits on the event queue,
        # which we're blocking). so unhook the observers first, then terminate
        # on a throwaway thread we don't wait on. if it still hangs, the
        # os._exit() at the bottom takes care of it.
        player = self.player
        self.player = None
        if player is not None:
            try:
                for prop in ("time-pos", "duration", "pause"):
                    try:
                        player.unobserve_property(prop)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                t = threading.Thread(target=lambda: self._safe_terminate(player),
                                     daemon=True)
                t.start()
                t.join(timeout=1.0)   # wait a sec, but don't get stuck on it
            except Exception:
                pass

        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

        # torch and mpv both leave threads running in the background, so a
        # clean exit isn't enough - this makes sure the process actually dies
        # instead of hanging around as "not responding".
        os._exit(0)

    @staticmethod
    def _safe_terminate(player):
        try:
            player.terminate()
        except Exception:
            pass

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            # if mainloop ever exits naturally, make sure we still die
            os._exit(0)


if __name__ == "__main__":
    KaraokeApp().run()