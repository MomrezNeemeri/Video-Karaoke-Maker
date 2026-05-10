"""
karaoke maker — strips vocals out of videos using demucs.
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

    def _run_demucs(self, wav_path, output_instrumental_path, progress_cb=None):
        progress_cb = progress_cb or self.on_progress

        self.on_status("Loading AI model (first run downloads ~200MB)...")
        progress_cb(12)

        model = get_model(self.model_name)
        model.eval()

        # cuda for nvidia, cpu for everyone else.
        # tried mps (apple silicon gpu) - demucs has conv layers with > 65536
        # output channels and apple's mps just refuses, hard limit. even with
        # the fallback flag it shuffles tensors between cpu and gpu so much
        # that it ends up SLOWER than just using cpu directly. so we don't.
        # M-series cpus are plenty fast anyway, ~1-2 min for a 4 min song
        if torch.cuda.is_available():
            device = "cuda"
            device_name = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            device = "cpu"
            if sys.platform == "darwin":
                device_name = "Apple Silicon CPU"
            else:
                device_name = "CPU (no NVIDIA GPU detected)"

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
        self.root.geometry("740x780")
        self.root.minsize(660, 700)
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

        self._styles()
        self._build()
        self._update_mode()

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

    def _build(self):
        m = ttk.Frame(self.root, style="App.TFrame", padding=30)
        m.pack(fill=tk.BOTH, expand=True)

        ttk.Label(m, text="🎤 Video Karaoke Maker", style="Title.TLabel").pack(pady=(0, 2))
        ttk.Label(m, text="Remove vocals from any video using AI  •  Powered by Meta Demucs",
                  style="Sub.TLabel").pack(pady=(0, 16))

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

    def _entry(self, parent, var):
        # standard text entry, just styled to match the theme
        e = tk.Entry(parent, textvariable=var,
                     bg=self.ENTRY_BG, fg=self.TEXT, insertbackground=self.TEXT,
                     font=("Segoe UI", 10), bd=0,
                     highlightthickness=1, highlightcolor=self.ACCENT,
                     highlightbackground=self.BORDER)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        return e

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
        messagebox.showinfo("Success", f"Saved to:\n{output_path}")
        self.status_lbl.config(style="Status.TLabel")

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
        # cancel any running work
        if self.processor:
            try:
                self.processor.cancel()
            except Exception:
                pass

        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

        # nuclear option. without this torch leaves threads alive and the app
        # just hangs around in task manager forever even after window closes
        os._exit(0)

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
