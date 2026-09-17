import yt_dlp
from pydub import AudioSegment
import os
from core.deno_setup import ensure_deno


DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


class YouTubeBlockedError(Exception):
    """Raised when YouTube refuses the download (commonly a datacenter-IP block)."""
    pass


def download_youtube_audio(url: str) -> str:

    deno_path = ensure_deno()

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",

        "outtmpl": output_path,

        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],

        "js_runtimes": {
            "deno": {
                "path": deno_path
            }
        },

        "quiet": False,
        "verbose": True,
    }

    # Optional cookie-based auth. Set YTDLP_COOKIES_FILE to the path of a
    # cookies.txt exported from a real logged-in browser session (e.g. via
    # the "Get cookies.txt LOCALLY" extension). On Streamlit Cloud, upload
    # this file as part of the repo (private) or via a secret file mount,
    # then set the env var to that path in your app secrets.
    cookies_file = os.getenv("YTDLP_COOKIES_FILE")
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info_dict = ydl.extract_info(
                url,
                download=True
            )

            filename = (
                ydl.prepare_filename(info_dict)
                .replace(".webm", ".wav")
                .replace(".m4a", ".wav")
                .replace(".mp4", ".wav")
            )

            return filename

    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        if "403" in msg or "Forbidden" in msg:
            raise YouTubeBlockedError(
                "YouTube refused this download. This usually happens when the "
                "app is running on a cloud server — YouTube blocks download "
                "requests from datacenter IPs. Try again from a local run, "
                "provide a cookies file (YTDLP_COOKIES_FILE), or upload the "
                "audio/video file directly instead of a YouTube link."
            ) from e
        raise


def convert_to_wav(input_file: str) -> str:
    output_file = os.path.splitext(input_file)[0] + '_converted.wav'
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Convert to mono and 16kHz set frame rate
    audio.export(output_file, format='wav')
    return output_file


def chunk_audio(wav_path: str, chunk_min: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_min * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)
    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected Youtube URL.Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected Local file.Converting to WAV...")
        wav_path = convert_to_wav(source)

    print(".......Chunking audio.......")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready ---{len(chunks)}chunks created.")
    return chunks