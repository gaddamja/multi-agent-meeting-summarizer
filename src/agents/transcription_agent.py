"""Transcription Agent: Whisper + pyannote.audio diarization.

Produces timestamped, speaker-labelled transcripts as JSON.
"""
from typing import List, Dict, Optional
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile

# Provide a valid CA bundle for model downloads.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

# Ensure any local FFmpeg shared libraries are visible to torchcodec on macOS.
if os.name == "posix":
    library_paths = [
        "/opt/anaconda3/lib",
        "/opt/homebrew/opt/ffmpeg/lib",
        "/usr/local/opt/ffmpeg/lib",
        "/opt/local/lib",
        "/usr/local/lib",
        "/usr/lib",
    ]

    # Attempt to discover the installed ffmpeg prefix automatically.
    try:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            bin_dir = Path(ffmpeg_bin).resolve().parent
            prefix_dir = bin_dir.parent
            library_paths.extend([
                str(prefix_dir / "lib"),
                str(prefix_dir / "lib64"),
                str(prefix_dir / "opt" / "ffmpeg" / "lib"),
            ])
    except Exception:
        pass

    # Attempt Homebrew-specific detection if brew is available.
    try:
        brew_prefix = subprocess.check_output(["brew", "--prefix", "ffmpeg"], text=True).strip()
        if brew_prefix:
            library_paths.extend([
                str(Path(brew_prefix) / "lib"),
                str(Path(brew_prefix) / "lib64"),
            ])
    except Exception:
        pass

    existing_paths = [p for p in dict.fromkeys(library_paths) if Path(p).is_dir()]
    if existing_paths:
        path_value = ":".join(existing_paths)
        os.environ.setdefault("DYLD_LIBRARY_PATH", path_value)
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", path_value)
        if "LD_LIBRARY_PATH" in os.environ:
            os.environ["LD_LIBRARY_PATH"] = path_value + ":" + os.environ["LD_LIBRARY_PATH"]
        else:
            os.environ["LD_LIBRARY_PATH"] = path_value

# Make ffmpeg discoverable for Whisper and other subprocess-based decoders.
ffmpeg_dirs = ["/opt/anaconda3/bin", "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]
path_dirs = os.environ.get("PATH", "").split(os.pathsep)
for candidate in ffmpeg_dirs:
    if os.path.isdir(candidate) and candidate not in path_dirs:
        path_dirs.insert(0, candidate)
os.environ["PATH"] = os.pathsep.join(path_dirs)

try:
    import numpy as np
    import soundfile as sf
    import torch
    import whisper
    from pyannote.audio import Pipeline
except Exception:
    whisper = None
    np = None
    sf = None
    torch = None
    Pipeline = None


def find_ffmpeg() -> str:
    candidates = ["ffmpeg", "/opt/anaconda3/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError("ffmpeg executable not found. Install ffmpeg or add it to PATH.")


def load_audio_tensor(audio_path: str) -> Dict[str, object]:
    if sf is None or np is None or torch is None:
        raise RuntimeError("soundfile, numpy, and torch are required for preloading audio")
    try:
        data, sample_rate = sf.read(audio_path, always_2d=True)
        waveform = torch.from_numpy(data.T.astype(np.float32))
        return {"waveform": waveform, "sample_rate": int(sample_rate)}
    except Exception:
        ffmpeg = find_ffmpeg()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_wav = tmp.name
        try:
            subprocess.run([ffmpeg, "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", temp_wav], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            data, sample_rate = sf.read(temp_wav, always_2d=True)
            waveform = torch.from_numpy(data.T.astype(np.float32))
            return {"waveform": waveform, "sample_rate": int(sample_rate)}
        finally:
            try:
                os.unlink(temp_wav)
            except OSError:
                pass


def transcribe_whisper(audio_path: str, model_name: str = "small") -> Dict:
    if whisper is None:
        raise RuntimeError("openai-whisper is not installed. Install via pip install openai-whisper")
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, verbose=False)
    # result contains 'text' and 'segments' with start/end/time
    return result


def diarize_pyannote(audio_path: str, hf_token: Optional[str] = None) -> List[Dict]:
    if Pipeline is None:
        raise RuntimeError("pyannote.audio is not installed. Install via pip install pyannote.audio")
    # Hugging Face token may be required for the pretrained pipeline
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        # Also log in via huggingface_hub to ensure token is recognized
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except Exception as login_err:
            print(f"[warning] Could not log in to Hugging Face: {login_err}")

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)
        audio = load_audio_tensor(audio_path)
        diarization = pipeline(audio)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
        return segments
    except Exception as e:
        # Common failure modes: gated HF repo (401/403), network, or missing torchcodec.
        msg = str(e)
        print("[warning] Speaker diarization failed:", msg)
        print("[info] Continuing without diarization — transcript only will be returned.")
        return []


def assign_speakers(trans_segments: List[Dict], diarization_segments: List[Dict]) -> List[Dict]:
    """Assign a speaker label to each transcription segment by maximum overlap."""
    labelled = []
    for seg in trans_segments:
        s_start = seg.get("start", 0.0)
        s_end = seg.get("end", 0.0)
        best = None
        best_overlap = 0.0
        for d in diarization_segments:
            d_start = d["start"]
            d_end = d["end"]
            overlap = max(0.0, min(s_end, d_end) - max(s_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best = d
        speaker = best["speaker"] if best is not None else "unknown"
        labelled.append({"start": s_start, "end": s_end, "speaker": speaker, "text": seg.get("text", "").strip()})
    return labelled


def process_audio(audio_path: str, output_json: Optional[str] = None, model_name: str = "small", hf_token: Optional[str] = None) -> Dict:
    """Run transcription and diarization and return structured result."""
    trans_result = transcribe_whisper(audio_path, model_name=model_name)
    trans_segments = trans_result.get("segments", [])
    diarization_segments = diarize_pyannote(audio_path, hf_token=hf_token)
    labelled = assign_speakers(trans_segments, diarization_segments)
    out = {"file": audio_path, "model": model_name, "transcript": trans_result.get("text", ""), "segments": labelled}
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to audio file (wav/mp3)")
    parser.add_argument("--model", default="small")
    parser.add_argument("--output", default="transcript.json")
    args = parser.parse_args()
    print("Processing... this may take a while")
    res = process_audio(args.audio, output_json=args.output, model_name=args.model)
    print(f"Wrote {args.output}")
