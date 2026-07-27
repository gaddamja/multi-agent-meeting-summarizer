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

import warnings
import time
import traceback
from datetime import datetime
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
warnings.filterwarnings("ignore", message=".*Could not load libtorchcodec.*")
warnings.filterwarnings("ignore", message=".*built-in audio decoding will fail.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0.*")


def _log(prefix: str, message: str) -> None:
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{prefix}] {message}")

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


def detect_silence_segments(audio_path: str, silence_threshold: str = "-50dB", min_silence_duration: float = 0.5) -> tuple[List[Dict[str, float]], float]:
    """Detect silent segments in audio using ffmpeg.
    
    Args:
        audio_path: Path to audio file
        silence_threshold: Silence threshold in dB (default: -50dB)
        min_silence_duration: Minimum silence duration in seconds
    
    Returns:
        Tuple of (list of non-silent segments, total duration)
    """
    ffmpeg = find_ffmpeg()
    
    # Use silencedetect filter to find silent periods
    cmd = [
        ffmpeg, "-i", audio_path,
        "-af", f"silencedetect=noise={silence_threshold}:d={min_silence_duration}",
        "-f", "null", "-"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr
    
    # Parse silence detection output
    silence_starts = []
    silence_ends = []
    
    for line in output.split('\n'):
        if 'silence_start:' in line:
            try:
                time_str = line.split('silence_start:')[1].strip()
                silence_starts.append(float(time_str))
            except (ValueError, IndexError):
                pass
        elif 'silence_end:' in line:
            try:
                time_str = line.split('silence_end:')[1].split()[0].strip()
                silence_ends.append(float(time_str))
            except (ValueError, IndexError):
                pass
    
    # Get total duration
    duration_cmd = [
        ffmpeg, "-i", audio_path,
        "-f", "null", "-"
    ]
    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
    
    total_duration = 0.0
    for line in duration_result.stderr.split('\n'):
        if 'Duration:' in line:
            try:
                time_str = line.split('Duration:')[1].split(',')[0].strip()
                parts = time_str.split(':')
                total_duration = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                break
            except (ValueError, IndexError):
                pass
    
    # Build non-silent segments
    segments = []
    current_start = 0.0
    
    for silence_start, silence_end in zip(silence_starts, silence_ends):
        if silence_start > current_start:
            segments.append({
                "start": current_start,
                "end": silence_start
            })
        current_start = silence_end
    
    # Add final segment after last silence
    if current_start < total_duration:
        segments.append({
            "start": current_start,
            "end": total_duration
        })
    
    return segments, total_duration


def load_audio_tensor(audio_path: str, trim_silence: bool = False, 
                      silence_threshold: str = "-50dB", 
                      min_silence_duration: float = 0.5) -> Dict[str, object]:
    """Load audio file and convert to tensor.
    
    Args:
        audio_path: Path to audio file
        trim_silence: If True, trim silence to speed up processing
        silence_threshold: Silence threshold in dB (default: -50dB)
        min_silence_duration: Minimum silence duration in seconds (default: 0.5s)
    """
    if sf is None or np is None or torch is None:
        raise RuntimeError("soundfile, numpy, and torch are required for preloading audio")
    
    ffmpeg = find_ffmpeg()
    
    # First, convert to WAV if needed
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_wav = tmp.name
    try:
        subprocess.run([ffmpeg, "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", temp_wav], 
                      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if trim_silence:
            _log("audio", "Detecting and removing silence segments...")
            non_silent_segments, total_duration = detect_silence_segments(temp_wav, silence_threshold, min_silence_duration)
            
            if non_silent_segments:
                _log("audio", f"Found {len(non_silent_segments)} non-silent segments")
                
                # Create filter complex to concatenate non-silent segments
                filter_parts = []
                for i, seg in enumerate(non_silent_segments):
                    filter_parts.append(
                        f"[0:a]atrim=start={seg['start']}:end={seg['end']},asetpts=PTS-STARTPTS[a{i}]"
                    )
                
                # Concatenate all segments
                concat_inputs = "".join([f"[a{i}]" for i in range(len(non_silent_segments))])
                filter_complex = ";".join(filter_parts) + f";{concat_inputs}concat=n={len(non_silent_segments)}:v=0:a=1[out]"
                
                trimmed_wav = temp_wav + ".trimmed.wav"
                subprocess.run([
                    ffmpeg, "-y", "-i", temp_wav,
                    "-filter_complex", filter_complex,
                    "-map", "[out]", "-ar", "16000", "-ac", "1", trimmed_wav
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(trimmed_wav, temp_wav)
                
                # Calculate time saved
                original_duration = sum(seg['end'] - seg['start'] for seg in non_silent_segments)
                _log("audio", f"✓ Trimmed to {original_duration:.1f}s (removed {total_duration - original_duration:.1f}s of silence)")
            else:
                _log("audio", "No non-silent segments detected, using original audio")
        
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
    _log("transcription", f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    _log("transcription", f"Starting transcription: {audio_path}")
    result = model.transcribe(audio_path, verbose=False)
    segments = result.get("segments", [])
    _log("transcription", f"✓ Completed: {len(segments)} segments detected")
    # result contains 'text' and 'segments' with start/end/time
    return result


# Cache for diarization pipeline to avoid re-loading
_diarization_pipeline_cache = {}


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
        _log("warning", f"Could not log in to Hugging Face: {login_err}")

    try:
        # Use cached pipeline if available
        cache_key = hf_token or "no_token"
        if cache_key not in _diarization_pipeline_cache:
            _log("diarization", "Loading pyannote diarization model (first time only)...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,  # Pass token directly to authenticate gated repo
            )
            _diarization_pipeline_cache[cache_key] = pipeline
            _log("diarization", "✓ Model loaded and cached")
        else:
            pipeline = _diarization_pipeline_cache[cache_key]
        
        _log("diarization", "Processing audio (this may take several minutes)...")
        start_time = time.time()
        
        # Load audio without silence trimming for diarization (use original audio)
        audio = load_audio_tensor(audio_path, trim_silence=False)
        diarization = pipeline(audio)
        
        segments = []
        # Handle different pyannote API versions
        if hasattr(diarization, 'itertracks'):
            # Old API: itertracks(yield_label=True)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
        elif hasattr(diarization, 'speakers_diarization'):
            # New API: speakers_diarization property
            for turn, speaker in diarization.speakers_diarization.items():
                segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
        elif hasattr(diarization, 'items'):
            # Alternative new API
            for turn, speaker in diarization.items():
                segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
        else:
            # Fallback: try to access as dict-like object
            for item in diarization:
                if hasattr(item, 'start') and hasattr(item, 'end') and hasattr(item, 'speaker'):
                    segments.append({"start": float(item.start), "end": float(item.end), "speaker": item.speaker})
        
        elapsed = time.time() - start_time
        _log("diarization", f"✓ Completed: {len(segments)} speaker segments in {elapsed:.1f}s")
        return segments
    except Exception as e:
        # Log full exception details for debugging
        msg = str(e)
        _log("error", f"Speaker diarization failed: {msg}")
        _log("error", f"Exception type: {type(e).__name__}")
        _log("error", "Full traceback:")
        for line in traceback.format_exc().split('\n'):
            _log("error", f"  {line}")
        if "403" in msg or "restricted" in msg or "not in the authorized list" in msg:
            _log("info", "To enable speaker diarization:")
            _log("info", "  1. Visit https://hf.co/pyannote/speaker-diarization-3.1 and accept the user conditions")
            _log("info", "  2. Generate a token at https://huggingface.co/settings/tokens")
            _log("info", "  3. Provide the token in the app's Advanced Settings or via HF_TOKEN env var")
        _log("info", "Continuing without diarization — transcript only will be returned.")
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


def process_audio(audio_path: str, output_json: Optional[str] = None, model_name: str = "small", hf_token: Optional[str] = None, skip_diarization: bool = False) -> Dict:
    """Run transcription and diarization and return structured result."""
    trans_result = transcribe_whisper(audio_path, model_name=model_name)
    trans_segments = trans_result.get("segments", [])
    
    if skip_diarization:
        _log("transcription", "Skipping diarization (skip_diarization=True)")
        # Use transcript segments without speaker labels
        labelled = [
            {"start": seg.get("start", 0.0), "end": seg.get("end", 0.0), "speaker": "unknown", "text": seg.get("text", "").strip()}
            for seg in trans_segments
        ]
    else:
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
    _log("main", "Processing... this may take a while")
    res = process_audio(args.audio, output_json=args.output, model_name=args.model)
    _log("main", f"Wrote {args.output}")
