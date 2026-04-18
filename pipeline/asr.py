"""
pipeline/asr.py
Stage 1: Speech-to-Text using OpenAI Whisper
Produces raw transcript + timestamped segments
"""

import whisper
import torch
from typing import Optional


_model_cache: dict = {}


def load_model(model_size: str = "base") -> whisper.Whisper:
    """Load and cache Whisper model."""
    if model_size not in _model_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ASR] Loading Whisper '{model_size}' on {device}...")
        _model_cache[model_size] = whisper.load_model(model_size, device=device)
    return _model_cache[model_size]


def transcribe_video(
    video_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
) -> dict:
    """
    Transcribe a video file using Whisper.

    Returns:
        {
            "text": full transcript string,
            "segments": [
                {
                    "id": int,
                    "start": float (seconds),
                    "end": float (seconds),
                    "text": str,
                    "avg_logprob": float,
                    "no_speech_prob": float,
                }
            ],
            "language": str
        }
    """
    model = load_model(model_size)

    transcribe_options = {
        "fp16": torch.cuda.is_available(),
        "verbose": False,
    }
    if language:
        transcribe_options["language"] = language

    print(f"[ASR] Transcribing: {video_path}")
    result = model.transcribe(video_path, **transcribe_options)

    # Normalize segments to include only needed fields
    segments = []
    for seg in result["segments"]:
        segments.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "avg_logprob": seg.get("avg_logprob", 0.0),
            "no_speech_prob": seg.get("no_speech_prob", 0.0),
        })

    print(f"[ASR] Done. {len(segments)} segments, language={result.get('language','?')}")

    return {
        "text": result["text"].strip(),
        "segments": segments,
        "language": result.get("language", "en"),
    }
