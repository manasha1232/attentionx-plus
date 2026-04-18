"""
pipeline/highlight_detector.py
Stage 3: Emotional Peak / Highlight Detection

Combines:
  - Audio energy (Librosa RMS) — finds loud/energetic moments
  - Sentiment scoring (TextBlob / VADER) — finds emotionally charged text
  - Segment filtering — removes low-confidence ASR segments

Paper reference: emotional peak detection for "golden nugget" extraction
"""

import numpy as np
import librosa
from typing import List, Dict


def extract_audio_energy(
    video_path: str,
    sr: int = 22050,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> tuple:
    """
    Extract per-frame RMS energy from audio track.
    Returns (times_array, rms_array).
    """
    print(f"[Highlight] Loading audio from: {video_path}")
    y, sr = librosa.load(video_path, sr=sr, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )
    return times, rms


def score_segment_energy(
    segment_start: float,
    segment_end: float,
    times: np.ndarray,
    rms: np.ndarray,
) -> float:
    """Mean RMS energy for a transcript segment's time window."""
    mask = (times >= segment_start) & (times <= segment_end)
    if not mask.any():
        return 0.0
    return float(rms[mask].mean())


def score_segment_sentiment(text: str) -> float:
    """
    Score emotional intensity of transcript text.
    Uses VADER (social media aware) if available, falls back to TextBlob.
    Returns absolute sentiment score in [0, 1].
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        # compound is in [-1, 1]; take absolute value for energy
        return abs(scores["compound"])
    except ImportError:
        pass

    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        return abs(blob.sentiment.polarity)
    except ImportError:
        pass

    # Fallback: simple word-count heuristic
    high_energy_words = {
        "amazing", "incredible", "shocking", "wow", "crazy", "unbelievable",
        "important", "critical", "never", "always", "best", "worst",
        "love", "hate", "fear", "excited", "urgent", "breaking",
        "no cap", "fr", "literally", "insane", "wild",
    }
    words = set(text.lower().split())
    hits = len(words & high_energy_words)
    return min(hits / 5.0, 1.0)


def detect_highlights(
    video_path: str,
    segments: List[Dict],
    n: int = 3,
    min_duration: float = 5.0,
    max_duration: float = 60.0,
    energy_weight: float = 0.5,
    sentiment_weight: float = 0.5,
) -> List[Dict]:
    """
    Detect top-N highlight moments from transcript segments.

    Scoring = energy_weight * audio_energy_score
             + sentiment_weight * sentiment_score

    Args:
        video_path:    Path to video file
        segments:      Whisper transcript segments with start/end/text
        n:             Number of highlights to return
        min_duration:  Minimum clip duration in seconds
        max_duration:  Maximum clip duration in seconds
        energy_weight: Weight for audio RMS energy component
        sentiment_weight: Weight for text sentiment component

    Returns:
        List of top-N segments sorted by score (highest first)
    """
    print(f"[Highlight] Scoring {len(segments)} segments...")

    # Extract audio energy once
    times, rms = extract_audio_energy(video_path)
    # Normalize RMS to [0, 1]
    rms_min, rms_max = rms.min(), rms.max()
    rms_norm = (rms - rms_min) / (rms_max - rms_min + 1e-9)

    scored = []
    for seg in segments:
        # Skip low-confidence or nearly-silent segments
        if seg.get("no_speech_prob", 0) > 0.7:
            continue
        if len(seg["text"].strip()) < 10:
            continue

        duration = seg["end"] - seg["start"]
        if duration < 2.0:
            continue

        energy_score = score_segment_energy(seg["start"], seg["end"], times, rms_norm)
        sentiment_score = score_segment_sentiment(seg["text"])

        combined_score = (
            energy_weight * energy_score + sentiment_weight * sentiment_score
        )

        scored.append({
            **seg,
            "energy_score": energy_score,
            "sentiment_score": sentiment_score,
            "combined_score": combined_score,
        })

    # Sort by combined score
    scored.sort(key=lambda x: x["combined_score"], reverse=True)

    # Merge adjacent segments to reach target duration
    highlights = []
    for top_seg in scored[:n * 2]:  # consider more than needed
        start = top_seg["start"]
        end = top_seg["end"]

        # Expand window if segment is too short
        if (end - start) < min_duration:
            padding = (min_duration - (end - start)) / 2
            start = max(0, start - padding)
            end = end + padding

        # Cap at max_duration
        if (end - start) > max_duration:
            end = start + max_duration

        highlights.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "text": top_seg["text"],
            "energy_score": top_seg["energy_score"],
            "sentiment_score": top_seg["sentiment_score"],
            "combined_score": top_seg["combined_score"],
        })

        if len(highlights) >= n:
            break

    print(f"[Highlight] Selected {len(highlights)} highlights")
    return highlights
