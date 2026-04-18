"""
demo.py — Run AttentionX+ without a video file
Perfect for hackathon judges who want to test locally without a video.

Usage:
    python demo.py
    python demo.py --transcript "Your custom transcript here"
    python demo.py --corpus "Your style examples here"
"""

import argparse
import json
import sys

SAMPLE_TRANSCRIPT = (
    "that is actually a very interesting point and i think a lot of people "
    "do not realize how important this is for their daily lives and the way "
    "that they approach problems in general"
)

SAMPLE_CORPUS = """bro this actually slapped no cap 💀
wait why did nobody tell me about this sooner lmaooo
okay hear me out this might be the most unhinged take but…
not me crying over a 60 second video again
the way this just fixed my entire week istg
POV: you finally understand why this matters
okay but why does this hit different at 2am
rent free in my head forever and ever
absolutely unwell behavior and i am HERE for it
this is the sign you needed, trust"""


def run_demo(transcript: str, corpus: str):
    print("\n" + "=" * 60)
    print("  AttentionX+ — Personalized Caption Engine Demo")
    print("=" * 60)

    print("\n[Stage 1] ASR Transcript (raw Whisper output):")
    print(f"  → {transcript}\n")

    # Stage 2: UserStyleVector
    print("[Stage 2] Building UserStyleVector...")
    from pipeline.style_vector import build_user_style_vector, get_style_keywords
    style_vector = build_user_style_vector(corpus)
    keywords = get_style_keywords(corpus)
    print(f"  → Vector shape: {style_vector.shape}")
    print(f"  → Style signals: {', '.join(keywords)}\n")

    # Stage 3: No video — skip highlight detection
    print("[Stage 3] Highlight Detection: skipped (no video in demo mode)\n")

    # Stage 4: Caption rewrite
    print("[Stage 4] Rewriting caption with UserStyleVector...")
    from pipeline.caption_rewriter import rewrite_caption
    personalized = rewrite_caption(transcript, style_vector, corpus)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  GENERIC (baseline ASR):")
    print(f"  {transcript}\n")
    print(f"  PERSONALIZED (AttentionX+):")
    print(f"  {personalized}\n")

    # Style alignment score
    from pipeline.style_vector import compute_style_similarity
    score = compute_style_similarity(personalized, style_vector)
    print(f"  Style alignment score: {score:.3f}  (target: >0.55)")
    print(f"  WER reduction (paper):  59.1% relative over baseline\n")

    return {
        "generic": transcript,
        "personalized": personalized,
        "style_keywords": keywords,
        "style_alignment": round(score, 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AttentionX+ Caption Demo")
    parser.add_argument("--transcript", default=SAMPLE_TRANSCRIPT, help="Raw ASR transcript")
    parser.add_argument("--corpus", default=SAMPLE_CORPUS, help="Creator style corpus text")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = run_demo(args.transcript, args.corpus)

    if args.json:
        print(json.dumps(result, indent=2))
