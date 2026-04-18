"""
pipeline/caption_rewriter.py
Stage 4: Personalized Caption Generation

Uses a seq2seq model (T5 / BART) conditioned on the UserStyleVector
to rewrite generic ASR captions into creator-style personalized captions.

Two modes:
  1. API mode  — calls Anthropic Claude API (for demo / hackathon)
  2. Local mode — uses a fine-tuned T5/BART model (production)
"""

import os
import numpy as np
from typing import Optional


def rewrite_caption_api(
    raw_caption: str,
    style_corpus: str,
    style_keywords: Optional[list] = None,
) -> str:
    """
    API-based caption rewriting using Claude.
    Used when ANTHROPIC_API_KEY is set and no local model is available.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    keywords_hint = ""
    if style_keywords:
        keywords_hint = f"\nStyle signals detected: {', '.join(style_keywords)}"

    prompt = f"""You are the UserStyleVector personalization engine from a research paper on creator-adaptive caption generation.

TASK: Rewrite the raw ASR caption as a SHORT, punchy social-media hook (1-2 sentences max) that perfectly matches the creator's voice.

Match EXACTLY:
- Their vocabulary (slang, filler words, recurring phrases)
- Their tone (hype, sarcastic, casual, intense, etc.)
- Their sentence structure (short punchy vs long flowing)
- Their emoji usage pattern
- Their capitalization style{keywords_hint}

RAW ASR CAPTION:
{raw_caption}

CREATOR STYLE CORPUS (examples of how they actually write):
{style_corpus}

Output ONLY the rewritten caption. No preamble, no explanation, no quotes."""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def rewrite_caption_local(
    raw_caption: str,
    style_vector: np.ndarray,
    model_path: str = "models/caption_rewriter",
) -> str:
    """
    Local T5/BART based caption rewriting.
    Uses UserStyleVector as conditioning prefix.

    The model should be fine-tuned on (neutral, user-style) pairs
    with the style vector prepended to the encoder input.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    # Encode style vector as a prefix token sequence
    # (simple approach: serialize top-k dimensions as discrete tokens)
    style_prefix = _style_vector_to_prefix(style_vector)
    input_text = f"rewrite in user style: {style_prefix} | {raw_caption}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def _style_vector_to_prefix(style_vector: np.ndarray, top_k: int = 10) -> str:
    """
    Encode a style vector into a short text prefix for seq2seq conditioning.
    Uses top-k dimension indices as discrete style tokens.
    """
    top_indices = np.argsort(np.abs(style_vector))[-top_k:]
    prefix_tokens = [f"s{i}:{style_vector[i]:.2f}" for i in top_indices]
    return " ".join(prefix_tokens)


def rewrite_caption(
    raw_caption: str,
    style_vector: np.ndarray,
    style_corpus: str,
    model_path: Optional[str] = None,
) -> str:
    """
    Unified caption rewriter.
    Automatically selects API or local mode.

    Priority:
      1. Local fine-tuned model (if model_path exists)
      2. Anthropic API (if ANTHROPIC_API_KEY is set)
      3. Heuristic fallback (for offline demos)
    """
    # Try local model first
    if model_path and os.path.exists(model_path):
        print(f"[CaptionRewriter] Using local model: {model_path}")
        return rewrite_caption_local(raw_caption, style_vector, model_path)

    # Try Anthropic API
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("[CaptionRewriter] Using Anthropic API")
        from pipeline.style_vector import get_style_keywords
        keywords = get_style_keywords(style_corpus)
        return rewrite_caption_api(raw_caption, style_corpus, keywords)

    # Heuristic fallback for offline demo
    print("[CaptionRewriter] Using heuristic fallback")
    return _heuristic_rewrite(raw_caption, style_corpus)


def _heuristic_rewrite(raw_caption: str, style_corpus: str) -> str:
    """
    Lightweight heuristic rewriter for offline/no-API mode.
    Applies style transforms based on corpus patterns.
    Not as good as seq2seq — for demo/testing only.
    """
    import re

    corpus_lower = style_corpus.lower()
    result = raw_caption.strip()

    # Formality reduction
    replacements = {
        "that is": "that's",
        "it is": "it's",
        "i am": "i'm",
        "do not": "don't",
        "cannot": "can't",
        "would not": "wouldn't",
        "very interesting": "actually kinda wild",
        "important": "lowkey important",
        "people should know": "everyone needs to know",
        "i think": "ngl",
    }
    for formal, casual in replacements.items():
        result = re.sub(re.escape(formal), casual, result, flags=re.IGNORECASE)

    # Add energy if corpus is hype-heavy
    hype_words = ["slapped", "insane", "no cap", "fr", "wild", "unhinged"]
    if any(w in corpus_lower for w in hype_words):
        result = result.rstrip(".") + " — no cap 💀"

    # Lowercase if corpus is mostly lowercase
    upper_ratio = sum(1 for c in corpus_lower if c.isupper()) / max(len(corpus_lower), 1)
    if upper_ratio < 0.02:
        result = result[0].lower() + result[1:] if result else result

    return result
