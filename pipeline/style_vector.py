"""
pipeline/style_vector.py
Stage 2: UserStyleVector Construction
Core research contribution from the paper:
  "Capturing User Linguistic Identity for Incremental Language Model Personalization"

Builds a mean-pooled sentence embedding vector from the creator's text corpus.
Supports incremental updates (active learning loop).
"""

import numpy as np
import re
from typing import List, Optional
from sentence_transformers import SentenceTransformer

# Multilingual MiniLM outperforms others on code-mixed corpora (paper §4.4)
DEFAULT_ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"

_encoder_cache: dict = {}


def load_encoder(model_name: str = DEFAULT_ENCODER) -> SentenceTransformer:
    """Load and cache sentence encoder."""
    if model_name not in _encoder_cache:
        print(f"[StyleVec] Loading encoder: {model_name}")
        _encoder_cache[model_name] = SentenceTransformer(model_name)
    return _encoder_cache[model_name]


def parse_corpus(corpus_text: str) -> List[str]:
    """
    Split corpus text into individual messages/sentences.
    Handles newline-separated and comma-separated formats.
    Filters empty lines and very short tokens.
    """
    lines = [l.strip() for l in corpus_text.strip().split("\n")]
    messages = [l for l in lines if len(l) > 5]
    return messages


def build_user_style_vector(
    corpus_text: str,
    encoder_name: str = DEFAULT_ENCODER,
    existing_vector: Optional[np.ndarray] = None,
    existing_count: int = 0,
) -> np.ndarray:
    """
    Construct (or incrementally update) the UserStyleVector.

    Algorithm (from paper §3.2):
        v = (1/n) Σ E(mᵢ)   for all messages mᵢ in the corpus

    For incremental updates (active learning loop):
        v_new = (existing_count * v_old + n_new * v_batch) / (existing_count + n_new)

    Args:
        corpus_text:      Raw creator text (newline-separated)
        encoder_name:     Which SBERT encoder to use
        existing_vector:  Previous UserStyleVector (for incremental updates)
        existing_count:   How many messages went into existing_vector

    Returns:
        UserStyleVector as numpy array of shape (embedding_dim,)
    """
    messages = parse_corpus(corpus_text)
    if not messages:
        raise ValueError("Style corpus is empty — provide at least 5 creator messages.")

    encoder = load_encoder(encoder_name)
    embeddings = encoder.encode(messages, convert_to_numpy=True, show_progress_bar=False)

    # Mean pooling (paper: optimal strategy vs max/attention pooling, §4.1)
    new_vector = embeddings.mean(axis=0)

    if existing_vector is not None and existing_count > 0:
        # Incremental update — no full retraining needed
        total = existing_count + len(messages)
        updated = (existing_count * existing_vector + len(messages) * new_vector) / total
        print(f"[StyleVec] Incremental update: {existing_count} → {total} messages")
        return updated.astype(np.float32)

    print(f"[StyleVec] Built UserStyleVector from {len(messages)} messages (dim={new_vector.shape[0]})")
    return new_vector.astype(np.float32)


def compute_style_similarity(
    generated_text: str,
    style_vector: np.ndarray,
    encoder_name: str = DEFAULT_ENCODER,
) -> float:
    """
    Compute cosine similarity between generated text and UserStyleVector.
    Used as style alignment metric (paper §3.4).
    """
    encoder = load_encoder(encoder_name)
    gen_embedding = encoder.encode([generated_text], convert_to_numpy=True)[0]

    # Cosine similarity
    dot = np.dot(gen_embedding, style_vector)
    norm = np.linalg.norm(gen_embedding) * np.linalg.norm(style_vector)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def get_style_keywords(corpus_text: str, top_n: int = 8) -> List[str]:
    """
    Extract characteristic style signals from the corpus.
    Heuristic approach for demo/UI display.
    """
    messages = parse_corpus(corpus_text)
    full_text = " ".join(messages).lower()

    signals = []

    # Emoji presence
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001FFFF\U00002700-\U000027BF\U0001F900-\U0001F9FF]",
        flags=re.UNICODE
    )
    if emoji_pattern.search(full_text):
        signals.append("emoji-heavy")

    # Lowercase dominance
    upper_count = sum(1 for c in full_text if c.isupper())
    if upper_count / max(len(full_text), 1) < 0.03:
        signals.append("lowercase")

    # Slang markers
    slang_words = ["no cap", "fr fr", "lowkey", "highkey", "slapped", "unhinged",
                   "rent free", "istg", "ngl", "tbh", "lmao", "lol", "omg", "bro",
                   "slay", "based", "pov", "idk", "rn", "imo"]
    found_slang = [w for w in slang_words if w in full_text]
    if found_slang:
        signals.append("slang-rich")
        signals.extend(found_slang[:2])

    # Code-mixing detection (non-ASCII chars)
    non_ascii = sum(1 for c in full_text if ord(c) > 127 and not emoji_pattern.match(c))
    if non_ascii > 10:
        signals.append("code-mixed")

    # Ellipsis / trailing dots
    if "..." in full_text or "…" in full_text:
        signals.append("trailing-dots")

    # All-caps words
    caps_words = re.findall(r'\b[A-Z]{2,}\b', " ".join(messages))
    if len(caps_words) > 2:
        signals.append("caps-emphasis")

    # Short punchy sentences
    avg_len = np.mean([len(m.split()) for m in messages]) if messages else 0
    if avg_len < 10:
        signals.append("punchy")
    elif avg_len > 20:
        signals.append("verbose")

    # Question-heavy
    q_count = sum(1 for m in messages if "?" in m)
    if q_count > len(messages) * 0.3:
        signals.append("question-driven")

    return list(dict.fromkeys(signals))[:top_n]  # deduplicate, cap at top_n
