"""
emotion_engine.py — Sentence splitting and emotion detection
=============================================================
Shared logic used by both narrate.py and the frontend.
"""

import os
import re
import torch
from typing import List, Tuple

# Lazy-loaded classifier (only downloaded/loaded on first call)
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        print("Loading emotion classifier (first run downloads ~250MB)...")
        _classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
            device=-1,  # always CPU — tiny model, fast enough
        )
        print("Emotion classifier ready.")
    return _classifier


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on .  !  ?  or newlines."""
    # Normalize newlines, then split on sentence-ending punctuation
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    # Split on punctuation followed by whitespace/newline, or on blank lines
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def classify_sentence(sentence: str, label_map: dict, fallback: str) -> str:
    """Return the emotion profile name for a single sentence."""
    clf = _get_classifier()
    result = clf(sentence[:512])  # truncate very long sentences
    if isinstance(result, list) and result and isinstance(result[0], dict):
        raw_label = result[0]["label"].lower()
    elif isinstance(result, list) and result and isinstance(result[0], list) and result[0]:
        raw_label = result[0][0]["label"].lower()
    else:
        return fallback
    return label_map.get(raw_label, fallback)


def analyze_text(text: str, label_map: dict, fallback: str) -> List[Tuple[str, str]]:
    """
    Split text into sentences and classify each one.
    Returns a list of (sentence, emotion) tuples.
    """
    sentences = split_sentences(text)
    results = []
    for sentence in sentences:
        emotion = classify_sentence(sentence, label_map, fallback)
        results.append((sentence, emotion))
    return results


def load_profile(profiles_dir: str, emotion: str, fallback: str, device):
    """
    Load an emotion profile from disk.
    Falls back to `fallback` emotion if the requested one isn't registered.
    Returns None if neither is available.
    """
    path = os.path.join(profiles_dir, f"{emotion}.pt")
    if not os.path.isfile(path):
        fallback_path = os.path.join(profiles_dir, f"{fallback}.pt")
        if os.path.isfile(fallback_path):
            return torch.load(fallback_path, map_location=device, weights_only=False), fallback
        return None, None
    return torch.load(path, map_location=device, weights_only=False), emotion


def list_registered(profiles_dir: str, emotions: list) -> Tuple[List[str], List[str]]:
    """Return (registered, missing) lists."""
    registered = [e for e in emotions if os.path.isfile(os.path.join(profiles_dir, f"{e}.pt"))]
    missing    = [e for e in emotions if e not in registered]
    return registered, missing
