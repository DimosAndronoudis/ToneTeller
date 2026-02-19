"""
narrate.py — CLI emotion-aware narrator
========================================
Reads a text file (or inline text), detects the emotion of each sentence,
picks the matching voice profile, synthesizes it, and concatenates everything
into a single narration WAV.

Usage:
    python narrate.py --file script.txt
    python narrate.py --text "She smiled. Then she cried. Then she screamed."
    python narrate.py --file script.txt --seed 42

Requires at least one emotion profile registered via register_emotions.py.
If a sentence's emotion has no profile, it falls back to 'neutral'.
"""

import os
import sys

# --- Bootstrap ---
STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STUDIO_DIR)

from config import GLMTTS_ROOT, EMOTIONS, EMOTION_LABEL_MAP, FALLBACK_EMOTION
sys.path.insert(0, GLMTTS_ROOT)
os.chdir(GLMTTS_ROOT)
# -----------------

import argparse
import datetime
import torch
import torchaudio

from lib.glmtts_inference import load_models, generate_long
from emotion_engine import analyze_text, load_profile, list_registered

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROFILES_DIR = os.path.join(STUDIO_DIR, "emotion_profiles")
OUTPUTS_DIR  = os.path.join(STUDIO_DIR, "outputs")


def _build_cache(profile: dict, device):
    """Turn a loaded profile dict into the cache structure generate_long expects."""
    embedding           = profile["embedding"].to(device)
    prompt_text_token   = profile["prompt_text_token"].to(device)
    speech_feat         = profile["speech_feat"].to(device)
    prompt_speech_token = profile["prompt_speech_token"].to(device)

    cache_speech_list = [prompt_speech_token.squeeze().tolist()]
    flow_prompt_token = torch.tensor(cache_speech_list, dtype=torch.int32).to(device)

    cache = {
        "cache_text":         [profile["prompt_text"]],
        "cache_text_token":   [prompt_text_token],
        "cache_speech_token": cache_speech_list,
        "use_cache":          True,
    }
    return embedding, speech_feat, flow_prompt_token, cache


def narrate(text: str, seed: int = 42) -> str:
    """Synthesize text with emotion-aware delivery. Returns path to output WAV."""

    registered, missing = list_registered(PROFILES_DIR, EMOTIONS)
    if not registered:
        print("[ERROR] No emotion profiles found.")
        print(f"Run register_emotions.py for at least one emotion. Suggested: --emotion neutral")
        sys.exit(1)

    print(f"Registered emotions : {registered}")
    if missing:
        print(f"Missing (will fall back to '{FALLBACK_EMOTION}'): {missing}")

    # Sentence-level emotion analysis
    print("\nAnalyzing text emotions...")
    sentence_emotions = analyze_text(text, EMOTION_LABEL_MAP, FALLBACK_EMOTION)
    if not sentence_emotions:
        raise ValueError("Input text did not produce any sentences to narrate.")

    print("\n--- Narration plan ---")
    for sentence, emotion in sentence_emotions:
        preview = sentence[:60] + "..." if len(sentence) > 60 else sentence
        print(f"  [{emotion:>9}]  {preview}")

    # Load TTS models once
    sample_rate = None
    for emotion, _ in [(e, None) for _, e in sentence_emotions]:
        p, _ = load_profile(PROFILES_DIR, emotion, FALLBACK_EMOTION, "cpu")
        if p:
            sample_rate = p["sample_rate"]
            break
    if sample_rate is None:
        sample_rate = 24000

    print(f"\nLoading TTS models (sample_rate={sample_rate})...")
    frontend, text_frontend, _, llm, flow = load_models(sample_rate=sample_rate)

    # Profile cache to avoid reloading the same profile multiple times
    _profile_cache = {}

    def get_profile(emotion_name: str):
        if emotion_name not in _profile_cache:
            profile, used = load_profile(PROFILES_DIR, emotion_name, FALLBACK_EMOTION, DEVICE)
            if profile is None:
                raise RuntimeError(
                    f"No profile found for '{emotion_name}' and fallback '{FALLBACK_EMOTION}' "
                    f"is also missing. Register at least a neutral profile."
                )
            if used != emotion_name:
                print(f"  [WARN] No profile for '{emotion_name}', using '{used}' instead.")
            _profile_cache[emotion_name] = profile
        return _profile_cache[emotion_name]

    # Synthesize each sentence
    print("\nSynthesizing...")
    segments = []
    for i, (sentence, emotion) in enumerate(sentence_emotions):
        print(f"  [{i+1}/{len(sentence_emotions)}] [{emotion}] {sentence[:50]}...")
        profile  = get_profile(emotion)
        embedding, speech_feat, flow_prompt_token, cache = _build_cache(profile, DEVICE)

        norm_text = text_frontend.text_normalize(sentence)

        tts_speech, _, _, _ = generate_long(
            frontend=frontend,
            text_frontend=text_frontend,
            llm=llm,
            flow=flow,
            text_info=[f"seg_{i}", norm_text],
            cache=cache,
            embedding=embedding,
            flow_prompt_token=flow_prompt_token,
            speech_feat=speech_feat,
            seed=seed,
            device=DEVICE,
        )
        segments.append(tts_speech)

    # Concatenate all segments
    final_wav = torch.cat(segments, dim=1)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUTS_DIR, f"narration_{ts}.wav")
    torchaudio.save(out_path, final_wav, sample_rate)

    print(f"\nNarration saved to: {out_path}")
    print(f"  Duration : {final_wav.shape[1] / sample_rate:.1f}s")
    print(f"  Segments : {len(segments)}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion-aware text narrator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a .txt script file")
    group.add_argument("--text", help="Inline text to narrate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    narrate(text, seed=args.seed)
