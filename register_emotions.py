"""
register_emotions.py — Register emotion voice profiles
=======================================================
Record yourself speaking in different emotional tones and register each one.
The narrator uses these profiles to match the emotional delivery of each sentence.

Run once per emotion you want to support:
    python register_emotions.py --audio angry.wav   --emotion angry   --text "What you said"
    python register_emotions.py --audio happy.wav   --emotion happy   --text "What you said"
    python register_emotions.py --audio sad.wav     --emotion sad     --text "What you said"
    python register_emotions.py --audio neutral.wav --emotion neutral --text "What you said"

Supported emotions: neutral, happy, sad, angry, fearful, surprised

Recording tips:
  - 5-10 seconds per clip is enough
  - Really commit to the emotion — exaggerate it slightly for best results
  - Keep the same voice/microphone across all clips
  - WAV format preferred

Run list_profiles.py (or just ls emotion_profiles/) to see what's registered.
"""

import os
import sys

# --- Bootstrap ---
STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STUDIO_DIR)

from config import GLMTTS_ROOT, EMOTIONS
sys.path.insert(0, GLMTTS_ROOT)
os.chdir(GLMTTS_ROOT)
# -----------------

import argparse
import torch
from functools import partial

from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
from utils import yaml_util
from utils.audio import mel_spectrogram
from transformers import AutoTokenizer

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROFILES_DIR = os.path.join(STUDIO_DIR, "emotion_profiles")


def build_frontend(sample_rate: int):
    """Load frontend only — no LLM or flow needed for registration."""
    if sample_rate == 32000:
        feat_extractor = partial(
            mel_spectrogram,
            sampling_rate=sample_rate, hop_size=640, n_fft=2560,
            num_mels=80, win_size=2560, fmin=0, fmax=8000, center=False,
        )
    else:
        feat_extractor = partial(
            mel_spectrogram,
            sampling_rate=sample_rate, hop_size=480, n_fft=1920,
            num_mels=80, win_size=1920, fmin=0, fmax=8000, center=False,
        )

    glm_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join("ckpt", "vq32k-phoneme-tokenizer"), trust_remote_code=True
    )
    tokenize_fn = lambda text: glm_tokenizer.encode(text)

    tok_model, tok_extractor = yaml_util.load_speech_tokenizer(
        os.path.join("ckpt", "speech_tokenizer")
    )
    speech_tokenizer = SpeechTokenizer(tok_model, tok_extractor)

    frontend = TTSFrontEnd(
        tokenize_fn,
        speech_tokenizer,
        feat_extractor,
        os.path.join("frontend", "campplus.onnx"),
        os.path.join("frontend", "spk2info.pt"),
        DEVICE,
    )
    text_frontend = TextFrontEnd(use_phoneme=False)
    return frontend, text_frontend


def register(audio_path: str, emotion: str, spoken_text: str, sample_rate: int = 24000):
    if emotion not in EMOTIONS:
        print(f"[ERROR] Unknown emotion '{emotion}'. Choose from: {EMOTIONS}")
        sys.exit(1)
    if not os.path.isfile(audio_path):
        print(f"[ERROR] File not found: {audio_path}")
        sys.exit(1)

    os.makedirs(PROFILES_DIR, exist_ok=True)

    print(f"Registering emotion: {emotion.upper()}")
    print("Loading frontend models...")
    frontend, text_frontend = build_frontend(sample_rate)

    print(f"Extracting features from: {audio_path}")
    norm_text = text_frontend.text_normalize(spoken_text) + " "

    prompt_text_token   = frontend._extract_text_token(norm_text)
    prompt_speech_token = frontend._extract_speech_token([audio_path])
    speech_feat         = frontend._extract_speech_feat(audio_path, sample_rate=sample_rate)
    embedding           = frontend._extract_spk_embedding(audio_path)

    profile = {
        "emotion":             emotion,
        "prompt_text":         norm_text,
        "prompt_text_token":   prompt_text_token.cpu(),
        "prompt_speech_token": prompt_speech_token.cpu(),
        "speech_feat":         speech_feat.cpu(),
        "embedding":           embedding.cpu(),
        "sample_rate":         sample_rate,
        "source_audio":        os.path.abspath(audio_path),
    }

    save_path = os.path.join(PROFILES_DIR, f"{emotion}.pt")
    torch.save(profile, save_path)

    print(f"\n--- [{emotion.upper()}] profile saved ---")
    print(f"  Path         : {save_path}")
    print(f"  Embedding    : {embedding.shape}")
    print(f"  Speech tokens: {prompt_speech_token.shape[1]}")
    print(f"  Prompt text  : '{norm_text.strip()}'")

    # Show what's registered now
    registered = [e for e in EMOTIONS if os.path.isfile(os.path.join(PROFILES_DIR, f"{e}.pt"))]
    missing    = [e for e in EMOTIONS if e not in registered]
    print(f"\n  Registered : {registered}")
    if missing:
        print(f"  Still needed: {missing}")
    else:
        print("  All emotions registered! Ready to narrate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register an emotion voice profile")
    parser.add_argument("--audio",       required=True, help="Path to your emotional voice recording (WAV)")
    parser.add_argument("--emotion",     required=True, choices=EMOTIONS, help="Emotion expressed in the audio")
    parser.add_argument("--text",        required=True, help="Exact words spoken in the audio")
    parser.add_argument("--sample_rate", type=int, default=24000, choices=[24000, 32000])
    args = parser.parse_args()

    register(args.audio, args.emotion, args.text, args.sample_rate)
