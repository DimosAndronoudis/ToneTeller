"""
frontend/app.py — Narrator Studio Web UI
==========================================
Paste a script, see the per-sentence emotion breakdown, and hear it narrated
with automatic emotional delivery in your cloned voice.

Usage:
    python frontend/app.py

Opens at a public Gradio URL (share=True).

Requires at least one emotion profile registered via register_emotions.py.
"""

import os
import sys
import datetime
import gc
import logging

# --- Bootstrap ---
STUDIO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STUDIO_DIR)

from config import GLMTTS_ROOT, EMOTIONS, EMOTION_LABEL_MAP, FALLBACK_EMOTION
sys.path.insert(0, GLMTTS_ROOT)
os.chdir(GLMTTS_ROOT)
# -----------------

import gradio as gr
import numpy as np
import torch
import torchaudio

from lib.glmtts_inference import load_models, generate_long
from emotion_engine import analyze_text, load_profile, list_registered

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROFILES_DIR = os.path.join(STUDIO_DIR, "emotion_profiles")
OUTPUTS_DIR  = os.path.join(STUDIO_DIR, "outputs")

_MODEL_CACHE = {"loaded": False, "sample_rate": None, "components": None}


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_models(sample_rate: int):
    if _MODEL_CACHE["loaded"] and _MODEL_CACHE["sample_rate"] == sample_rate:
        return _MODEL_CACHE["components"]
    if _MODEL_CACHE["components"]:
        del _MODEL_CACHE["components"]
        gc.collect()
        _clear_cuda_cache()
    logging.info(f"Loading TTS models (sample_rate={sample_rate})...")
    components = load_models(sample_rate=sample_rate)
    _MODEL_CACHE["components"]  = components
    _MODEL_CACHE["sample_rate"] = sample_rate
    _MODEL_CACHE["loaded"]      = True
    logging.info("Models ready.")
    return components


def _build_cache(profile: dict):
    embedding           = profile["embedding"].to(DEVICE)
    prompt_text_token   = profile["prompt_text_token"].to(DEVICE)
    speech_feat         = profile["speech_feat"].to(DEVICE)
    prompt_speech_token = profile["prompt_speech_token"].to(DEVICE)
    cache_speech_list   = [prompt_speech_token.squeeze().tolist()]
    flow_prompt_token   = torch.tensor(cache_speech_list, dtype=torch.int32).to(DEVICE)
    cache = {
        "cache_text":         [profile["prompt_text"]],
        "cache_text_token":   [prompt_text_token],
        "cache_speech_token": cache_speech_list,
        "use_cache":          True,
    }
    return embedding, speech_feat, flow_prompt_token, cache


def analyze(text: str):
    """Run emotion analysis and return a table. No audio generation."""
    if not text.strip():
        return []
    sentence_emotions = analyze_text(text, EMOTION_LABEL_MAP, FALLBACK_EMOTION)
    table = []
    for sentence, emotion in sentence_emotions:
        has_profile = os.path.isfile(os.path.join(PROFILES_DIR, f"{emotion}.pt"))
        profile_status = emotion if has_profile else f"{emotion} → {FALLBACK_EMOTION} (fallback)"
        preview = sentence[:80] + "..." if len(sentence) > 80 else sentence
        table.append([preview, profile_status])
    return table


def narrate(text: str, seed: int):
    """Full narration pipeline."""
    if not text.strip():
        raise gr.Error("Please enter some text to narrate.")

    registered, _ = list_registered(PROFILES_DIR, EMOTIONS)
    if not registered:
        raise gr.Error(
            "No emotion profiles found. "
            "Run register_emotions.py to register at least a 'neutral' profile."
        )

    # Detect emotions per sentence
    sentence_emotions = analyze_text(text, EMOTION_LABEL_MAP, FALLBACK_EMOTION)
    if not sentence_emotions:
        raise gr.Error("Could not extract any sentences from the input text.")

    # Determine sample rate from first available profile
    sample_rate = 24000
    for _, emotion in sentence_emotions:
        p, _ = load_profile(PROFILES_DIR, emotion, FALLBACK_EMOTION, "cpu")
        if p:
            sample_rate = p["sample_rate"]
            break

    frontend, text_frontend, _, llm, flow = _get_models(sample_rate)

    _profile_cache = {}
    def get_profile(emotion_name):
        if emotion_name not in _profile_cache:
            p, used = load_profile(PROFILES_DIR, emotion_name, FALLBACK_EMOTION, DEVICE)
            if p is None:
                raise gr.Error(
                    f"No profile for '{emotion_name}' and fallback '{FALLBACK_EMOTION}' is also missing. "
                    "Register at least a neutral profile."
                )
            _profile_cache[emotion_name] = p
        return _profile_cache[emotion_name]

    # Synthesize each sentence
    segments = []
    for i, (sentence, emotion) in enumerate(sentence_emotions):
        logging.info(f"[{i+1}/{len(sentence_emotions)}] [{emotion}] {sentence[:50]}")
        profile = get_profile(emotion)
        embedding, speech_feat, flow_prompt_token, cache = _build_cache(profile)
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
            seed=int(seed),
            device=DEVICE,
        )
        segments.append(tts_speech)

    final_wav = torch.cat(segments, dim=1)

    # Save to disk
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUTS_DIR, f"narration_{ts}.wav")
    torchaudio.save(out_path, final_wav, sample_rate)

    # Convert for Gradio
    audio_np  = final_wav.squeeze().cpu().numpy()
    audio_np  = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = (audio_np * 32767.0).astype(np.int16)

    return (sample_rate, audio_i16), out_path


def clear_vram():
    global _MODEL_CACHE
    if _MODEL_CACHE["components"]:
        del _MODEL_CACHE["components"]
    _MODEL_CACHE = {"loaded": False, "sample_rate": None, "components": None}
    gc.collect()
    _clear_cuda_cache()
    return gr.update(value="VRAM cleared. Models will reload on next narration.", visible=True)


# --- Profile status ---
_registered, _missing = list_registered(PROFILES_DIR, EMOTIONS)
_status_lines = []
if _registered:
    _status_lines.append(f"Registered: {', '.join(_registered)}")
if _missing:
    _status_lines.append(f"Missing: {', '.join(_missing)} (will use fallback)")
if not _registered:
    _status_lines.append("No profiles yet — run `register_emotions.py` first.")
_status = "  |  ".join(_status_lines)

# --- UI ---
with gr.Blocks(title="My Narrator Studio", theme=gr.themes.Soft()) as app:
    gr.Markdown("# My Narrator Studio")
    gr.Markdown(
        "Paste any script. Each sentence is analyzed for emotion and narrated "
        "in your cloned voice with matching emotional delivery."
    )
    gr.Markdown(f"**Emotion profiles:** {_status}")

    with gr.Row():
        # Left — script input
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Script",
                placeholder=(
                    'Paste your text here. Example:\n\n'
                    '"She opened the letter with trembling hands. '
                    'It was finally good news. She laughed until she cried. '
                    'Then the second envelope fell out."'
                ),
                lines=10,
            )
            with gr.Row():
                analyze_btn = gr.Button("Analyze emotions", variant="secondary")
                narrate_btn = gr.Button("Narrate", variant="primary")

        # Right — settings
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            with gr.Accordion("Advanced", open=False):
                seed = gr.Number(label="Seed", value=42, precision=0)

    # Emotion breakdown table
    emotion_table = gr.Dataframe(
        headers=["Sentence", "Emotion / Profile used"],
        label="Per-sentence emotion breakdown",
        interactive=False,
        wrap=True,
    )

    # Output
    with gr.Row():
        output_audio = gr.Audio(label="Narration")
        saved_path   = gr.Textbox(label="Saved to", interactive=False)

    clear_btn = gr.Button("Clear VRAM", variant="secondary", size="sm")
    clear_msg = gr.Textbox(label="", interactive=False, visible=False)

    # Events
    analyze_btn.click(
        fn=analyze,
        inputs=[text_input],
        outputs=[emotion_table],
    )

    narrate_btn.click(
        fn=lambda text, seed: (analyze(text), *narrate(text, seed)),
        inputs=[text_input, seed],
        outputs=[emotion_table, output_audio, saved_path],
    )

    clear_btn.click(
        fn=clear_vram,
        inputs=None,
        outputs=[clear_msg],
    )


if __name__ == "__main__":
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
    )
