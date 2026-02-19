# My Narrator Studio — Project Description

## Overview

An emotion-aware text narrator. Paste any script — every sentence is automatically
analyzed for emotional tone, then synthesized in your cloned voice using a matching
emotional reference recording. The result is a single narration WAV where your voice
naturally shifts between happy, sad, angry, neutral, and other emotional deliveries.

---

## Source Repository

Built on top of **GLM-TTS** by Zhipu AI (CogAudio Group).

| Field | Value |
|---|---|
| Git clone source | https://github.com/zai-org/GLM-TTS.git |
| Paper | arXiv:2512.14291 |
| HuggingFace | https://huggingface.co/zai-org/GLM-TTS |
| License | Apache 2.0 |

The GLM-TTS repository is cloned as a **sibling folder** (`../GLM-TTS`).
It is never modified. This project only reads its model weights and imports
its Python modules at runtime.

---

## Architecture

```
narrator_studio/
├── config.py               — Path resolver + emotion definitions + label map
├── register_emotions.py    — One-time per-emotion voice profile extraction
├── emotion_engine.py       — Sentence splitting + emotion classification (shared)
├── narrate.py              — CLI narration pipeline
├── frontend/
│   └── app.py              — Gradio web UI (public URL via share=True)
├── lib/
│   └── glmtts_inference.py — Verbatim copy of GLM-TTS inference script
├── emotion_profiles/
│   ├── neutral.pt          — Voice profile for neutral delivery
│   ├── happy.pt            — Voice profile for happy delivery
│   ├── sad.pt              — etc.
│   ├── angry.pt
│   ├── fearful.pt
│   └── surprised.pt
└── outputs/
    └── narration_TIMESTAMP.wav
```

### Full pipeline

```
[Script text]
      ↓  emotion_engine.split_sentences()
["She smiled.", "Then she cried.", "Then she screamed!"]
      ↓  emotion_engine.classify_sentence()  ←  j-hartmann/emotion-english-distilroberta-base
[("She smiled.",       "happy"   ),
 ("Then she cried.",   "sad"     ),
 ("Then she screamed!","angry"   )]
      ↓  for each sentence:
      │     load emotion_profiles/{emotion}.pt
      │     ↓
      │   ┌────────────────────────────────────────────┐
      │   │  LLM Stage  (Llama-based GLMTTS)           │
      │   │  text tokens + emotion prompt tokens        │
      │   │  → discrete speech token sequence           │
      │   └────────────────────────────────────────────┘
      │     ↓
      │   ┌────────────────────────────────────────────┐
      │   │  Flow Matching Stage  (DiT)                 │
      │   │  speech tokens + emotion mel + embedding    │
      │   │  → mel-spectrogram                          │
      │   └────────────────────────────────────────────┘
      │     ↓
      │   ┌────────────────────────────────────────────┐
      │   │  Vocoder  (HiFi-GAN 24kHz / Vocos 32kHz)  │
      │   │  → waveform segment                         │
      │   └────────────────────────────────────────────┘
      ↓  torch.cat(all segments, dim=1)
[narration_TIMESTAMP.wav]
```

### How emotion control works

GLM-TTS has no emotion parameter at inference time. Emotion was only used during
GRPO training as a reward signal — it shaped the model's prosody but is not exposed
as a runtime argument.

The emotion control mechanism used here is **prompt audio selection**: GLM-TTS is
a zero-shot voice cloner that inherits both voice identity AND delivery style from
the reference audio. By recording 6 short clips of yourself (one per emotion) and
selecting the matching clip as the prompt for each sentence, the emotional quality
of the reference carries through to the synthesized output.

```
Emotion detected: "angry"
     → load emotion_profiles/angry.pt
     → that profile's prompt audio was you sounding angry
     → GLM-TTS clones your angry delivery onto the new sentence
```

---

## Special Components

### config.py
- Auto-resolves `GLMTTS_ROOT` as `../GLM-TTS` relative to the file
- Defines `EMOTIONS` list: `["neutral", "happy", "sad", "angry", "fearful", "surprised"]`
- Defines `EMOTION_LABEL_MAP`: maps classifier output labels to profile names
  - `joy → happy`, `anger/disgust → angry`, `fear → fearful`, `surprise → surprised`
- Defines `FALLBACK_EMOTION = "neutral"` — used when a sentence's emotion has no registered profile

### emotion_engine.py
Shared module used by both `narrate.py` and `frontend/app.py`:
- `split_sentences(text)` — regex-based split on `.  !  ?` and double newlines
- `_get_classifier()` — lazy-loads the Hugging Face classifier on first call only
- `classify_sentence(sentence, label_map, fallback)` — runs the classifier, maps to profile name
- `analyze_text(text, ...)` — full pipeline, returns `[(sentence, emotion), ...]`
- `load_profile(profiles_dir, emotion, fallback, device)` — loads `.pt`, falls back gracefully
- `list_registered(profiles_dir, emotions)` — returns `(registered, missing)` lists

### Emotion classifier
- Model: `j-hartmann/emotion-english-distilroberta-base`
- ~250MB download on first use, cached locally by Hugging Face
- Always runs on CPU (small model, fast enough, saves GPU for TTS)
- Returns one of: `anger, disgust, fear, joy, neutral, sadness, surprise`
- Input truncated to 512 tokens for very long sentences

### register_emotions.py
- Identical extraction logic to `my_voice_studio/register_voice.py`
- Accepts `--emotion` flag from the defined `EMOTIONS` list
- Saves profile to `emotion_profiles/{emotion}.pt`
- After saving, prints which emotions are registered and which are still missing
- Only loads frontend models (SpeechTokenizer + CAMPPLUS) — no LLM or flow needed

### narrate.py (CLI)
- Runs the full pipeline: analyze → per-sentence synthesis → concatenate
- Profile cache: avoids reloading the same emotion profile multiple times
- Graceful fallback: if a sentence's emotion profile is missing, uses `FALLBACK_EMOTION`
- Prints a narration plan table before synthesis so you can review the emotion assignments
- Output: single concatenated WAV with duration and segment count printed

### frontend/app.py (Gradio UI)
- **Lazy model loading** — TTS models load on first Narrate click
- **Two-step interaction:**
  - *Analyze* button — runs only emotion classification, shows breakdown table instantly (no TTS)
  - *Narrate* button — runs full pipeline, shows table + plays audio
- **Emotion breakdown table** — shows each sentence alongside its detected emotion and which profile will be used (or fallback)
- `share=True` — public HTTPS URL accessible from any device
- Port: `7861` (separated from my_voice_studio on `7860`)

---

## Emotion Profile Format

Each `.pt` file is a Python dict saved with `torch.save`:

```python
{
    "emotion":             str,          # e.g. "happy"
    "prompt_text":         str,          # normalized text spoken in the recording
    "prompt_text_token":   Tensor,       # shape (1, N) int32
    "prompt_speech_token": Tensor,       # shape (1, K) int32 — discrete audio tokens
    "speech_feat":         Tensor,       # shape (1, F, 80) float32 — mel-spectrogram
    "embedding":           Tensor,       # shape (1, 192) float32 — speaker identity
    "sample_rate":         int,          # 24000 or 32000
    "source_audio":        str,          # absolute path to original recording
}
```

---

## What It Uses from GLM-TTS

| Component | GLM-TTS source | Purpose |
|---|---|---|
| `GLMTTS` LLM | `llm/glmtts.py` | Text → speech token generation (Llama backbone) |
| `TTSFrontEnd` | `cosyvoice/cli/frontend.py` | Text tokenization, speech tokenization, feature extraction |
| `SpeechTokenizer` | `cosyvoice/cli/frontend.py` | Audio → discrete token encoding |
| `TextFrontEnd` | `cosyvoice/cli/frontend.py` | Text normalization, sentence splitting |
| `Token2Wav` | `utils/tts_model_util.py` | Speech tokens + mel → waveform (wraps flow + vocoder) |
| CAMPPLUS ONNX | `frontend/campplus.onnx` | 192-dim speaker embedding extraction |
| Flow model | `ckpt/flow/flow.pt` | DiT-based token-to-mel flow matching |
| LLM weights | `ckpt/llm/` | Llama model weights |
| Speech tokenizer | `ckpt/speech_tokenizer/` | Tokenizer model weights |
| GLM tokenizer | `ckpt/vq32k-phoneme-tokenizer/` | Text tokenizer |
| Vocos vocoder | `ckpt/` (32kHz mode) | Mel → waveform |
| HiFi-GAN vocoder | `ckpt/` (24kHz mode) | Mel → waveform |
| `yaml_util` | `utils/yaml_util.py` | Model loading helpers |
| `seed_util` | `utils/seed_util.py` | Reproducible generation |

**External (not from GLM-TTS):**
- `j-hartmann/emotion-english-distilroberta-base` — emotion classifier (Hugging Face Hub)

---

## How to Run

### Prerequisites
1. Clone and set up GLM-TTS (install its dependencies, download model weights into `GLM-TTS/ckpt/`)
2. Ensure `narrator_studio` is a sibling of `GLM-TTS` under the same parent folder

### Step 1 — Register emotion profiles (once per emotion)
```bash
# Minimum to get started — neutral is the universal fallback
python register_emotions.py --audio neutral.wav  --emotion neutral  --text "What you said"

# Add more for richer narration
python register_emotions.py --audio happy.wav    --emotion happy    --text "What you said"
python register_emotions.py --audio sad.wav      --emotion sad      --text "What you said"
python register_emotions.py --audio angry.wav    --emotion angry    --text "What you said"
python register_emotions.py --audio fearful.wav  --emotion fearful  --text "What you said"
python register_emotions.py --audio surprised.wav --emotion surprised --text "What you said"
```

### Step 2a — CLI
```bash
python narrate.py --text "She smiled. Then the letter fell from her hands."
python narrate.py --file my_script.txt
python narrate.py --file script.txt --seed 123
```

### Step 2b — Web UI
```bash
python frontend/app.py
# Prints: Running on public URL: https://xxxx.gradio.live
```

---

## Key Dependencies (from GLM-TTS requirements)

- Python 3.10–3.12
- PyTorch 2.3.1 + TorchAudio 2.3.1
- Transformers 4.57.3 (also used for emotion classifier)
- ONNX Runtime (CAMPPLUS speaker embedding)
- Gradio (web UI)
- Librosa, SoundFile (audio processing)
- WeTextProcessing / Jieba / PyPinyin (Chinese text normalization)
