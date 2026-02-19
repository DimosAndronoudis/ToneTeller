# ToneTeller

An emotion-aware AI narrator built on [GLM-TTS](https://github.com/THUDM/GLM-TTS).
ToneTeller analyzes every sentence in your script for emotional context and narrates them
using your cloned voice with perfectly matched emotional delivery.

## How it works

```
Your script
    ↓
Split into sentences
    ↓
Emotion classifier (j-hartmann/emotion-english-distilroberta-base)
    ↓  anger / joy / sadness / fear / surprise / neutral
Pick matching voice profile (your recording in that emotion)
    ↓
GLM-TTS synthesizes each sentence with that emotional prompt
    ↓
Concatenated into one final narration WAV
```

The emotion is controlled via **prompt audio** — because GLM-TTS is a zero-shot
voice cloner, the emotional quality of the reference recording carries through
to the synthesized output.

## Folder layout

```
delete-this-cloning/
├── GLM-TTS/
├── my_voice_studio/
└── narrator_studio/
    ├── config.py               ← path config + emotion definitions
    ├── register_emotions.py    ← Step 1: record yourself in each emotion
    ├── narrate.py              ← Step 2: CLI narration
    ├── emotion_engine.py       ← shared sentence splitting + classification
    ├── frontend/
    │   └── app.py              ← Gradio web UI
    ├── lib/
    │   └── glmtts_inference.py ← copied from GLM-TTS
    ├── emotion_profiles/       ← saved emotion profiles (one .pt per emotion)
    └── outputs/                ← generated narration WAVs
```

## Setup

Install GLM-TTS dependencies first (from inside `GLM-TTS/`).
The emotion classifier (`j-hartmann/emotion-english-distilroberta-base`) downloads
automatically (~250MB) on first use.

## Usage

### Step 1 — Register emotion profiles (run once per emotion)

Record yourself speaking with a clear emotional tone for 5-10 seconds each:

```bash
python register_emotions.py --audio neutral.wav  --emotion neutral  --text "What you said"
python register_emotions.py --audio happy.wav    --emotion happy    --text "What you said"
python register_emotions.py --audio sad.wav      --emotion sad      --text "What you said"
python register_emotions.py --audio angry.wav    --emotion angry    --text "What you said"
python register_emotions.py --audio fearful.wav  --emotion fearful  --text "What you said"
python register_emotions.py --audio surprised.wav --emotion surprised --text "What you said"
```

You only need `neutral` to get started. Other emotions fall back to neutral if not registered.

### Step 2a — CLI narration

```bash
python narrate.py --file my_script.txt
python narrate.py --text "She smiled. Then she cried. Then she screamed."
```

### Step 2b — Web UI

```bash
python frontend/app.py
```

Opens a public Gradio URL (share=True). Features:
- Paste your script
- Click **Analyze emotions** to preview the per-sentence breakdown before generating
- Click **Narrate** to synthesize the full narration

## Supported emotions

| Classifier label | Profile used |
|---|---|
| joy | happy |
| anger, disgust | angry |
| sadness | sad |
| fear | fearful |
| surprise | surprised |
| neutral | neutral |

## Notes

- Commit to the emotion when recording — exaggerate it slightly for best results
- Use the same microphone for all emotion clips
- Model weights live in `GLM-TTS/ckpt/` — never copied here
- Port 7861 is used (7860 is reserved for my_voice_studio)
