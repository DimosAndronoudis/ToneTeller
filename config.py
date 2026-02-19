"""
config.py — Path configuration for narrator_studio
====================================================
By default, GLM-TTS is assumed to be a sibling folder of narrator_studio.

    delete-this-cloning/
    ├── GLM-TTS/           ← cloned repo (model weights + source modules)
    ├── my_voice_studio/
    └── narrator_studio/   ← this project

If your folder layout is different, change GLMTTS_ROOT below.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_ROOTS = [
    os.environ.get("GLMTTS_ROOT"),
    os.path.join(_THIS_DIR, "GLM-TTS"),  # bundled inside narrator_studio
    os.path.join(os.path.dirname(_THIS_DIR), "GLM-TTS"),  # sibling folder
]

GLMTTS_ROOT = next((p for p in _CANDIDATE_ROOTS if p and os.path.isdir(p)), None)

if GLMTTS_ROOT is None:
    checked = [p for p in _CANDIDATE_ROOTS if p]
    raise FileNotFoundError(
        "GLM-TTS directory not found. Checked:\n"
        + "\n".join(f"- {p}" for p in checked)
        + "\nSet GLMTTS_ROOT environment variable or edit config.py."
    )

# Supported emotion names — these are the profile filenames (without .pt)
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]

# Mapping from transformers emotion labels → our profile names
EMOTION_LABEL_MAP = {
    "neutral":  "neutral",
    "joy":      "happy",
    "sadness":  "sad",
    "anger":    "angry",
    "disgust":  "angry",    # fallback — no disgust profile
    "fear":     "fearful",
    "surprise": "surprised",
}

# Fallback profile when a sentence's emotion has no registered sample
FALLBACK_EMOTION = "neutral"
