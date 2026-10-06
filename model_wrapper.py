"""
model_wrapper.py — Moondream VLM wrapper.

Loads the model once and exposes a lightweight query interface used by
both the agent and the baseline.
"""

import torch
from transformers import AutoModelForCausalLM
from PIL import Image

# Use the official 4-bit release as suggested in the coursework spec
MODEL_ID = "moondream/moondream-2b-2025-04-14-4bit"


def load_model(device: str = "cuda"):
    """
    Load the Moondream VLM. Standard Moondream 2 is used as a stable
    alternative if CUDA is unavailable or the 4-bit release fails.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU (slow).")
        device = "cpu"

    print(f"Loading {MODEL_ID} on {device} …")
    
    try:
        # Load the 4-bit model with trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map={"": device}
        )
    except Exception as e:
        print(f"Error loading {MODEL_ID}: {e}")
        print("Falling back to standard moondream2...")
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", 
            trust_remote_code=True,
            device_map={"": device}
        )
        
    print("Model loaded.")

    if device == "cuda":
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"VRAM after load: {vram_mb:.0f} MB")

    return model


def query_vlm(model, image: Image.Image, prompt: str) -> str:
    """
    Send *image* + *prompt* to the VLM and return the text answer.
    """
    try:
        resp = model.query(image, prompt)
        return resp.get("answer", "")
    except Exception as exc:
        print(f"VLM query error: {exc}")
        return ""


def measure_vram() -> float:
    """Return current VRAM usage in MB (0 on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0
