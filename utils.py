"""
utils.py — Shared utilities for the Agentic Vision project.
Handles I/O, answer extraction, metrics, seed setting, and path resolution.
"""

import json
import random
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set random seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    """Load a JSONL file, returning a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: str, records: list):
    """Write a list of dicts as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def resolve_image_path(image_field: str, data_file_path: str) -> str:
    """
    Resolve the image path stored in the JSONL record to an actual file.
    Tries several candidate locations so the code works regardless of
    whether a ``data/`` directory exists.
    """
    data_dir = Path(data_file_path).parent
    basename = Path(image_field).name
    stripped = image_field.replace("data/", "", 1)

    candidates = [
        Path(image_field),
        data_dir / image_field,
        Path(stripped),
        data_dir / stripped,
        Path("images") / basename,
        data_dir / "images" / basename,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError(f"Cannot find image: {image_field}")


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_COLORS = ["red", "green", "blue", "purple"]


def extract_answer(text: str) -> str:
    """
    Extract a clean answer (integer or colour word) from free-form model
    output.  Handles patterns like "The answer is 5", "5 blue squares", etc.
    """
    if not text:
        return ""
    text = text.strip()

    # Already a clean integer
    if re.match(r"^\d+$", text):
        return text

    # Already a single colour word
    if text.lower() in _COLORS:
        return text.lower()

    # "answer is X" / "result = X"
    m = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*[\"']?(\w+)[\"']?", text, re.I)
    if m:
        val = m.group(1)
        return val.lower() if val.lower() in _COLORS else val

    # "there are N"
    m = re.search(r"there\s+(?:are|is)\s+(\d+)", text, re.I)
    if m:
        return m.group(1)

    # first standalone integer
    numbers = re.findall(r"\b(\d+)\b", text)
    if numbers:
        return numbers[0]

    # colour word anywhere
    for color in _COLORS:
        if color in text.lower():
            return color

    return text.strip()


# ---------------------------------------------------------------------------
# Code extraction from VLM output
# ---------------------------------------------------------------------------

def extract_code(response: str):
    """
    Try to pull a Python code block out of a VLM response.
    Returns the code string or *None* if nothing usable was found.
    """
    if not response:
        return None

    # Fenced code blocks
    for pat in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```", r"```(.*?)```"]:
        m = re.search(pat, response, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if code:
                return code

    # Heuristic: if ≥50 % of lines look like Python, treat the whole thing as code
    lines = response.strip().splitlines()
    indicators = ["import ", "result[", "np.", "cv2.", "image", "=", "for ", "if ", "def ", "print("]
    code_lines = sum(1 for l in lines if any(tok in l for tok in indicators))
    if len(lines) >= 2 and code_lines >= len(lines) * 0.4:
        return response.strip()

    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predictions: list, ground_truth: list) -> dict:
    """
    Compute accuracy, failure-rate, and per-type breakdown.
    ``predictions`` is a list of dicts with at least {'id', 'prediction'}.
    ``ground_truth`` is a list of dicts with at least {'id', 'answer'}.
    """
    gt_map = {r["id"]: r for r in ground_truth}
    correct = 0
    total = 0
    failures = 0
    per_type = {}

    for pred in predictions:
        tid = pred["id"]
        gt = gt_map.get(tid)
        if gt is None:
            continue
        total += 1
        pred_val = str(pred.get("prediction", "")).strip().lower()
        gt_val = str(gt.get("answer", "")).strip().lower()
        task_type = gt.get("task_type", "unknown")

        if task_type not in per_type:
            per_type[task_type] = {"correct": 0, "total": 0}
        per_type[task_type]["total"] += 1

        if not pred_val or pred_val in ("error", "none", ""):
            failures += 1
            continue

        match = pred_val == gt_val
        if not match:
            try:
                if abs(float(pred_val) - float(gt_val)) <= 2:
                    match = True
            except ValueError:
                pass
        if match:
            correct += 1
            per_type[task_type]["correct"] += 1

    accuracy = correct / total if total else 0
    failure_rate = failures / total if total else 0
    for v in per_type.values():
        v["accuracy"] = v["correct"] / v["total"] if v["total"] else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "failure_rate": failure_rate,
        "failures": failures,
        "per_type": per_type,
    }
