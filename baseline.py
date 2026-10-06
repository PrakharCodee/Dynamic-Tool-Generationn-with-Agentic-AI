"""
baseline.py — Direct VQA baseline (single-pass, no tool execution).

Sends the image and question directly to the VLM and returns the answer.
This serves as the comparison baseline for the agentic approach.
"""

from PIL import Image
from model_wrapper import query_vlm
from utils import extract_answer


def solve_baseline(model, image: Image.Image, question: str) -> str:
    """
    Answer a visual question with a **single** model call.

    No code generation, no sandbox execution — just direct VQA.
    """
    response = query_vlm(model, image, question)
    answer = extract_answer(response)
    return answer
