import sys
import json
from pathlib import Path
from PIL import Image
import torch
from model_wrapper import load_model
from utils import resolve_image_path
from agent import solve as agent_solve

def test_task(task_id, data_path="dev.jsonl"):
    with open(data_path, "r") as f:
        for line in f:
            task = json.loads(line)
            if task["id"] == task_id:
                break
        else:
            print(f"Task {task_id} not found")
            return

    print(f"Testing Task: {task_id}")
    print(f"Question: {task['question']}")
    img_path = resolve_image_path(task["image"], data_path)
    
    img = Image.open(img_path).convert("RGB")
    
    # We can pass None for model if we only want to test the tool part, 
    # but agent.solve calls query_vlm on fallback. 
    # To avoid loading the whole model, we can mock it or just load it once.
    model = load_model(device="cuda")
    
    result = agent_solve(model, img, task["question"])
    print(f"Prediction: {result.prediction}")
    print(f"Method: {result.method}")
    print(f"Ground Truth: {task.get('answer')}")
    
    for i, step in enumerate(result.steps):
        print(f"Step {i}: {step.plan}")
        if step.execution:
            print(f"  Success: {step.execution.get('success')}")
            if step.execution.get('stdout'):
                print(f"  Stdout: {step.execution.get('stdout')}")
            if not step.execution.get('success'):
                print(f"  Error: {step.execution.get('exception')}")
                print(f"  Stderr: {step.execution.get('stderr')}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_task(sys.argv[1])
    else:
        test_task("dev_008") # line_angle
        print("-" * 40)
        test_task("dev_007") # region_fraction
