

import argparse
import json
import time
import statistics
from pathlib import Path

from PIL import Image

from utils import (
    load_jsonl, save_jsonl, resolve_image_path,
    set_seed, compute_metrics, extract_answer,
)
from model_wrapper import load_model, measure_vram
from agent import solve as agent_solve
from baseline import solve_baseline


def _save_trace(output_dir: str, task_id: str, agent_result):
    """Persist per-task trace (step code, logs, artifacts)."""
    trace_dir = Path(output_dir) / "traces" / task_id
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace_data = agent_result.to_dict() if hasattr(agent_result, "to_dict") else {}
    trace_data["id"] = task_id

    # Save per-step code files
    for step in trace_data.get("steps", []):
        k = step.get("k", 0)
        step_dir = trace_dir / f"step_{k}"
        step_dir.mkdir(exist_ok=True)
        if step.get("code"):
            (step_dir / "code.py").write_text(step["code"], encoding="utf-8")
        # Save execution logs
        exec_info = step.get("execution", {})
        if exec_info:
            log_text = f"success: {exec_info.get('success')}\n"
            log_text += f"stdout:\n{exec_info.get('stdout', '')}\n"
            log_text += f"stderr:\n{exec_info.get('stderr', '')}\n"
            if exec_info.get("exception"):
                log_text += f"exception: {exec_info['exception']}\n"
            log_text += f"result: {exec_info.get('result', {})}\n"
            (step_dir / "log.txt").write_text(log_text, encoding="utf-8")

    # Write trace.json
    trace_json = {
        "id": task_id,
        "question": trace_data.get("question", ""),
        "steps": trace_data.get("steps", []),
        "final": {
            "prediction": trace_data.get("prediction", ""),
            "method": trace_data.get("method", ""),
        },
    }
    (trace_dir / "trace.json").write_text(
        json.dumps(trace_json, indent=2, default=str), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Vision evaluation runner"
    )
    parser.add_argument("--data", required=True,
                        help="Path to .jsonl task file")
    parser.add_argument("--out", required=True,
                        help="Output directory")
    parser.add_argument("--baseline", action="store_true",
                        help="Run Direct VQA baseline instead of agent")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--max-samples", type=int, default=None, help="Stop after N samples.")
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Max agent steps per task (default: 5)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Sandbox timeout per step in seconds (default: 5)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for model inference (default: cuda)")
    args = parser.parse_args()

    # ── setup ──
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data).resolve()
    tasks = load_jsonl(args.data)
    if args.max_samples:
        tasks = tasks[:args.max_samples]
    print(f"Loaded {len(tasks)} tasks from {args.data}")

    model = load_model(device=args.device)
    vram = measure_vram()
    mode = "BASELINE" if args.baseline else "AGENT"
    print(f"Mode: {mode}  |  VRAM: {vram:.0f} MB  |  Seed: {args.seed}")
    print(f"Max steps: {args.max_steps}  |  Timeout: {args.timeout}s")
    print("-" * 60)

    # ── run ──
    predictions = []
    runtimes = []
    step_counts = []
    failures = 0

    for i, task in enumerate(tasks):
        tid = task["id"]
        question = task["question"]
        image_field = task["image"]

        try:
            img_path = resolve_image_path(image_field, str(data_path))
        except FileNotFoundError as e:
            print(f"[{i+1}/{len(tasks)}] {tid}: IMAGE NOT FOUND — {e}")
            predictions.append({"id": tid, "prediction": ""})
            failures += 1
            continue

        img = Image.open(img_path).convert("RGB")

        t0 = time.time()
        try:
            if args.baseline:
                pred = solve_baseline(model, img, question)
                method = "baseline"
                n_steps = 1
            else:
                result = agent_solve(
                    model, img, question,
                    max_steps=args.max_steps,
                    timeout=args.timeout,
                )
                pred = result.prediction
                method = result.method
                n_steps = len(result.steps)

                # Save trace
                _save_trace(str(out_dir), tid, result)
        except Exception as exc:
            pred = ""
            method = "error"
            n_steps = 0
            failures += 1
            print(f"  ERROR: {exc}")

        elapsed = time.time() - t0
        runtimes.append(elapsed)
        step_counts.append(n_steps)

        predictions.append({"id": tid, "prediction": pred})
        gt = task.get("answer", "?")
        match = "[OK]" if pred.lower() == str(gt).lower() else "[KO]"
        print(f"[{i+1}/{len(tasks)}] {tid}: pred={pred}  gt={gt}  "
              f"{match}  ({method}, {n_steps} steps, {elapsed:.1f}s)")

    # ── save predictions ──
    pred_path = out_dir / "predictions.jsonl"
    save_jsonl(str(pred_path), predictions)
    print(f"\nPredictions saved to {pred_path}")

    # ── metrics (if ground truth available) ──
    has_gt = all("answer" in t for t in tasks)
    if has_gt:
        metrics = compute_metrics(predictions, tasks)
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Accuracy:       {metrics['accuracy']:.1%}  "
              f"({metrics['correct']}/{metrics['total']})")
        print(f"Failure rate:   {metrics['failure_rate']:.1%}  "
              f"({metrics['failures']} failures)")
        print(f"Mean runtime:   {statistics.mean(runtimes):.2f}s")
        print(f"Median runtime: {statistics.median(runtimes):.2f}s")
        if step_counts:
            print(f"Avg steps:      {statistics.mean(step_counts):.1f}")

        if metrics.get("per_type"):
            print("\nPer-type accuracy:")
            for ttype, info in sorted(metrics["per_type"].items()):
                print(f"  {ttype:25s} {info['accuracy']:.1%}  "
                      f"({info['correct']}/{info['total']})")

        # Save metrics
        metrics_out = {
            "accuracy": metrics["accuracy"],
            "correct": metrics["correct"],
            "total": metrics["total"],
            "failure_rate": metrics["failure_rate"],
            "failures": metrics["failures"],
            "mean_runtime": statistics.mean(runtimes),
            "median_runtime": statistics.median(runtimes),
            "avg_steps": statistics.mean(step_counts) if step_counts else 0,
            "per_type": metrics.get("per_type", {}),
            "mode": mode,
            "seed": args.seed,
            "vram_mb": vram,
        }
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics_out, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nMetrics saved to {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
