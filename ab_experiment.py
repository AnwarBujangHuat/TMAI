#!/usr/bin/env python3
"""Run and compare scratch vs transfer KWS experiments under one protocol."""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime


def model_size_mb(path):
    if not os.path.exists(path):
        return None
    return os.path.getsize(path) / (1024 * 1024)


def load_prediction_results(path):
    with open(path, "r") as f:
        payload = json.load(f)

    items = {}
    for row in payload.get("results", []):
        filename = row.get("filename")
        if not filename:
            continue
        items[filename] = {
            "prediction": row.get("prediction"),
            "probabilities": row.get("probabilities", {}),
        }
    return items


def infer_weak_label(filename):
    """Heuristic label mapping for flat test folder conventions."""
    f = filename.lower()
    if f.startswith("positive") or "heytm" in f or "hey_tm" in f:
        return "heytm"
    if "background" in f or "silence" in f or "noise" in f:
        return "background"
    return "unknown"


def compute_metrics(pred_map):
    if not pred_map:
        return {
            "total_files": 0,
            "predicted_counts": {},
            "keyword_recall": None,
            "false_trigger_rate": None,
            "weak_label_accuracy": None,
            "avg_positive_score_on_positive": None,
            "avg_positive_score_on_non_positive": None,
        }

    predicted_counts = Counter(v.get("prediction") for v in pred_map.values())

    positives = []
    non_positives = []
    weak_correct = 0

    for filename, row in pred_map.items():
        gt = infer_weak_label(filename)
        pred = row.get("prediction")
        probs = row.get("probabilities", {})
        pos_score = float(probs.get("positive", 0.0))

        if gt == "heytm":
            positives.append((pred, pos_score))
        else:
            non_positives.append((pred, pos_score))

        if pred == gt:
            weak_correct += 1

    keyword_recall = None
    if positives:
        keyword_recall = sum(1 for pred, _ in positives if pred == "heytm") / len(positives)

    false_trigger_rate = None
    if non_positives:
        false_trigger_rate = (
            sum(1 for pred, _ in non_positives if pred == "heytm") / len(non_positives)
        )

    avg_pos_on_pos = None
    if positives:
        avg_pos_on_pos = sum(score for _, score in positives) / len(positives)

    avg_pos_on_non_pos = None
    if non_positives:
        avg_pos_on_non_pos = sum(score for _, score in non_positives) / len(non_positives)

    weak_acc = weak_correct / len(pred_map)

    return {
        "total_files": len(pred_map),
        "predicted_counts": dict(predicted_counts),
        "keyword_recall": keyword_recall,
        "false_trigger_rate": false_trigger_rate,
        "weak_label_accuracy": weak_acc,
        "avg_positive_score_on_positive": avg_pos_on_pos,
        "avg_positive_score_on_non_positive": avg_pos_on_non_pos,
        "positive_eval_count": len(positives),
        "non_positive_eval_count": len(non_positives),
    }


def choose_winner(scratch_metrics, transfer_metrics):
    # Primary: keyword recall. Secondary: lower false trigger rate.
    s_recall = scratch_metrics.get("keyword_recall")
    t_recall = transfer_metrics.get("keyword_recall")
    s_far = scratch_metrics.get("false_trigger_rate")
    t_far = transfer_metrics.get("false_trigger_rate")

    if s_recall is None or t_recall is None:
        return "undetermined", "Missing comparable recall metrics."

    if abs(s_recall - t_recall) > 1e-9:
        winner = "scratch" if s_recall > t_recall else "transfer"
        return winner, "Higher keyword recall on positive samples."

    if s_far is not None and t_far is not None and abs(s_far - t_far) > 1e-9:
        winner = "scratch" if s_far < t_far else "transfer"
        return winner, "Same recall; lower false trigger rate on non-positive samples."

    return "tie", "Metrics are effectively tied under current protocol."


def run_cmd(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def write_markdown(path, results):
    scratch = results["models"]["scratch"]
    transfer = results["models"]["transfer"]
    winner = results["winner"]

    lines = [
        "# A/B Experiment Summary: Scratch vs Transfer",
        "",
        f"Generated: `{results['timestamp']}`",
        "",
        "## Winner",
        f"- Decision: **{winner['decision']}**",
        f"- Reason: {winner['reason']}",
        "",
        "## Scratch Metrics",
        f"- Keyword recall: `{fmt_pct(scratch['metrics'].get('keyword_recall'))}`",
        f"- False trigger rate: `{fmt_pct(scratch['metrics'].get('false_trigger_rate'))}`",
        f"- Weak-label accuracy: `{fmt_pct(scratch['metrics'].get('weak_label_accuracy'))}`",
        f"- Avg positive score on positives: `{fmt_num(scratch['metrics'].get('avg_positive_score_on_positive'))}`",
        f"- Avg positive score on non-positives: `{fmt_num(scratch['metrics'].get('avg_positive_score_on_non_positive'))}`",
        f"- Quantized model size (MB): `{fmt_num(scratch['artifacts'].get('quant_tflite_mb'))}`",
        "",
        "## Transfer Metrics",
        f"- Keyword recall: `{fmt_pct(transfer['metrics'].get('keyword_recall'))}`",
        f"- False trigger rate: `{fmt_pct(transfer['metrics'].get('false_trigger_rate'))}`",
        f"- Weak-label accuracy: `{fmt_pct(transfer['metrics'].get('weak_label_accuracy'))}`",
        f"- Avg positive score on positives: `{fmt_num(transfer['metrics'].get('avg_positive_score_on_positive'))}`",
        f"- Avg positive score on non-positives: `{fmt_num(transfer['metrics'].get('avg_positive_score_on_non_positive'))}`",
        f"- Quantized model size (MB): `{fmt_num(transfer['artifacts'].get('quant_tflite_mb'))}`",
        "",
        "## Protocol Notes",
        "- Both methods use the same external `test/` folder outputs.",
        "- For flat test files, labels are inferred with filename heuristic:",
        "  - `positive*` => `heytm`",
        "  - names containing `background|silence|noise` => `background`",
        "  - otherwise => `unknown`",
        "- Prefer replacing heuristic labels with a manually labeled test set for final model selection.",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%"


def fmt_num(v):
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def main():
    parser = argparse.ArgumentParser(description="A/B compare scratch vs transfer KWS")
    parser.add_argument("--run-training", action="store_true", help="Run training scripts before compare")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable")
    parser.add_argument("--scratch-script", default="main_preset_model.py")
    parser.add_argument("--transfer-script", default="transfer_model.py")
    parser.add_argument("--scratch-preds", default="test_predictions.json")
    parser.add_argument("--transfer-preds", default="test_predictions_transfer.json")
    parser.add_argument("--out-json", default="ab_results.json")
    parser.add_argument("--out-md", default="ab_summary.md")
    args = parser.parse_args()

    if args.run_training:
        run_cmd([args.python_bin, args.scratch_script])
        run_cmd([args.python_bin, args.transfer_script])

    if not os.path.exists(args.scratch_preds):
        raise FileNotFoundError(f"Missing scratch prediction file: {args.scratch_preds}")
    if not os.path.exists(args.transfer_preds):
        raise FileNotFoundError(f"Missing transfer prediction file: {args.transfer_preds}")

    scratch_preds = load_prediction_results(args.scratch_preds)
    transfer_preds = load_prediction_results(args.transfer_preds)

    common_files = sorted(set(scratch_preds.keys()) & set(transfer_preds.keys()))
    scratch_common = {k: scratch_preds[k] for k in common_files}
    transfer_common = {k: transfer_preds[k] for k in common_files}

    scratch_metrics = compute_metrics(scratch_common)
    transfer_metrics = compute_metrics(transfer_common)

    disagreements = []
    for name in common_files:
        s_pred = scratch_common[name].get("prediction")
        t_pred = transfer_common[name].get("prediction")
        if s_pred != t_pred:
            disagreements.append({"filename": name, "scratch": s_pred, "transfer": t_pred})

    winner_decision, winner_reason = choose_winner(scratch_metrics, transfer_metrics)

    results = {
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "common_file_count": len(common_files),
            "scratch_prediction_file": args.scratch_preds,
            "transfer_prediction_file": args.transfer_preds,
        },
        "winner": {"decision": winner_decision, "reason": winner_reason},
        "models": {
            "scratch": {
                "metrics": scratch_metrics,
                "artifacts": {
                    "quant_tflite": "models_scratch/KWS_scratch.tflite",
                    "quant_tflite_mb": model_size_mb("models_scratch/KWS_scratch.tflite"),
                    "float_tflite": "models_scratch/KWS_scratch_float.tflite",
                    "float_tflite_mb": model_size_mb("models_scratch/KWS_scratch_float.tflite"),
                },
            },
            "transfer": {
                "metrics": transfer_metrics,
                "artifacts": {
                    "quant_tflite": "models_transfer/KWS_mobilenet_transfer.tflite",
                    "quant_tflite_mb": model_size_mb("models_transfer/KWS_mobilenet_transfer.tflite"),
                    "float_tflite": "models_transfer/KWS_mobilenet_transfer_float.tflite",
                    "float_tflite_mb": model_size_mb("models_transfer/KWS_mobilenet_transfer_float.tflite"),
                },
            },
        },
        "disagreements": disagreements,
    }

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    write_markdown(args.out_md, results)

    print("\nA/B comparison finished.")
    print(f"- JSON: {args.out_json}")
    print(f"- Summary: {args.out_md}")
    print(f"- Common files compared: {len(common_files)}")
    print(f"- Winner: {winner_decision} ({winner_reason})")


if __name__ == "__main__":
    main()
