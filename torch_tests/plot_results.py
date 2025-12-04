#!/usr/bin/env python3
import os
import re
import argparse
from typing import Dict, Tuple, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for batch use
import matplotlib.pyplot as plt


def parse_scores_file(path: str) -> Dict[str, Any]:
    """
    Parse a result_*.res file and extract:
      - per-model scores per mode
      - global per-mode scores
      - accumulated total score
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    per_model: Dict[Tuple[str, str], float] = {}
    global_scores: Dict[str, float] = {}
    total_score = None

    state = None  # None | "per_model" | "global" | "total"

    for line in lines:
        stripped = line.strip()

        if "Per-model scores" in line:
            state = "per_model"
            continue
        if "Global per-mode scores" in line:
            state = "global"
            continue
        if "Accumulated score" in line:
            state = "total"
            continue

        # End of a block on blank line or separator
        if stripped == "" or stripped.startswith("===="):
            state = None
            continue

        if state == "per_model":
            # Example: "  convnext_base  AMP :   1681.4"
            m = re.match(r"\s*(\S+)\s+(\S+)\s*:\s*([0-9.]+)", line)
            if m:
                model, mode, score = m.groups()
                per_model[(model, mode)] = float(score)
        elif state == "global":
            # Example: "  FP32 score:   1156.7"
            m = re.match(r"\s*(\S+)\s+score\s*:\s*([0-9.]+)", line)
            if m:
                mode, score = m.groups()
                global_scores[mode] = float(score)
        elif state == "total":
            # Example: "  TOTAL:   3281.0"
            m = re.match(r"\s*TOTAL\s*:\s*([0-9.]+)", line)
            if m:
                total_score = float(m.group(1))

    return {
        "file": path,
        "per_model": per_model,
        "global": global_scores,
        "total": total_score,
    }


def label_from_filename(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    # Strip leading "result_" if present
    if name.startswith("result_"):
        name = name[len("result_") :]
    return name


def plot_per_file(result: Dict[str, Any], output_dir: str) -> str:
    """
    For one file, create a grouped bar plot of per-model scores per mode.
    """
    per_model = result["per_model"]
    label = label_from_filename(result["file"])

    if not per_model:
        return ""

    # Custom order: tiny, small, base, large
    size_order = {"tiny": 0, "small": 1, "base": 2, "large": 3}

    def model_sort_key(m: str):
        suffix = m.split("_")[-1]  # e.g. convnext_tiny -> tiny
        return (size_order.get(suffix, 99), m)

    models = sorted({m for (m, _) in per_model.keys()}, key=model_sort_key)
    modes = sorted({mode for (_, mode) in per_model.keys()})

    x = np.arange(len(models))
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, mode in enumerate(modes):
        scores = [per_model.get((model, mode), 0.0) for model in models]
        offset = (i - (len(modes) - 1) / 2.0) * width
        ax.bar(x + offset, scores, width=width, label=mode)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Images/s")
    ax.set_title(f"Per-model scores: {label}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{label}_per_model.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def plot_comparison_totals(results: List[Dict[str, Any]], output_dir: str) -> str:
    """
    Comparison plot of total scores per file.
    """
    labels = []
    totals = []

    for res in results:
        label = label_from_filename(res["file"])
        total = res["total"]

        if total is None:
            # Fallback if TOTAL is not present
            if res["global"]:
                total = sum(res["global"].values())
            else:
                total = sum(res["per_model"].values())

        labels.append(label)
        totals.append(total)

    if not labels:
        return ""

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, totals, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Total score")
    ax.set_title("Accumulated score comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "comparison_totals.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def plot_comparison_global_modes(results: List[Dict[str, Any]], output_dir: str) -> str:
    """
    Comparison plot of global per-mode scores across files.
    """
    modes_all = sorted({mode for r in results for mode in r["global"].keys()})
    if not modes_all:
        return ""

    labels = [label_from_filename(r["file"]) for r in results]
    x = np.arange(len(labels))
    width = 0.8 / max(len(modes_all), 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, mode in enumerate(modes_all):
        scores = [r["global"].get(mode, 0.0) for r in results]
        offset = (i - (len(modes_all) - 1) / 2.0) * width
        ax.bar(x + offset, scores, width=width, label=mode)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Global score (images/s)")
    ax.set_title("Global per-mode score comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "comparison_global_modes.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Parse benchmark .res files and create per-file and comparison plots."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Result files to parse (e.g. result_*.res)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="plots",
        help="Directory to store generated plots (default: plots)",
    )

    args = parser.parse_args()

    # Expand globs manually if necessary
    all_paths: List[str] = []
    for pattern in args.files:
        # If pattern is an existing file, use as-is; else try glob
        if os.path.isfile(pattern):
            all_paths.append(pattern)
        else:
            import glob

            matches = glob.glob(pattern)
            all_paths.extend(matches)

    # Deduplicate
    all_paths = sorted(set(all_paths))

    results: List[Dict[str, Any]] = []
    for path in all_paths:
        if not os.path.isfile(path):
            continue
        res = parse_scores_file(path)
        results.append(res)
        plot_per_file(res, args.output_dir)

    if results:
        plot_comparison_totals(results, args.output_dir)
        plot_comparison_global_modes(results, args.output_dir)


if __name__ == "__main__":
    main()
