#!/usr/bin/env python3
"""
Benchmark ConvNeXt models (tiny/small/base/large) in PyTorch using synthetic data.

Models:
    - convnext_tiny
    - convnext_small
    - convnext_base
    - convnext_large

Batch sizes:
    - 4, 8, 16, 32

Runs on:
    - CUDA GPU if available (or forced with --device cuda)
    - CPU if forced (with --device cpu) or if CUDA is not available

Requirements:
    pip install torch torchvision

Usage:
    python bench_convnext_all.py
    python bench_convnext_all.py --device cpu
    python bench_convnext_all.py --device cuda
"""

import sys
import os
import time
import argparse
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torchvision
from torchvision.models import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
)

def configure_threads(auto_threads: bool = True, reserve_cores: int = 0) -> None:
    if not auto_threads:
        return

    logical_cores = os.cpu_count() or 1
    num_threads = max(1, logical_cores - reserve_cores)

    # PyTorch threading
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, min(4, reserve_cores)))  # small inter-op pool

    # BLAS / OpenMP backends
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)

    print(f"Auto thread config: logical_cores={logical_cores}, "
          f"num_threads={num_threads}")

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


class NoOpContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def benchmark_model(
    model: nn.Module,
    model_name: str,
    device: torch.device,
    batch_sizes: List[int],
    img_size: int = 224,
    warmup_iters: int = 10,
    bench_iters: int = 30,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Returns:
        results[mode_name][batch_size] = {
            "throughput": imgs_per_sec,
            "avg_latency": avg,
            "p50_latency": p50,
            "p90_latency": p90,
        }
    """
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    is_cuda = device.type == "cuda"
    results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for use_amp in (False, True if is_cuda else False):
        mode_name = "AMP" if use_amp else "FP32"
        print_header(f"{model_name} | {mode_name}")

        # Prepare autocast context
        if use_amp and is_cuda:
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            autocast_ctx = NoOpContext()

        results[mode_name] = {}

        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")

            # Pre-allocate input once for this batch size
            try:
                x = torch.randn(
                    batch_size,
                    3,
                    img_size,
                    img_size,
                    device=device,
                    dtype=torch.float16 if use_amp and is_cuda else torch.float32,
                )
            except RuntimeError as e:
                print(f"  Failed to allocate input tensor (likely OOM): {e}")
                continue

            if is_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            # Warmup
            for _ in range(warmup_iters):
                with autocast_ctx:
                    _ = model(x)
                if is_cuda:
                    torch.cuda.synchronize()

            # Benchmark
            times: List[float] = []
            for i in range(bench_iters):
                try:
                    start = time.time()
                    with autocast_ctx:
                        _ = model(x)
                    if is_cuda:
                        torch.cuda.synchronize()
                    end = time.time()
                    times.append(end - start)
                except RuntimeError as e:
                    print(f"  Iteration {i} failed (likely OOM): {e}")
                    break

            if not times:
                print("  No successful iterations, skipping stats.")
                continue

            times_sorted = sorted(times)
            avg = sum(times) / len(times)
            p50 = times_sorted[len(times_sorted) // 2]
            p90 = times_sorted[int(len(times_sorted) * 0.9)]
            imgs_per_sec = batch_size / avg

            print(f"  Avg latency / batch : {avg * 1000:.3f} ms")
            print(f"  Median latency      : {p50 * 1000:.3f} ms")
            print(f"  p90 latency         : {p90 * 1000:.3f} ms")
            print(f"  Throughput          : {imgs_per_sec:.1f} images/s")

            if is_cuda:
                peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                print(f"  Peak memory         : {peak_mem:.1f} MiB")

            results[mode_name][batch_size] = {
                "throughput": imgs_per_sec,
                "avg_latency": avg,
                "p50_latency": p50,
                "p90_latency": p90,
            }

    return results


def compute_scores(
    all_results: Dict[str, Dict[str, Dict[int, Dict[str, float]]]]
) -> None:
    """
    Score design:

    - For each model and mode (FP32 / AMP), take the average throughput (images/s)
      across all successfully measured batch sizes. This gives a per-model, per-mode score.
    - Per-mode global score (FP32 score, AMP score) is the average of the per-model
      scores of that mode.
    - Accumulated score is the sum of the per-mode global scores (FP32 + AMP) for this device.
      Higher is better. Same script on different GPUs/CPUs gives comparable scalar numbers.
    """
    print_header("Scores")

    per_model_scores: Dict[str, Dict[str, float]] = {}
    per_mode_scores: Dict[str, List[float]] = {}

    # Compute per-model, per-mode scores
    for model_name, model_results in all_results.items():
        per_model_scores[model_name] = {}
        for mode_name, mode_results in model_results.items():
            if not mode_results:
                continue
            throughputs = [m["throughput"] for m in mode_results.values()]
            if not throughputs:
                continue
            score = sum(throughputs) / len(throughputs)  # mean throughput over batch sizes
            per_model_scores[model_name][mode_name] = score
            per_mode_scores.setdefault(mode_name, []).append(score)

    # Print per-model scores
    print("Per-model scores (higher is better, units ~ images/s averaged over batch sizes):")
    for model_name in sorted(per_model_scores.keys()):
        modes = per_model_scores[model_name]
        for mode_name in sorted(modes.keys()):
            score = modes[mode_name]
            print(f"  {model_name:14s} {mode_name:4s}: {score:8.1f}")

    # Global per-mode scores
    print("\nGlobal per-mode scores:")
    global_mode_scores: Dict[str, float] = {}
    for mode_name, scores in per_mode_scores.items():
        if not scores:
            continue
        global_score = sum(scores) / len(scores)
        global_mode_scores[mode_name] = global_score
        print(f"  {mode_name:4s} score: {global_score:8.1f}")

    # Accumulated score
    if global_mode_scores:
        accumulated = sum(global_mode_scores.values())
        print("\nAccumulated score (FP32 + AMP where available):")
        print(f"  TOTAL: {accumulated:8.1f}")
    else:
        print("\nNo scores computed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="ConvNeXt synthetic benchmark")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Select device: 'auto' (default), 'cpu', or 'cuda'",
    )
    args = parser.parse_args()

    print_header("Environment")
    print(f"Python version   : {sys.version.split()[0]}")
    print(f"PyTorch version  : {torch.__version__}")
    print(f"Torchvision ver. : {torchvision.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available   : {cuda_available}")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not cuda_available:
            print("Requested CUDA device but CUDA is not available.")
            return 1
        device = torch.device("cuda:0")
    else:  # auto
        device = torch.device("cuda:0" if cuda_available else "cpu")

    if device.type == "cpu":
        configure_threads()

    print(f"Using device     : {device}")

    models: List[Tuple[str, nn.Module]] = [
        ("convnext_tiny", convnext_tiny(weights=None)),
        ("convnext_small", convnext_small(weights=None)),
        ("convnext_base", convnext_base(weights=None)),
        ("convnext_large", convnext_large(weights=None)),
    ]

    batch_sizes = [4, 8, 16, 32]

    all_results: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}

    for name, model in models:
        print_header(f"Model: {name}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params / 1e6:.2f} M")

        model_results = benchmark_model(
            model=model,
            model_name=name,
            device=device,
            batch_sizes=batch_sizes,
            img_size=224,
            warmup_iters=10,
            bench_iters=30,
        )
        all_results[name] = model_results

    compute_scores(all_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
