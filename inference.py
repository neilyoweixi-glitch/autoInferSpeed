"""
Inference Speed Benchmark - Mac Mini Edition
Maximize tokens/sec for Qwen3.5-2B 8-bit on Apple Silicon using MLX.

Usage: uv run inference.py
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime

# Configuration - AGENT MODIFIES THESE
MODEL_NAME = "mlx-community/Qwen3.5-2B-8bit"  # Fixed: 8-bit quantized via MLX
PROMPT = "Write a short story about a robot learning to paint."
MAX_TOKENS = 50
NUM_WARMUP = 1
NUM_RUNS = 3
BATCH_SIZE = 1  # Fixed per requirements

# Results file
RESULTS_FILE = "results.tsv"

# Baseline targets (from initial benchmarks)
BASELINE_TOK_SEC = 40.3


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    tokens_per_sec: float
    time_to_first_token_ms: float
    total_latency_ms: float
    tokens_generated: int
    memory_mb: float
    notes: str = ""


def get_memory_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


# =============================================================================
# MLX Backend (Apple Silicon only)
# =============================================================================

def benchmark_mlx() -> Optional[BenchmarkResult]:
    """Benchmark Qwen3.5-2B 8-bit using Apple MLX framework."""
    try:
        import mlx.core as mx
        from mlx_lm import load, stream_generate

        print(f"  Loading model: {MODEL_NAME}...")

        mem_before = get_memory_mb()

        model, tokenizer = load(MODEL_NAME)

        def gen(prompt: str, max_tokens: int) -> Tuple[str, float, float, int]:
            """Generate tokens and measure timing.

            Returns: (response_text, ttft_ms, total_ms, tokens_generated)
            """
            start = time.perf_counter()
            ttft = None
            tokens_generated = 0
            response = ""

            for chunk in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000
                response += chunk.text if hasattr(chunk, 'text') else str(chunk)
                tokens_generated += 1

            total_ms = (time.perf_counter() - start) * 1000
            if ttft is None:
                ttft = total_ms

            return response, ttft, total_ms, tokens_generated

        # Warmup
        print("  Warming up...")
        for _ in range(NUM_WARMUP):
            gen(PROMPT, 5)

        # Benchmark
        print("  Benchmarking...")
        latencies, ttfts, tokens_list = [], [], []
        for _ in range(NUM_RUNS):
            _, ttft, total_ms, tokens = gen(PROMPT, MAX_TOKENS)
            latencies.append(total_ms)
            ttfts.append(ttft)
            tokens_list.append(tokens)

        mem_after = get_memory_mb()

        avg_latency = sum(latencies) / len(latencies)
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tokens = sum(tokens_list) / len(tokens_list)
        tok_per_sec = (avg_tokens / avg_latency) * 1000

        del model

        return BenchmarkResult(
            tokens_per_sec=tok_per_sec,
            time_to_first_token_ms=avg_ttft,
            total_latency_ms=avg_latency,
            tokens_generated=int(avg_tokens),
            memory_mb=mem_after - mem_before,
        )

    except ImportError:
        print("  ERROR: MLX not installed. This benchmark requires Apple Silicon with MLX.")
        print("  Install with: uv sync")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


# =============================================================================
# Main Runner
# =============================================================================

def run_benchmark() -> Optional[BenchmarkResult]:
    """Run the MLX inference benchmark."""
    print("\n" + "=" * 60)
    print("INFERENCE SPEED BENCHMARK")
    print(f"Model: {MODEL_NAME}")
    print(f"Backend: MLX (Apple Silicon)")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Runs: {NUM_RUNS}")
    print("=" * 60 + "\n")

    print("[MLX 8-bit]")
    result = benchmark_mlx()
    if result:
        print(f"  {result.tokens_per_sec:.1f} tok/s | TTFT: {result.time_to_first_token_ms:.0f}ms | Latency: {result.total_latency_ms:.0f}ms | Memory: {result.memory_mb:.0f}MB")

    return result


def save_results(result: BenchmarkResult):
    """Save result to TSV file."""
    file_exists = os.path.exists(RESULTS_FILE)

    with open(RESULTS_FILE, "a") as f:
        if not file_exists:
            f.write("timestamp\ttokens_per_sec\tttft_ms\tlatency_ms\ttokens\tmemory_mb\tnotes\n")

        f.write(
            f"{datetime.now().isoformat()}\t"
            f"{result.tokens_per_sec:.2f}\t"
            f"{result.time_to_first_token_ms:.1f}\t"
            f"{result.total_latency_ms:.1f}\t"
            f"{result.tokens_generated}\t"
            f"{result.memory_mb:.0f}\t"
            f"{result.notes}\n"
        )


def print_summary(result: BenchmarkResult):
    """Print summary."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n  Tokens/sec:  {result.tokens_per_sec:.1f}")
    print(f"  TTFT:        {result.time_to_first_token_ms:.0f} ms")
    print(f"  Latency:     {result.total_latency_ms:.0f} ms")
    print(f"  Tokens:      {result.tokens_generated}")
    print(f"  Memory:      {result.memory_mb:.0f} MB")

    improvement = ((result.tokens_per_sec - BASELINE_TOK_SEC) / BASELINE_TOK_SEC) * 100
    print(f"\n  vs Baseline ({BASELINE_TOK_SEC} tok/s): {improvement:+.1f}%")
    print("-" * 60)


def main():
    result = run_benchmark()
    if result:
        save_results(result)
        print_summary(result)
    else:
        print("\nBenchmark failed. MLX with Apple Silicon is required.")

    return result


if __name__ == "__main__":
    main()
