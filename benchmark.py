"""
Inference Speed Benchmark - Mac Mini Edition
Find the fastest way to run Qwen3.5-2B on Apple Silicon.

Usage: python benchmark.py
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime

# Configuration - AGENT MODIFIES THESE
MODEL_NAME = "Qwen/Qwen3.5-2B"  # Target model for Mac
PROMPT = "Write a short story about a robot learning to paint."
MAX_TOKENS = 50  # Shorter for faster testing
NUM_WARMUP = 1
NUM_RUNS = 3
BATCH_SIZE = 1  # Fixed per requirements

# Results file
RESULTS_FILE = "results.tsv"

# Baseline targets (from research)
BASELINE_TOK_SEC = 30


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    backend: str
    precision: str
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
    except:
        return 0.0


# =============================================================================
# Backend Implementations
# =============================================================================

def benchmark_transformers_mps(precision: str = "fp16") -> Optional[BenchmarkResult]:
    """Benchmark using HuggingFace transformers with MPS."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        print(f"  Loading model with transformers ({precision})...")

        # Device selection
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Load model
        dtype = torch.float16 if precision == "fp16" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        # Fix for Qwen3.5 - use device_map instead of .to()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=device,
        )
        model.eval()

        mem_before = get_memory_mb()

        def generate(prompt: str, max_tokens: int) -> Tuple[str, float, int]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            total_ms = (time.perf_counter() - start) * 1000

            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            num_tokens = len(new_tokens)

            return text, total_ms, num_tokens

        # Warmup
        print("  Warming up...")
        generate(PROMPT, 5)

        # Benchmark
        print("  Benchmarking...")
        latencies, tokens_list = [], []
        for _ in range(NUM_RUNS):
            _, total_ms, tokens = generate(PROMPT, MAX_TOKENS)
            latencies.append(total_ms)
            tokens_list.append(tokens)

        mem_after = get_memory_mb()

        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_list) / len(tokens_list)
        tok_per_sec = (avg_tokens / avg_latency) * 1000

        # Cleanup
        del model
        del tokenizer
        if device == "mps":
            torch.mps.empty_cache()

        return BenchmarkResult(
            backend="transformers",
            precision=precision,
            tokens_per_sec=tok_per_sec,
            time_to_first_token_ms=avg_latency,  # Approximate (no streaming)
            total_latency_ms=avg_latency,
            tokens_generated=int(avg_tokens),
            memory_mb=mem_after - mem_before,
        )

    except Exception as e:
        print(f"  Error: {e}")
        return None


def benchmark_mlx(precision: str = "fp16") -> Optional[BenchmarkResult]:
    """Benchmark using Apple MLX framework."""
    try:
        from mlx_lm import load, generate

        print(f"  Loading model with MLX ({precision})...")

        mem_before = get_memory_mb()

        # Load model
        model, tokenizer = load(MODEL_NAME)

        def gen(prompt: str, max_tokens: int) -> Tuple[str, float, int]:
            start = time.perf_counter()

            # MLX-LM generate returns the full response
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )

            total_ms = (time.perf_counter() - start) * 1000

            # Count tokens
            tokens = len(tokenizer.encode(response))

            return response, total_ms, tokens

        # Warmup
        print("  Warming up...")
        gen(PROMPT, 5)

        # Benchmark
        print("  Benchmarking...")
        latencies, tokens_list = [], []
        for _ in range(NUM_RUNS):
            _, total_ms, tokens = gen(PROMPT, MAX_TOKENS)
            latencies.append(total_ms)
            tokens_list.append(tokens)

        mem_after = get_memory_mb()

        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_list) / len(tokens_list)
        tok_per_sec = (avg_tokens / avg_latency) * 1000

        del model

        return BenchmarkResult(
            backend="mlx",
            precision=precision,
            tokens_per_sec=tok_per_sec,
            time_to_first_token_ms=avg_latency,  # Approximate
            total_latency_ms=avg_latency,
            tokens_generated=int(avg_tokens),
            memory_mb=mem_after - mem_before,
        )

    except ImportError:
        print("  MLX not installed, skipping")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def benchmark_mlx_prequantized(bits: int = 4) -> Optional[BenchmarkResult]:
    """Benchmark MLX with pre-quantized model from HuggingFace."""
    try:
        from mlx_lm import load, generate

        # Try MLX Community quantized models
        quantized_model = f"mlx-community/{MODEL_NAME.split('/')[1]}-{bits}bit"

        print(f"  Loading pre-quantized model: {quantized_model}...")

        mem_before = get_memory_mb()

        try:
            model, tokenizer = load(quantized_model)
        except Exception as e:
            print(f"  Pre-quantized model not available: {e}")
            return None

        def gen(prompt: str, max_tokens: int) -> Tuple[str, float, int]:
            start = time.perf_counter()
            response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
            total_ms = (time.perf_counter() - start) * 1000
            tokens = len(tokenizer.encode(response))
            return response, total_ms, tokens

        # Warmup
        print("  Warming up...")
        gen(PROMPT, 5)

        # Benchmark
        print("  Benchmarking...")
        latencies, tokens_list = [], []
        for _ in range(NUM_RUNS):
            _, total_ms, tokens = gen(PROMPT, MAX_TOKENS)
            latencies.append(total_ms)
            tokens_list.append(tokens)

        mem_after = get_memory_mb()

        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_list) / len(tokens_list)
        tok_per_sec = (avg_tokens / avg_latency) * 1000

        del model

        return BenchmarkResult(
            backend="mlx_quantized",
            precision=f"{bits}bit",
            tokens_per_sec=tok_per_sec,
            time_to_first_token_ms=avg_latency,
            total_latency_ms=avg_latency,
            tokens_generated=int(avg_tokens),
            memory_mb=mem_after - mem_before,
        )

    except ImportError:
        print("  MLX not installed, skipping")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def benchmark_ollama() -> Optional[BenchmarkResult]:
    """Benchmark using Ollama (if installed)."""
    try:
        import subprocess
        import json

        print("  Checking Ollama...")

        # Check if ollama is installed
        result = subprocess.run(["which", "ollama"], capture_output=True)
        if result.returncode != 0:
            print("  Ollama not installed, skipping")
            return None

        mem_before = get_memory_mb()

        # Pull model if needed
        model_name = MODEL_NAME.split("/")[1].lower()
        print(f"  Using Ollama model: {model_name}")

        def gen(prompt: str, max_tokens: int) -> Tuple[str, float, int]:
            start = time.perf_counter()

            result = subprocess.run(
                ["ollama", "run", model_name, prompt],
                capture_output=True,
                text=True,
                timeout=120,
            )

            total_ms = (time.perf_counter() - start) * 1000
            response = result.stdout.strip()

            # Rough token count (words * 1.3)
            tokens = int(len(response.split()) * 1.3)

            return response, total_ms, tokens

        # Warmup
        print("  Warming up...")
        gen("Hi", 5)

        # Benchmark
        print("  Benchmarking...")
        latencies, tokens_list = [], []
        for _ in range(NUM_RUNS):
            _, total_ms, tokens = gen(PROMPT, MAX_TOKENS)
            latencies.append(total_ms)
            tokens_list.append(tokens)

        mem_after = get_memory_mb()

        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_list) / len(tokens_list)
        tok_per_sec = (avg_tokens / avg_latency) * 1000

        return BenchmarkResult(
            backend="ollama",
            precision="native",
            tokens_per_sec=tok_per_sec,
            time_to_first_token_ms=avg_latency,
            total_latency_ms=avg_latency,
            tokens_generated=int(avg_tokens),
            memory_mb=mem_after - mem_before,
        )

    except Exception as e:
        print(f"  Error: {e}")
        return None


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_all_benchmarks() -> List[BenchmarkResult]:
    """Run all available benchmarks."""
    results = []

    print("\n" + "="*60)
    print("INFERENCE SPEED BENCHMARK")
    print(f"Model: {MODEL_NAME}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Runs per benchmark: {NUM_RUNS}")
    print("="*60 + "\n")

    # Test each backend
    backends = [
        ("Transformers (MPS, fp16)", lambda: benchmark_transformers_mps("fp16")),
        ("Transformers (MPS, fp32)", lambda: benchmark_transformers_mps("fp32")),
        ("MLX (fp16)", lambda: benchmark_mlx("fp16")),
        ("MLX 4-bit (pre-quantized)", lambda: benchmark_mlx_prequantized(4)),
        ("MLX 8-bit (pre-quantized)", lambda: benchmark_mlx_prequantized(8)),
        ("Ollama", lambda: benchmark_ollama()),
    ]

    for name, benchmark_fn in backends:
        print(f"\n[{name}]")
        result = benchmark_fn()
        if result:
            results.append(result)
            print(f"  ✓ {result.tokens_per_sec:.1f} tok/s | Latency: {result.total_latency_ms:.0f}ms | Memory: {result.memory_mb:.0f}MB")
        else:
            print(f"  ✗ Skipped")

    return results


def save_results(results: List[BenchmarkResult]):
    """Save results to TSV file."""
    file_exists = os.path.exists(RESULTS_FILE)

    with open(RESULTS_FILE, "a") as f:
        if not file_exists:
            f.write("timestamp\tbackend\tprecision\ttokens_per_sec\tttft_ms\tlatency_ms\ttokens\tmemory_mb\tnotes\n")

        for r in results:
            f.write(f"{datetime.now().isoformat()}\t{r.backend}\t{r.precision}\t{r.tokens_per_sec:.2f}\t{r.time_to_first_token_ms:.1f}\t{r.total_latency_ms:.1f}\t{r.tokens_generated}\t{r.memory_mb:.0f}\t{r.notes}\n")


def print_summary(results: List[BenchmarkResult]):
    """Print summary of results."""
    if not results:
        print("\nNo results to summarize.")
        return

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Sort by tokens per second
    sorted_results = sorted(results, key=lambda x: x.tokens_per_sec, reverse=True)

    print(f"\n{'Backend':<25} {'Precision':<10} {'tok/s':>8} {'Latency(ms)':>12} {'Memory(MB)':>10}")
    print("-"*65)

    for r in sorted_results:
        print(f"{r.backend:<25} {r.precision:<10} {r.tokens_per_sec:>8.1f} {r.total_latency_ms:>12.0f} {r.memory_mb:>10.0f}")

    # Best result
    best = sorted_results[0]
    print("\n" + "-"*65)
    print(f"BEST: {best.backend} ({best.precision})")
    print(f"  Tokens/sec: {best.tokens_per_sec:.1f}")
    print(f"  Latency: {best.total_latency_ms:.0f}ms")
    print(f"  Memory: {best.memory_mb:.0f}MB")

    # Comparison to baseline
    improvement = ((best.tokens_per_sec - BASELINE_TOK_SEC) / BASELINE_TOK_SEC) * 100
    print(f"  vs Baseline ({BASELINE_TOK_SEC} tok/s): {improvement:+.1f}%")
    print("-"*65)


def main():
    print(f"Device: {'MPS' if __import__('torch').backends.mps.is_available() else 'CPU'}")

    results = run_all_benchmarks()
    save_results(results)
    print_summary(results)

    return results


if __name__ == "__main__":
    main()
