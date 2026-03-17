"""
Inference Speed Benchmark - Mac Mini Edition
Find the fastest way to run Qwen3.5-2B on Apple Silicon.

Usage: python benchmark.py
"""

import os
import sys
import time
import json
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime

# Configuration - AGENT MODIFIES THESE
MODEL_NAME = "Qwen/Qwen3-1.7B"  # Smaller for faster testing
PROMPT = "Write a short story about a robot learning to paint."
MAX_TOKENS = 100
NUM_WARMUP = 1
NUM_RUNS = 3
BATCH_SIZE = 1  # Fixed per requirements

# Results file
RESULTS_FILE = "results.tsv"

# Baseline targets (from research)
# M1/M2 typical: 30-60 tok/sec for 2B model
# MLX optimized: 2x improvement possible
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


def measure_generation(gen_func, prompt: str, max_tokens: int) -> Tuple[str, float, float, int]:
    """
    Measure generation performance.
    Returns: (generated_text, ttft_ms, total_ms, num_tokens)
    """
    start = time.perf_counter()

    # Generate with timing
    result = gen_func(prompt, max_tokens)

    end = time.perf_counter()
    total_ms = (end - start) * 1000

    # Extract results
    if isinstance(result, tuple):
        text, ttft_ms, tokens = result
    else:
        text = result
        ttft_ms = total_ms  # Approximate if not provided
        tokens = len(text.split())  # Rough estimate

    return text, ttft_ms, total_ms, tokens


# =============================================================================
# Backend Implementations
# =============================================================================

def benchmark_transformers_mps(precision: str = "fp16") -> Optional[BenchmarkResult]:
    """Benchmark using HuggingFace transformers with MPS."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        model.eval()

        mem_before = get_memory_mb()

        def generate(prompt: str, max_tokens: int) -> Tuple[str, float, int]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Time to first token
            ttft_start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            ttft_ms = (time.perf_counter() - ttft_start) * 1000

            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            tokens = len(outputs[0]) - inputs["input_ids"].shape[1]
            return text, ttft_ms, tokens

        # Warmup
        print("  Warming up...")
        generate(PROMPT, 10)

        # Benchmark
        print("  Benchmarking...")
        ttfts, latencies, tokens_list = [], [], []
        for _ in range(NUM_RUNS):
            _, ttft_ms, total_ms, tokens = measure_generation(generate, PROMPT, MAX_TOKENS)
            ttfts.append(ttft_ms)
            latencies.append(total_ms)
            tokens_list.append(tokens)

        mem_after = get_memory_mb()

        avg_ttft = sum(ttfts) / len(ttfts)
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
            time_to_first_token_ms=avg_ttft,
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
        import mlx.core as mx
        from mlx_lm import load, generate

        print(f"  Loading model with MLX ({precision})...")

        mem_before = get_memory_mb()

        # Load model
        model, tokenizer = load(MODEL_NAME)

        def gen(prompt: str, max_tokens: int) -> Tuple[str, float, int]:
            ttft_start = time.perf_counter()

            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=0.0,  # Greedy for reproducibility
            )

            ttft_ms = (time.perf_counter() - ttft_start) * 1000

            # Count tokens
            tokens = len(tokenizer.encode(response))

            return response, ttft_ms, tokens

        # Warmup
        print("  Warming up...")
        gen(PROMPT, 10)

        # Benchmark
        print("  Benchmarking...")
        ttfts, latencies, tokens_list = [], [], []
        for _ in range(NUM_RUNS):
            _, ttft_ms, total_ms, tokens = measure_generation(gen, PROMPT, MAX_TOKENS)
            ttfts.append(ttft_ms)
            latencies.append(total_ms)
            tokens_list.append(tokens)

        mem_after = get_memory_mb()

        avg_ttft = sum(ttfts) / len(ttfts)
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_list) / len(tokens_list)
        tok_per_sec = (avg_tokens / avg_latency) * 1000

        # Cleanup
        del model

        return BenchmarkResult(
            backend="mlx",
            precision=precision,
            tokens_per_sec=tok_per_sec,
            time_to_first_token_ms=avg_ttft,
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


def benchmark_mlx_quantized(bits: int = 4) -> Optional[BenchmarkResult]:
    """Benchmark MLX with quantization."""
    try:
        import mlx.core as mx
        from mlx_lm import load, generate
        from mlx_lm.utils import quantize_model

        print(f"  Loading model with MLX ({bits}-bit quantization)...")

        mem_before = get_memory_mb()

        # Load and quantize
        model, tokenizer = load(MODEL_NAME)
        model, _ = quantize_model(model, bits=bits)

        def gen(prompt: str, max_tokens: int) -> Tuple[str, float, int]:
            ttft_start = time.perf_counter()
            response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, temp=0.0)
            ttft_ms = (time.perf_counter() - ttft_start) * 1000
            tokens = len(tokenizer.encode(response))
            return response, ttft_ms, tokens

        # Warmup
        print("  Warming up...")
        gen(PROMPT, 10)

        # Benchmark
        print("  Benchmarking...")
        ttfts, latencies, tokens_list = [], [], []
        for _ in range(NUM_RUNS):
            _, ttft_ms, total_ms, tokens = measure_generation(gen, PROMPT, MAX_TOKENS)
            ttfts.append(ttft_ms)
            latencies.append(total_ms)
            tokens_list.append(tokens)

        mem_after = get_memory_mb()

        avg_ttft = sum(ttfts) / len(ttfts)
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_list) / len(tokens_list)
        tok_per_sec = (avg_tokens / avg_latency) * 1000

        del model

        return BenchmarkResult(
            backend="mlx_quantized",
            precision=f"{bits}bit",
            tokens_per_sec=tok_per_sec,
            time_to_first_token_ms=avg_ttft,
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


def benchmark_llamacpp() -> Optional[BenchmarkResult]:
    """Benchmark using llama.cpp (if GGUF model available)."""
    try:
        from llama_cpp import Llama

        print("  Loading model with llama.cpp...")

        mem_before = get_memory_mb()

        # Try to find or download GGUF model
        # For Qwen, we'd need to convert or find a pre-converted GGUF
        # This is a simplified version
        gguf_path = os.path.expanduser("~/.cache/inferspeed/model.gguf")

        if not os.path.exists(gguf_path):
            print("  GGUF model not found, skipping llama.cpp")
            return None

        llm = Llama(
            model_path=gguf_path,
            n_ctx=512,
            n_gpu_layers=-1,  # Use all GPU layers
            verbose=False,
        )

        def gen(prompt: str, max_tokens: int) -> Tuple[str, float, int]:
            ttft_start = time.perf_counter()
            output = llm(prompt, max_tokens=max_tokens, temperature=0.0)
            ttft_ms = (time.perf_counter() - ttft_start) * 1000
            text = output["choices"][0]["text"]
            tokens = output["usage"]["completion_tokens"]
            return text, ttft_ms, tokens

        # Warmup
        print("  Warming up...")
        gen(PROMPT, 10)

        # Benchmark
        print("  Benchmarking...")
        ttfts, latencies, tokens_list = [], [], []
        for _ in range(NUM_RUNS):
            _, ttft_ms, total_ms, tokens = measure_generation(gen, PROMPT, MAX_TOKENS)
            ttfts.append(ttft_ms)
            latencies.append(total_ms)
            tokens_list.append(tokens)

        mem_after = get_memory_mb()

        avg_ttft = sum(ttfts) / len(ttfts)
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_list) / len(tokens_list)
        tok_per_sec = (avg_tokens / avg_latency) * 1000

        del llm

        return BenchmarkResult(
            backend="llamacpp",
            precision="gguf",
            tokens_per_sec=tok_per_sec,
            time_to_first_token_ms=avg_ttft,
            total_latency_ms=avg_latency,
            tokens_generated=int(avg_tokens),
            memory_mb=mem_after - mem_before,
        )

    except ImportError:
        print("  llama-cpp-python not installed, skipping")
        return None
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
        ("MLX (4-bit)", lambda: benchmark_mlx_quantized(4)),
        ("MLX (8-bit)", lambda: benchmark_mlx_quantized(8)),
        ("llama.cpp", lambda: benchmark_llamacpp()),
    ]

    for name, benchmark_fn in backends:
        print(f"\n[{name}]")
        result = benchmark_fn()
        if result:
            results.append(result)
            print(f"  ✓ {result.tokens_per_sec:.1f} tok/s | TTFT: {result.time_to_first_token_ms:.1f}ms | Memory: {result.memory_mb:.0f}MB")
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

    print(f"\n{'Backend':<20} {'Precision':<10} {'tok/s':>10} {'TTFT(ms)':>10} {'Memory(MB)':>12}")
    print("-"*62)

    for r in sorted_results:
        print(f"{r.backend:<20} {r.precision:<10} {r.tokens_per_sec:>10.1f} {r.time_to_first_token_ms:>10.1f} {r.memory_mb:>12.0f}")

    # Best result
    best = sorted_results[0]
    print("\n" + "-"*62)
    print(f"BEST: {best.backend} ({best.precision})")
    print(f"  Tokens/sec: {best.tokens_per_sec:.1f}")
    print(f"  TTFT: {best.time_to_first_token_ms:.1f}ms")
    print(f"  Memory: {best.memory_mb:.0f}MB")

    # Comparison to baseline
    improvement = ((best.tokens_per_sec - BASELINE_TOK_SEC) / BASELINE_TOK_SEC) * 100
    print(f"  vs Baseline ({BASELINE_TOK_SEC} tok/s): {improvement:+.1f}%")

    print("-"*62)


def main():
    print(f"Device: {'MPS' if __import__('torch').backends.mps.is_available() else 'CPU'}")

    results = run_all_benchmarks()
    save_results(results)
    print_summary(results)

    return results


if __name__ == "__main__":
    main()
