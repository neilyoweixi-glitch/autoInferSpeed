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

def benchmark_transformers(precision: str = "fp16") -> Optional[BenchmarkResult]:
    """Benchmark using HuggingFace transformers with MPS (fixed for Qwen3.5)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoConfig

        print(f"  Loading model with transformers ({precision})...")

        # Device selection
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Load model - use legacy mode to avoid config issues with Qwen3.5
        dtype = torch.float16 if precision == "fp16" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        # Load config and patch for Qwen3.5 compatibility
        # Qwen3.5 has nested config where model params are in text_config
        config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if hasattr(config, 'text_config'):
            # Copy all text_config attributes to main config for compatibility
            text_config = config.text_config
            for attr in ['vocab_size', 'hidden_size', 'num_hidden_layers',
                         'num_attention_heads', 'intermediate_size', 'max_position_embeddings',
                         'pad_token_id', 'bos_token_id', 'eos_token_id']:
                if hasattr(text_config, attr) and not hasattr(config, attr):
                    setattr(config, attr, getattr(text_config, attr))
            # Also set model_type to help with architecture detection
            if hasattr(text_config, 'model_type'):
                config.model_type = text_config.model_type

        # Load model with patched config
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Manually move to device
        model = model.to(device)
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
                    use_cache=True,
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
            time_to_first_token_ms=avg_latency,  # Approximate
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
            start = time.perf_counter()

            # Use stream_generate for token counting
            from mlx_lm import stream_generate
            tokens_generated = 0
            response = ""

            for chunk in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
                response += chunk.text if hasattr(chunk, 'text') else str(chunk)
                tokens_generated += 1

            total_ms = (time.perf_counter() - start) * 1000

            return response, total_ms, tokens_generated

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


def benchmark_mlx_prequantized(bits: int = 4) -> Optional[BenchmarkResult]:
    """Benchmark MLX with pre-quantized model."""
    try:
        import mlx.core as mx
        from mlx_lm import load, stream_generate

        # Map model name to pre-quantized version
        quantized_model = MODEL_NAME.replace("Qwen/", "mlx-community/").replace("-2B", f"-2B-{bits}bit")

        print(f"  Loading pre-quantized model: {quantized_model}...")

        mem_before = get_memory_mb()

        model, tokenizer = load(quantized_model)

        def gen(prompt: str, max_tokens: int) -> Tuple[str, float, int]:
            start = time.perf_counter()

            tokens_generated = 0
            response = ""

            for chunk in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
                response += chunk.text if hasattr(chunk, 'text') else str(chunk)
                tokens_generated += 1

            total_ms = (time.perf_counter() - start) * 1000

            return response, total_ms, tokens_generated

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


def benchmark_vllm() -> Optional[BenchmarkResult]:
    """Benchmark using vLLM (if available)."""
    try:
        from vllm import LLM, SamplingParams

        print("  Loading model with vLLM...")

        mem_before = get_memory_mb()

        # vLLM-Metal for Apple Silicon
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            dtype="float16",
            # For Apple Silicon, vLLM uses MLX backend
        )

        sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=0.0,  # Greedy
        )

        def gen(prompt: str) -> Tuple[str, float, int]:
            start = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params)
            total_ms = (time.perf_counter() - start) * 1000

            text = outputs[0].outputs[0].text
            tokens = len(outputs[0].outputs[0].token_ids)

            return text, total_ms, tokens

        # Warmup
        print("  Warming up...")
        gen(PROMPT)

        # Benchmark
        print("  Benchmarking...")
        latencies, tokens_list = [], []
        for _ in range(NUM_RUNS):
            _, total_ms, tokens = gen(PROMPT)
            latencies.append(total_ms)
            tokens_list.append(tokens)

        mem_after = get_memory_mb()

        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_list) / len(tokens_list)
        tok_per_sec = (avg_tokens / avg_latency) * 1000

        del llm

        return BenchmarkResult(
            backend="vllm",
            precision="fp16",
            tokens_per_sec=tok_per_sec,
            time_to_first_token_ms=avg_latency,
            total_latency_ms=avg_latency,
            tokens_generated=int(avg_tokens),
            memory_mb=mem_after - mem_before,
        )

    except ImportError:
        print("  vLLM not installed, skipping")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def benchmark_sglang() -> Optional[BenchmarkResult]:
    """Benchmark using SGLang (if available)."""
    try:
        import sglang as sgl

        print("  Loading model with SGLang...")

        mem_before = get_memory_mb()

        # SGLang runtime
        runtime = sgl.Runtime(
            model_path=MODEL_NAME,
            trust_remote_code=True,
        )

        def gen(prompt: str, max_tokens: int) -> Tuple[str, float, int]:
            start = time.perf_counter()

            response = runtime.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.0,
            )

            total_ms = (time.perf_counter() - start) * 1000

            # Count tokens
            tokens = len(runtime.tokenizer.encode(response))

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

        del runtime

        return BenchmarkResult(
            backend="sglang",
            precision="fp16",
            tokens_per_sec=tok_per_sec,
            time_to_first_token_ms=avg_latency,
            total_latency_ms=avg_latency,
            tokens_generated=int(avg_tokens),
            memory_mb=mem_after - mem_before,
        )

    except ImportError:
        print("  SGLang not installed, skipping")
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
        ("Transformers (MPS, fp16)", lambda: benchmark_transformers("fp16")),
        ("MLX (fp16)", lambda: benchmark_mlx("fp16")),
        ("MLX 4-bit (pre-quantized)", lambda: benchmark_mlx_prequantized(4)),
        ("MLX 8-bit (pre-quantized)", lambda: benchmark_mlx_prequantized(8)),
        ("vLLM", lambda: benchmark_vllm()),
        ("SGLang", lambda: benchmark_sglang()),
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
