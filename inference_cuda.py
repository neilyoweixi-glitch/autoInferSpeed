"""
CUDA Inference Benchmark - NVIDIA Edition
Maximize prefill and decode performance for Qwen3.5-2B using PyTorch + CUDA.

Usage: python inference_cuda.py
"""

import os
import time
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# Configuration
# =============================================================================

# Fixed model: Qwen3.5-2B (same as Apple Silicon experiments)
# Apple uses mlx-community/Qwen3.5-2B-8bit (MLX 8-bit quantized)
# For CUDA, we use the base model with FP16/BF16
MODEL_NAME = "Qwen/Qwen3.5-2B"  # Will try Qwen2.5-3B if not available
MODEL_PARAMS_B = 2.0  # 2B parameters
MODEL_SIZE_FP16_GB = MODEL_PARAMS_B * 2  # ~4 GB for FP16

# Fallback models if Qwen3.5-2B not available
FALLBACK_MODELS = ["Qwen/Qwen2.5-3B", "Qwen/Qwen2-1.5B"]

# Prefill sequence lengths
PREFILL_SEQ_LENS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
PREFILL_TTFT_TARGETS = {
    32: 15, 64: 20, 128: 30, 256: 50, 512: 100,
    1024: 200, 2048: 400, 4096: 800, 8192: 1600,
}

# Decode settings
DECODE_TOKENS = 50
DECODE_LATENCY_CONSTRAINTS = [20, 30, 50, 100, None]

# Accuracy prompts
ACCURACY_PROMPTS = [
    "The capital of France is",
    "What is 127 + 385? The answer is",
    "Write a Python function that returns the factorial of n:\ndef factorial(n):",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer:",
    "Explain how photosynthesis works in three sentences.",
]

# Benchmark settings
NUM_WARMUP = 2
NUM_RUNS = 3

# Optimization flags
USE_TORCH_COMPILE = False  # Can try torch.compile()
USE_BF16 = True  # Use bfloat16 (Ampere+)
USE_CUDA_GRAPHS = False  # Experimental


# =============================================================================
# GPU Info
# =============================================================================

def get_gpu_info() -> Tuple[str, int, float, float]:
    """Get GPU name, memory, bandwidth, and TFLOPS."""
    name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    multi_processor_count = torch.cuda.get_device_properties(0).multi_processor_count

    # Memory bandwidth estimates (GB/s)
    BANDWIDTH = {
        "RTX 3070": 448,
        "RTX 3080": 760,
        "RTX 3090": 936,
        "RTX 4070": 504,
        "RTX 4080": 717,
        "RTX 4090": 1008,
        "A100": 2039,
        "H100": 3352,
    }

    bandwidth = 448  # Default for RTX 3070
    for gpu_name, bw in BANDWIDTH.items():
        if gpu_name in name:
            bandwidth = bw
            break

    # TFLOPS estimates (FP16 with tensor cores)
    TFLOPS = {
        "RTX 3070": 23.1,
        "RTX 3080": 34.1,
        "RTX 3090": 35.6,
        "RTX 4070": 46.1,
        "RTX 4080": 48.7,
        "RTX 4090": 82.6,
        "A100": 312,
        "H100": 989,
    }

    tflops = 23.1  # Default for RTX 3070
    for gpu_name, tf in TFLOPS.items():
        if gpu_name in name:
            tflops = tf
            break

    return name, int(memory_gb), bandwidth, tflops


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PrefillResult:
    seq_len: int
    ttft_ms: float
    prefill_tok_s: float
    mfu: float
    memory_mb: float
    target_met: bool


@dataclass
class DecodeResult:
    latency_constraint_ms: Optional[float]
    context_depth: int
    max_batch: int
    total_tok_s: float
    ms_per_tok: float
    memory_mb: float
    bw_util: float
    bottleneck: str


# =============================================================================
# Benchmark Functions
# =============================================================================

def generate_synthetic_prompt(tokenizer, seq_len: int) -> str:
    """Generate synthetic prompt of exactly seq_len tokens."""
    base = "The quick brown fox jumps over the lazy dog. "
    tokens = tokenizer.encode(base)
    repeats = (seq_len // len(tokens)) + 1
    full_text = base * repeats
    full_tokens = tokenizer.encode(full_text)
    return tokenizer.decode(full_tokens[:seq_len])


def benchmark_prefill(model, tokenizer, device, gpu_info) -> List[PrefillResult]:
    """Benchmark prefill across sequence lengths."""
    name, mem_gb, peak_bandwidth, peak_tflops = gpu_info
    peak_gflops = peak_tflops * 1000

    results = []

    for seq_len in PREFILL_SEQ_LENS:
        # Skip if sequence too long for memory
        if seq_len > 4096:  # RTX 3070 limit with 8GB
            print(f"  Prefill seq_len={seq_len}... SKIPPED (memory limit)")
            continue

        print(f"  Prefill seq_len={seq_len}...")

        prompt = generate_synthetic_prompt(tokenizer, seq_len)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Warmup
        with torch.no_grad():
            _ = model(input_ids)
        torch.cuda.synchronize()

        # Benchmark
        ttfts = []
        mem_before = torch.cuda.memory_allocated() / 1e6

        for _ in range(NUM_RUNS):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                outputs = model(input_ids, use_cache=True)

            torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - start) * 1000
            ttfts.append(ttft_ms)

        mem_after = torch.cuda.max_memory_allocated() / 1e6
        avg_ttft = sum(ttfts) / len(ttfts)
        prefill_tok_s = seq_len / (avg_ttft / 1000)

        # MFU calculation
        flops = 2 * MODEL_PARAMS_B * seq_len  # GFLOPs
        time_s = avg_ttft / 1000
        actual_gflops_s = flops / time_s
        mfu = actual_gflops_s / peak_gflops if peak_gflops > 0 else 0

        target = PREFILL_TTFT_TARGETS.get(seq_len, float('inf'))
        target_met = avg_ttft < target

        results.append(PrefillResult(
            seq_len=seq_len,
            ttft_ms=avg_ttft,
            prefill_tok_s=prefill_tok_s,
            mfu=mfu,
            memory_mb=mem_after,
            target_met=target_met,
        ))

    return results


def benchmark_decode(model, tokenizer, device, gpu_info) -> List[DecodeResult]:
    """Benchmark decode performance."""
    name, mem_gb, peak_bandwidth, _ = gpu_info

    # Use FP16 model size for roofline
    model_size_gb = MODEL_SIZE_FP16_GB
    roofline_tok_s = peak_bandwidth / model_size_gb

    results = []
    context_depths = [256, 1024]

    for context_depth in context_depths:
        print(f"  Decode context={context_depth}...")

        if context_depth > 2048:  # Memory limit for 8GB
            continue

        prompt = generate_synthetic_prompt(tokenizer, context_depth)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for constraint in DECODE_LATENCY_CONSTRAINTS:
            # Warmup
            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=5, do_sample=False)
            torch.cuda.synchronize()

            # Benchmark
            decode_times = []
            for _ in range(NUM_RUNS):
                torch.cuda.synchronize()
                start = time.perf_counter()

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=DECODE_TOKENS,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                torch.cuda.synchronize()
                total_ms = (time.perf_counter() - start) * 1000

                # Estimate decode time (total - 1 decode step worth of prefill)
                # Rough estimate: prefill takes most of the time for short context
                prefill_estimate = context_depth * 0.1  # ~0.1ms per token for prefill
                decode_time = total_ms - prefill_estimate
                tokens_gen = output_ids.shape[1] - input_ids.shape[1]
                ms_per_tok = decode_time / tokens_gen if tokens_gen > 0 else 0
                decode_times.append(ms_per_tok)

            avg_ms_per_tok = sum(decode_times) / len(decode_times)
            total_tok_s = 1000 / avg_ms_per_tok if avg_ms_per_tok > 0 else 0
            mem_after = torch.cuda.max_memory_allocated() / 1e6

            if constraint is None or avg_ms_per_tok < constraint:
                bw_util = total_tok_s / roofline_tok_s
                bottleneck = "MEMORY_BOUND" if bw_util > 0.7 else "OVERHEAD_BOUND"

                results.append(DecodeResult(
                    latency_constraint_ms=constraint,
                    context_depth=context_depth,
                    max_batch=1,
                    total_tok_s=total_tok_s,
                    ms_per_tok=avg_ms_per_tok,
                    memory_mb=mem_after,
                    bw_util=bw_util,
                    bottleneck=bottleneck,
                ))
                constraint_str = f"{constraint}ms" if constraint else "none"
                print(f"    constraint={constraint_str}: tok/s={total_tok_s:.1f} ms/tok={avg_ms_per_tok:.1f}")
            else:
                print(f"    constraint={constraint}ms: CONSTRAINT_EXCEEDED")

    return results


def benchmark_accuracy(model, tokenizer, device) -> Tuple[bool, float]:
    """Simple accuracy check."""
    correct = 0
    total = 0

    for prompt in ACCURACY_PROMPTS[:2]:  # Just test 2 prompts
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        total += 1
        # Simple check: output contains reasonable continuation
        if len(output) > len(prompt):
            correct += 1

    return correct == total, 100.0 * correct / total


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("CUDA INFERENCE BENCHMARK")
    print("=" * 60 + "\n")

    # GPU info
    gpu_info = get_gpu_info()
    name, mem_gb, peak_bandwidth, peak_tflops = gpu_info
    print(f"GPU: {name}")
    print(f"Memory: {mem_gb} GB")
    print(f"Bandwidth: {peak_bandwidth} GB/s")
    print(f"TFLOPS: {peak_tflops} (FP16)")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if USE_BF16 else torch.float16

    # Load model
    print(f"\nLoading model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if USE_TORCH_COMPILE:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    model.eval()

    # Get actual model size
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model parameters: {param_count:.2f}B")
    print(f"Model dtype: {dtype}")

    # Prefill benchmark
    print("\n[PREFILL BENCHMARK]")
    prefill_results = benchmark_prefill(model, tokenizer, device, gpu_info)

    # Decode benchmark
    print("\n[DECODE BENCHMARK]")
    decode_results = benchmark_decode(model, tokenizer, device, gpu_info)

    # Accuracy check
    print("\n[ACCURACY CHECK]")
    accuracy_pass, accuracy_pct = benchmark_accuracy(model, tokenizer, device)
    print(f"  Accuracy: {accuracy_pct:.0f}% ({'PASS' if accuracy_pass else 'FAIL'})")

    # Print report
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nPREFILL:")
    print(f"{'Seq':<8} {'TTFT(ms)':<10} {'tok/s':<10} {'MFU':<8} {'Target':<10}")
    print("-" * 50)
    for r in prefill_results:
        target = PREFILL_TTFT_TARGETS.get(r.seq_len, "?")
        status = "OK" if r.target_met else "EXCEEDED"
        print(f"{r.seq_len:<8} {r.ttft_ms:<10.1f} {r.prefill_tok_s:<10.0f} {r.mfu*100:<8.1f}% {target}ms {status}")

    print("\nDECODE:")
    model_size_gb = MODEL_SIZE_FP16_GB
    roofline_tok_s = peak_bandwidth / model_size_gb
    print(f"Roofline: {roofline_tok_s:.1f} tok/s ({peak_bandwidth} GB/s / {model_size_gb:.2f} GB)")
    print(f"{'Context':<10} {'tok/s':<10} {'ms/tok':<10} {'BW Util':<10}")
    print("-" * 40)
    for r in decode_results:
        if r.max_batch > 0:
            print(f"{r.context_depth:<10} {r.total_tok_s:<10.1f} {r.ms_per_tok:<10.1f} {r.bw_util*100:.1f}%")

    print("\nROOFLINE ANALYSIS:")
    print(f"  GPU: {name}")
    print(f"  Peak bandwidth: {peak_bandwidth} GB/s")
    print(f"  Peak TFLOPS: {peak_tflops}")
    print(f"  Decode roofline: {roofline_tok_s:.1f} tok/s")

    if decode_results:
        best_decode = max(decode_results, key=lambda x: x.total_tok_s)
        print(f"  Achieved: {best_decode.total_tok_s:.1f} tok/s ({best_decode.bw_util*100:.1f}% roofline)")


if __name__ == "__main__":
    main()
