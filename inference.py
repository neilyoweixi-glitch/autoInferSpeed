"""
Inference Speed Benchmark - Apple Silicon Edition
Maximize prefill and decode performance for Qwen3.5-2B 8-bit using MLX.

Usage: uv run inference.py
"""

import os
import subprocess
import time
import mlx.core as mx
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Generator, Any
from datetime import datetime

# =============================================================================
# Configuration - AGENT MODIFIES THESE
# =============================================================================

MODEL_NAME = "mlx-community/Qwen3.5-2B-8bit"  # Fixed: 8-bit quantized via MLX
NUM_WARMUP = 2
NUM_RUNS = 3

# Prefill settings
PREFILL_SEQ_LENS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
PREFILL_TTFT_TARGETS = {  # ms
    32: 15, 64: 20, 128: 30, 256: 50, 512: 100,
    1024: 200, 2048: 400, 4096: 800, 8192: 1600, 16384: 3200
}

# Decode settings
DECODE_LATENCY_CONSTRAINTS = [20, 30, 50, 100, None]  # ms per token, None = unlimited
DECODE_CONTEXT_DEPTHS = [256, 1024]  # tokens of context
DECODE_TOKENS = 50  # tokens to generate per sequence

# Accuracy settings
ACCURACY_MAX_TOKENS = 50

# Optimization flags
USE_STREAM = True
PREFETCH_STEP_SIZE = 8192  # Larger step for better throughput
USE_COMPILE = False  # mx.compile() doesn't help with cached KV
USE_ASYNC_EVAL = True  # Use mx.async_eval() for overlapping

# KV cache quantization (memory optimization for decode)
KV_BITS = None  # No quantization - 8-bit didn't help performance
KV_GROUP_SIZE = 64
KV_QUANT_START = 0

# Results file
RESULTS_FILE = "results.tsv"

# Reference file for baseline outputs
REFERENCE_FILE = ".baseline_reference.pkl"

# Accuracy test prompts
ACCURACY_PROMPTS = [
    "The capital of France is",
    "What is 127 + 385? The answer is",
    "Write a Python function that returns the factorial of n:\ndef factorial(n):",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer:",
    "Explain how photosynthesis works in three sentences.",
]

# Apple Silicon specs
CHIP_BANDWIDTH = {"M1": 68.25, "M2": 100.0, "M3": 100.0, "M4": 120.0}
CHIP_TFLOPS = {"M1": 2.6, "M2": 3.6, "M3": 4.8, "M4": 5.4}

# Model specs for Qwen3.5-2B
MODEL_PARAMS_B = 2.0
MODEL_SIZE_GB = 2.47


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PrefillResult:
    """Result from a prefill benchmark."""
    seq_len: int
    ttft_ms: float
    prefill_tok_s: float
    mfu: float
    memory_mb: float
    target_met: bool


@dataclass
class DecodeResult:
    """Result from a decode benchmark."""
    latency_constraint_ms: Optional[float]
    context_depth: int
    max_batch: int
    total_tok_s: float
    ms_per_tok: float
    memory_mb: float
    bw_util: float
    bottleneck: str


@dataclass
class AccuracyResult:
    """Accuracy verification result for a single prompt."""
    prompt_idx: int
    prompt_preview: str
    token_match: int
    token_total: int
    match_pct: float
    first_divergence: Optional[int]
    semantic_pass: bool
    semantic_type: str


@dataclass
class AccuracySuiteResult:
    """Full accuracy suite result."""
    results: List[AccuracyResult] = field(default_factory=list)
    overall_pass: bool = True
    avg_match_pct: float = 0.0
    overall_status: str = "PASS"
    generated_outputs: Optional[List[List[int]]] = None


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    prefill: List[PrefillResult] = field(default_factory=list)
    decode: List[DecodeResult] = field(default_factory=list)
    accuracy: Optional[AccuracySuiteResult] = None
    chip: str = ""
    gpu_cores: int = 0
    peak_bandwidth_gbs: float = 0.0
    peak_tflops: float = 0.0


# =============================================================================
# Utility Functions
# =============================================================================

def get_memory_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def get_chip_info() -> Tuple[str, int, float, float]:
    """Detect Apple Silicon chip info."""
    chip_name = "Unknown"
    gpu_cores = 0

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        chip_name = result.stdout.strip()

        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10
        )
        import re
        match = re.search(r"Total Number of Cores:\s*(\d+)", result.stdout)
        if match:
            gpu_cores = int(match.group(1))
    except Exception:
        pass

    bandwidth = 100.0
    tflops = 3.6
    for chip_key, bw in CHIP_BANDWIDTH.items():
        if chip_key in chip_name:
            bandwidth = bw
            tflops = CHIP_TFLOPS.get(chip_key, 3.6)
            break

    return chip_name, gpu_cores, bandwidth, tflops


def verify_semantic_answer(prompt: str, response: str) -> Tuple[bool, str]:
    """Check semantic correctness for verifiable prompts."""
    response_lower = response.lower().strip()

    if "127 + 385" in prompt:
        return ("512" in response, "PASS" if "512" in response else "FAIL")

    if "capital of france" in prompt.lower():
        return ("paris" in response_lower, "PASS" if "paris" in response_lower else "FAIL")

    if "roses are flowers" in prompt.lower():
        is_correct = any(x in response_lower for x in ["no", "cannot", "not necessarily", "invalid", "fallacy"])
        return (is_correct, "PASS" if is_correct else "FAIL")

    return (True, "N/A")


def load_reference_outputs() -> Optional[List[List[int]]]:
    """Load baseline reference outputs from file."""
    import pickle
    if os.path.exists(REFERENCE_FILE):
        try:
            with open(REFERENCE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def save_reference_outputs(outputs: List[List[int]]):
    """Save baseline reference outputs to file."""
    import pickle
    with open(REFERENCE_FILE, "wb") as f:
        pickle.dump(outputs, f)


def generate_synthetic_prompt(tokenizer, seq_len: int) -> str:
    """Generate a synthetic prompt of approximately seq_len tokens."""
    # Use a simple repeated pattern that tokenizes consistently
    base = "The quick brown fox jumps over the lazy dog. "
    tokens = tokenizer.encode(base)
    repeats = (seq_len // len(tokens)) + 1
    full_text = base * repeats
    full_tokens = tokenizer.encode(full_text)
    # Trim to exact length
    trimmed = tokenizer.decode(full_tokens[:seq_len])
    return trimmed


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_prefill(model, tokenizer, stream, chip_info) -> List[PrefillResult]:
    """Benchmark prefill across all sequence lengths."""
    from mlx_lm.generate import generate_step
    _, _, _, peak_tflops = chip_info
    peak_gflops = peak_tflops * 1000

    results = []

    for seq_len in PREFILL_SEQ_LENS:
        print(f"  Prefill seq_len={seq_len}...")

        # Generate synthetic prompt tokens
        prompt = generate_synthetic_prompt(tokenizer, seq_len)
        prompt_tokens = mx.array(tokenizer.encode(prompt))

        # Warmup
        mx.clear_cache()
        gen = generate_step(prompt_tokens, model, max_tokens=1, prefill_step_size=PREFETCH_STEP_SIZE,
                          kv_bits=KV_BITS, kv_group_size=KV_GROUP_SIZE, quantized_kv_start=KV_QUANT_START)
        _ = next(gen, None)
        del gen

        # Benchmark - measure time to first token (prefill + first decode step)
        ttfts = []
        for _ in range(NUM_RUNS):
            mx.clear_cache()
            mem_before = get_memory_mb()
            start = time.perf_counter()

            gen = generate_step(prompt_tokens, model, max_tokens=1, prefill_step_size=PREFETCH_STEP_SIZE,
                              kv_bits=KV_BITS, kv_group_size=KV_GROUP_SIZE, quantized_kv_start=KV_QUANT_START)
            first_token, _ = next(gen)

            ttft_ms = (time.perf_counter() - start) * 1000
            ttfts.append(ttft_ms)
            del gen

        mem_after = get_memory_mb()
        avg_ttft = sum(ttfts) / len(ttfts)
        prefill_tok_s = seq_len / (avg_ttft / 1000)

        # Calculate MFU for prefill (compute-bound)
        # FLOPs for prefill = 2 * params * seq_len
        flops = 2 * MODEL_PARAMS_B * seq_len  # in GFLOPs
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
            memory_mb=mem_after - mem_before,
            target_met=target_met,
        ))

    return results


def benchmark_decode(model, tokenizer, stream, chip_info) -> List[DecodeResult]:
    """Benchmark decode across latency constraints and context depths."""
    from mlx_lm.generate import generate_step

    results = []
    _, peak_bandwidth, _, _ = chip_info
    roofline_tok_s = peak_bandwidth / MODEL_SIZE_GB

    for context_depth in DECODE_CONTEXT_DEPTHS:
        print(f"  Decode context={context_depth}...")

        # Create a context prompt
        context_prompt = generate_synthetic_prompt(tokenizer, context_depth)
        context_tokens = mx.array(tokenizer.encode(context_prompt))

        for constraint in DECODE_LATENCY_CONSTRAINTS:
            # Warmup first
            mx.clear_cache()
            gen = generate_step(context_tokens, model, max_tokens=5, prefill_step_size=PREFETCH_STEP_SIZE,
                              kv_bits=KV_BITS, kv_group_size=KV_GROUP_SIZE, quantized_kv_start=KV_QUANT_START)
            for _ in gen:
                pass
            del gen

            # Measure decode-only performance
            decode_times = []
            for _ in range(NUM_RUNS):
                mx.clear_cache()
                start_total = time.perf_counter()

                gen = generate_step(context_tokens, model, max_tokens=DECODE_TOKENS, prefill_step_size=PREFETCH_STEP_SIZE,
                                  kv_bits=KV_BITS, kv_group_size=KV_GROUP_SIZE, quantized_kv_start=KV_QUANT_START)
                tokens_gen = 0
                ttft = None

                for token, _ in gen:
                    tokens_gen += 1
                    if ttft is None:
                        ttft = (time.perf_counter() - start_total) * 1000

                total_ms = (time.perf_counter() - start_total) * 1000
                decode_time = total_ms - ttft if ttft else total_ms
                ms_per_tok = decode_time / (tokens_gen - 1) if tokens_gen > 1 else 0
                decode_times.append(ms_per_tok)
                del gen

            avg_ms_per_tok = sum(decode_times) / len(decode_times)
            total_tok_s = 1000 / avg_ms_per_tok if avg_ms_per_tok > 0 else 0
            mem_after = get_memory_mb()

            # Check if within constraint
            if constraint is None or avg_ms_per_tok < constraint:
                bw_util = total_tok_s / roofline_tok_s

                # Determine bottleneck
                if bw_util > 0.7:
                    bottleneck = "MEMORY_BOUND"
                else:
                    bottleneck = "OVERHEAD_BOUND"

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
                print(f"    constraint={constraint_str}: batch=1 tok/s={total_tok_s:.1f} ms/tok={avg_ms_per_tok:.1f}")
            else:
                results.append(DecodeResult(
                    latency_constraint_ms=constraint,
                    context_depth=context_depth,
                    max_batch=0,
                    total_tok_s=0,
                    ms_per_tok=avg_ms_per_tok,
                    memory_mb=mem_after,
                    bw_util=0,
                    bottleneck="CONSTRAINT_EXCEEDED",
                ))
                print(f"    constraint={constraint}ms: CONSTRAINT_EXCEEDED")

    return results


def benchmark_accuracy(model, tokenizer, reference_outputs=None) -> AccuracySuiteResult:
    """Run accuracy verification."""
    from mlx_lm import generate

    results = []
    generated_outputs = []

    for idx, prompt in enumerate(ACCURACY_PROMPTS):
        output_text = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=ACCURACY_MAX_TOKENS,
            verbose=False
        )

        output_tokens = list(tokenizer.encode(output_text))
        generated_outputs.append(output_tokens)

        if reference_outputs is not None and idx < len(reference_outputs):
            ref_tokens = reference_outputs[idx]
            min_len = min(len(output_tokens), len(ref_tokens))
            matches = sum(1 for i in range(min_len) if output_tokens[i] == ref_tokens[i])

            first_div = None
            for i in range(min_len):
                if output_tokens[i] != ref_tokens[i]:
                    first_div = i
                    break
            if first_div is None and len(output_tokens) != len(ref_tokens):
                first_div = min_len

            match_pct = (matches / len(ref_tokens)) * 100 if ref_tokens else 0
            semantic_pass, semantic_type = verify_semantic_answer(prompt, output_text)

            results.append(AccuracyResult(
                prompt_idx=idx,
                prompt_preview=prompt[:40] + "..." if len(prompt) > 40 else prompt,
                token_match=matches,
                token_total=len(ref_tokens),
                match_pct=match_pct,
                first_divergence=first_div,
                semantic_pass=semantic_pass,
                semantic_type=semantic_type,
            ))
        else:
            results.append(AccuracyResult(
                prompt_idx=idx,
                prompt_preview=prompt[:40] + "..." if len(prompt) > 40 else prompt,
                token_match=len(output_tokens),
                token_total=len(output_tokens),
                match_pct=100.0,
                first_divergence=None,
                semantic_pass=True,
                semantic_type="N/A",
            ))

    if not results:
        return AccuracySuiteResult()

    avg_match = sum(r.match_pct for r in results) / len(results)
    any_semantic_fail = any(not r.semantic_pass for r in results)

    if any_semantic_fail or avg_match < 80.0:
        overall_status = "FAIL"
        overall_pass = False
    elif avg_match < 90.0:
        overall_status = "WARN"
        overall_pass = True
    else:
        overall_status = "PASS"
        overall_pass = True

    return AccuracySuiteResult(
        results=results,
        overall_pass=overall_pass,
        avg_match_pct=avg_match,
        overall_status=overall_status,
        generated_outputs=generated_outputs,
    )


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_benchmark() -> Optional[BenchmarkResults]:
    """Run the complete benchmark suite."""
    from mlx_lm import load

    print("\n" + "=" * 60)
    print("BENCHMARK: Qwen3.5-2B 8-bit")
    print("=" * 60 + "\n")

    try:
        print(f"Loading model: {MODEL_NAME}...")
        mem_before = get_memory_mb()
        model, tokenizer = load(MODEL_NAME)

        stream = mx.stream(mx.gpu) if USE_STREAM else None
        chip_info = get_chip_info()
        chip, gpu_cores, peak_bandwidth, peak_tflops = chip_info

        print(f"Chip: {chip} ({gpu_cores}-core GPU, {peak_bandwidth:.0f} GB/s, {peak_tflops:.1f} TFLOPS)")

        results = BenchmarkResults(
            chip=chip,
            gpu_cores=gpu_cores,
            peak_bandwidth_gbs=peak_bandwidth,
            peak_tflops=peak_tflops,
        )

        # Prefill benchmark
        print("\n[PREFILL BENCHMARK]")
        results.prefill = benchmark_prefill(model, tokenizer, stream, chip_info)

        # Decode benchmark
        print("\n[DECODE BENCHMARK]")
        results.decode = benchmark_decode(model, tokenizer, stream, chip_info)

        # Accuracy verification
        print("\n[ACCURACY VERIFICATION]")
        reference_outputs = load_reference_outputs()
        if reference_outputs is None:
            print("  Creating baseline reference outputs...")
            results.accuracy = benchmark_accuracy(model, tokenizer, None)
            if results.accuracy.generated_outputs:
                save_reference_outputs(results.accuracy.generated_outputs)
            results.accuracy.overall_status = "BASELINE"
        else:
            results.accuracy = benchmark_accuracy(model, tokenizer, reference_outputs)

        del model
        return results

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_report(results: BenchmarkResults):
    """Print the full benchmark report."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    # Prefill results
    print("\nPREFILL BENCHMARK")
    for r in results.prefill:
        target_str = "OK" if r.target_met else "EXCEEDED"
        target_ms = PREFILL_TTFT_TARGETS.get(r.seq_len, "?")
        print(f"  seq_len={r.seq_len:<6} TTFT={r.ttft_ms:>6.0f}ms  prefill_tok/s={r.prefill_tok_s:>6.0f}  MFU={r.mfu*100:>5.1f}%  [< {target_ms}ms {target_str}]")

    # Decode results
    print("\nDECODE BENCHMARK")
    current_ctx = None
    for r in results.decode:
        if r.context_depth != current_ctx:
            current_ctx = r.context_depth
            print(f"  (context={current_ctx})")
        constraint_str = f"{r.latency_constraint_ms}ms" if r.latency_constraint_ms else "none"
        if r.max_batch > 0:
            print(f"    constraint={constraint_str:<6} max_batch={r.max_batch:<3} tok/s={r.total_tok_s:>6.1f}  ms/tok={r.ms_per_tok:>5.1f}  {r.bottleneck}")
        else:
            print(f"    constraint={constraint_str:<6} CONSTRAINT_EXCEEDED")

    # Accuracy results
    if results.accuracy:
        print("\nACCURACY VERIFICATION (vs baseline)")
        for r in results.accuracy.results:
            first_div_str = str(r.first_divergence) if r.first_divergence is not None else "none"
            print(f'  "{r.prompt_preview}"  tokens: {r.token_match}/{r.token_total} ({r.match_pct:.0f}%)  first_div: {first_div_str}  semantic: {r.semantic_type}')
        print(f"  OVERALL: {results.accuracy.overall_status} (avg token match: {results.accuracy.avg_match_pct:.1f}%)")

    # Roofline analysis
    print("\nROOFLINE ANALYSIS")
    print(f"  chip: {results.chip} ({results.gpu_cores}-core GPU, {results.peak_bandwidth_gbs:.0f} GB/s, {results.peak_tflops:.1f} TFLOPS)")
    print(f"  model_size: {MODEL_SIZE_GB:.2f} GB (8-bit quantized)")

    print("\n  PREFILL ROOFLINE (compute-bound regime)")
    print(f"    peak_flops: {results.peak_tflops:.1f} TFLOPS")
    for r in results.prefill:
        if r.seq_len in [256, 1024]:
            achieved_tflops = r.mfu * results.peak_tflops
            print(f"    seq_len={r.seq_len}: achieved {achieved_tflops:.1f} TFLOPS (MFU={r.mfu*100:.0f}%)")

    print("\n  DECODE ROOFLINE (memory-bandwidth-bound at batch=1)")
    roofline_tok_s = results.peak_bandwidth_gbs / MODEL_SIZE_GB
    print(f"    roofline_tok_s: {roofline_tok_s:.1f} tok/s ({results.peak_bandwidth_gbs:.0f} GB/s / {MODEL_SIZE_GB:.2f} GB)")
    for r in results.decode:
        if r.context_depth == 256 and r.latency_constraint_ms == 20:
            print(f"    batch=1: actual {r.total_tok_s:.1f} tok/s, bandwidth_util={r.bw_util*100:.0f}%")

    print("-" * 60)


def save_results(results: BenchmarkResults, description: str = ""):
    """Save results to TSV file."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    accuracy_pct = results.accuracy.avg_match_pct if results.accuracy else 0
    status = "keep" if results.accuracy and results.accuracy.overall_status != "FAIL" else "discard"

    file_exists = os.path.exists(RESULTS_FILE)

    with open(RESULTS_FILE, "a") as f:
        if not file_exists:
            f.write("commit\tstage\tseq_len\tlatency_ms\tcontext\tmax_batch\ttok_sec\tttft_ms\tms_per_tok\tmemory_mb\tmfu_or_bw_util\tbottleneck\taccuracy_pct\tstatus\tdescription\n")

        # Write prefill results
        for r in results.prefill:
            f.write(f"{commit}\tprefill\t{r.seq_len}\t-\t-\t-\t{r.prefill_tok_s:.1f}\t{r.ttft_ms:.0f}\t-\t{r.memory_mb:.0f}\t{r.mfu:.2f}\tCOMPUTE_BOUND\t{accuracy_pct:.1f}\t{status}\t{description}\n")

        # Write decode results
        for r in results.decode:
            latency_str = str(r.latency_constraint_ms) if r.latency_constraint_ms else "none"
            f.write(f"{commit}\tdecode\t-\t{latency_str}\t{r.context_depth}\t{r.max_batch}\t{r.total_tok_s:.1f}\t-\t{r.ms_per_tok:.1f}\t{r.memory_mb:.0f}\t{r.bw_util:.2f}\t{r.bottleneck}\t{accuracy_pct:.1f}\t{status}\t{description}\n")


def main():
    results = run_benchmark()
    if results:
        print_report(results)
        save_results(results, "baseline run")
    else:
        print("\nBenchmark failed.")
    return results


if __name__ == "__main__":
    main()
