"""
Data preparation and test infrastructure for autoInferSpeed.

THIS FILE MUST NOT BE MODIFIED BY THE AGENT.
It defines the fixed test inputs, accuracy verification, hardware detection,
and data classes. Only the human may edit this file.

Changes to test parameters, accuracy prompts, or hardware specs go here.
Changes to inference optimization go in inference.py.
"""

import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# =============================================================================
# Fixed Model and Test Configuration
# =============================================================================

MODEL_NAME = "mlx-community/Qwen3.5-2B-8bit"  # Fixed: 8-bit quantized via MLX
MODEL_PARAMS_B = 2.0
MODEL_SIZE_GB = 2.47

# Prefill sequence length settings and TTFT targets (ms)
PREFILL_SEQ_LENS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
PREFILL_TTFT_TARGETS = {  # ms
    32: 15, 64: 20, 128: 30, 256: 50, 512: 100,
    1024: 200, 2048: 400, 4096: 800, 8192: 1600, 16384: 3200,
}

# Decode latency constraint modes (ms per token, None = unlimited)
DECODE_LATENCY_CONSTRAINTS = [20, 30, 50, 100, None]
DECODE_CONTEXT_DEPTHS = [256, 1024]  # tokens of context for decode tests
DECODE_TOKENS = 50  # tokens to generate per sequence

# Accuracy settings
ACCURACY_MAX_TOKENS = 50

# Accuracy test prompts — fixed, do not change
ACCURACY_PROMPTS = [
    # Factual recall
    "The capital of France is",
    # Arithmetic
    "What is 127 + 385? The answer is",
    # Code generation
    "Write a Python function that returns the factorial of n:\ndef factorial(n):",
    # Reasoning
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer:",
    # Long-form coherence
    "Explain how photosynthesis works in three sentences.",
]

# Reference file for baseline outputs (used for accuracy verification)
REFERENCE_FILE = ".baseline_reference.pkl"

# Results file
RESULTS_FILE = "results.tsv"

# Apple Silicon hardware specs
CHIP_BANDWIDTH = {"M1": 68.25, "M2": 100.0, "M3": 100.0, "M4": 120.0}  # GB/s
CHIP_TFLOPS = {"M1": 2.6, "M2": 3.6, "M3": 4.8, "M4": 5.4}  # FP16/FP32 mixed


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
    semantic_type: str  # PASS, FAIL, N/A


@dataclass
class AccuracySuiteResult:
    """Full accuracy suite result."""
    results: List[AccuracyResult] = field(default_factory=list)
    overall_pass: bool = True
    avg_match_pct: float = 0.0
    overall_status: str = "PASS"  # PASS, WARN, FAIL, BASELINE
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
# Hardware Detection
# =============================================================================

def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def get_chip_info() -> Tuple[str, int, float, float]:
    """Detect Apple Silicon chip, GPU cores, peak bandwidth, and peak TFLOPS.

    Returns: (chip_name, gpu_cores, peak_bandwidth_gbs, peak_tflops)
    """
    chip_name = "Unknown"
    gpu_cores = 0

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        chip_name = result.stdout.strip()

        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10,
        )
        import re
        match = re.search(r"Total Number of Cores:\s*(\d+)", result.stdout)
        if match:
            gpu_cores = int(match.group(1))
    except Exception:
        pass

    bandwidth = 100.0  # default M2
    tflops = 3.6
    for chip_key, bw in CHIP_BANDWIDTH.items():
        if chip_key in chip_name:
            bandwidth = bw
            tflops = CHIP_TFLOPS.get(chip_key, 3.6)
            break

    return chip_name, gpu_cores, bandwidth, tflops


# =============================================================================
# Test Data Generation
# =============================================================================

def generate_synthetic_prompt(tokenizer, seq_len: int) -> str:
    """Generate a synthetic prompt of approximately seq_len tokens.

    Uses a simple repeated pattern that tokenizes consistently.
    The accuracy suite uses separate real prompts (ACCURACY_PROMPTS).
    """
    base = "The quick brown fox jumps over the lazy dog. "
    tokens = tokenizer.encode(base)
    repeats = (seq_len // len(tokens)) + 1
    full_text = base * repeats
    full_tokens = tokenizer.encode(full_text)
    # Trim to exact length
    trimmed = tokenizer.decode(full_tokens[:seq_len])
    return trimmed


# =============================================================================
# Accuracy Verification
# =============================================================================

def verify_semantic_answer(prompt: str, response: str) -> Tuple[bool, str]:
    """Check if the response is semantically correct for verifiable prompts.

    Returns: (is_correct, status_string)
    """
    response_lower = response.lower().strip()

    # Arithmetic: "What is 127 + 385? The answer is"
    if "127 + 385" in prompt:
        return ("512" in response, "PASS" if "512" in response else "FAIL")

    # Factual: "The capital of France is"
    if "capital of france" in prompt.lower():
        return ("paris" in response_lower, "PASS" if "paris" in response_lower else "FAIL")

    # Reasoning: syllogism about roses
    if "roses are flowers" in prompt.lower():
        is_correct = any(
            x in response_lower
            for x in ["no", "cannot", "not necessarily", "invalid", "fallacy"]
        )
        return (is_correct, "PASS" if is_correct else "FAIL")

    # Code and long-form prompts don't have verifiable answers
    return (True, "N/A")


def load_reference_outputs() -> Optional[List[List[int]]]:
    """Load baseline reference outputs from pickle file."""
    import pickle

    if os.path.exists(REFERENCE_FILE):
        try:
            with open(REFERENCE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def save_reference_outputs(outputs: List[List[int]]):
    """Save baseline reference outputs to pickle file."""
    import pickle

    with open(REFERENCE_FILE, "wb") as f:
        pickle.dump(outputs, f)
