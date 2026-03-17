# autoInferSpeed

Autonomous LLM inference optimization on Apple Silicon. An AI agent iterates on a benchmark harness to maximize tokens/sec for **Qwen3.5-2B 8-bit** with batch_size=1, while verifying correctness and tracking how close it gets to the hardware roofline.

Inspired by [autoresearch](https://github.com/karpathy/autoresearch), but instead of training a better model, we're squeezing maximum inference speed from a fixed model on Apple Silicon.

## How it works

The repo has three files that matter:

- **`inference.py`** — the single file the agent edits. MLX-only inference harness with accuracy verification and roofline analysis. **This file is edited and iterated on by the agent**.
- **`program.md`** — instructions for the autonomous agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.
- **`results.tsv`** — experiment log. Tab-separated results from all benchmark runs, including speed, accuracy, and roofline metrics.

The agent modifies `inference.py`, runs the benchmark, checks that outputs are still correct, measures how close to the hardware roofline it is, logs results, keeps improvements, discards regressions, and repeats.

## Key principles

- **Fixed model**: Qwen3.5-2B 8-bit only. No model switching — we optimize the inference path, not the model choice.
- **Correctness first**: Every optimization must pass an accuracy verification suite that compares outputs against a baseline reference. Speed without correctness is rejected.
- **Roofline-aware**: Each run reports hardware bandwidth utilization and identifies the current bottleneck (memory-bound, compute-bound, overhead-bound, CPU-bound). This guides the agent toward the right optimization strategy.
- **Full hardware utilization**: Explores using both GPU and CPU simultaneously — offloading tokenization, embedding lookup, KV cache management to CPU while GPU handles compute.

## Quick start

**Requirements:** Apple Silicon Mac (M1/M2/M3/M4), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Run a single benchmark manually
uv run inference.py
```

If the above works, your setup is ready for autonomous mode.

## Running the agent

Spin up Claude Code (or your preferred agent) in this repo, then prompt:

```
Hi, have a look at program.md and let's kick off inference optimization! Let's do the setup first.
```

## Project structure

```
inference.py    — inference harness + accuracy verification + roofline analysis (agent modifies this)
program.md      — autonomous agent protocol
results.tsv     — experiment log (untracked by git)
pyproject.toml  — dependencies
CLAUDE.md       — project context for Claude Code
```

## What the agent optimizes

The agent focuses on these categories, guided by the roofline bottleneck analysis:

| Bottleneck | Optimization strategies |
|------------|------------------------|
| OVERHEAD_BOUND | Reduce Python overhead, pre-allocate KV cache, optimize JIT compilation |
| MEMORY_BOUND | KV cache quantization, weight prefetching, reduce memory traffic |
| COMPUTE_BOUND | Custom Metal kernels, fused operations, optimized tiling |
| CPU_BOUND | CPU+GPU co-execution, parallel tokenization, offload embedding to CPU |

## Accuracy verification

Every benchmark run includes a correctness check against baseline reference outputs:

- **5 fixed test prompts** covering factual recall, arithmetic, code generation, reasoning, and coherence
- **Token-level exact match** against the unmodified model's greedy-decoded output
- **Semantic correctness** for prompts with verifiable answers
- **Pass/fail gate**: optimizations that drop below 80% token match or produce wrong answers are rejected

## Current results

From initial benchmarks on Qwen3.5-2B 8-bit (MLX):

| Metric | Value |
|--------|-------|
| tok/s | ~40 |
| TTFT | ~1240 ms |
| Memory | ~1900 MB |

## Design choices

- **Single model, deep optimization.** Instead of searching across models, we fix the model and go deep — kernel optimizations, memory layout, hardware co-execution.
- **batch_size=1 focus.** Real-time interactive inference is always batch_size=1. This is memory-bandwidth-bound on Apple Silicon, making it a well-defined optimization target.
- **Roofline-driven.** By measuring bandwidth utilization, the agent knows whether it's close to the hardware limit or has room to improve — and where to focus effort.
- **Correctness-gated.** No speed improvement is accepted without passing the accuracy suite. This prevents optimizations that silently corrupt model outputs.

## License

MIT
