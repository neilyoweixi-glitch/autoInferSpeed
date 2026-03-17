# autoInferSpeed

Autonomous inference optimization for **Qwen3.5-2B 8-bit** on Apple Silicon.

## Quick reference

- **Agent instructions**: See `program.md` for the full autonomous experiment protocol.
- **Human overview**: See `README.md` for project context and quick start.
- **Agent modifies**: `benchmark.py` only (may also create kernel `.py` files).
- **Model**: Qwen3.5-2B 8-bit — fixed, do not change.
- **Metric**: `tokens_per_sec` — higher is better.
- **Constraint**: `batch_size=1` — always fixed.
- **Correctness**: Every run must pass accuracy verification vs baseline reference outputs.
- **Roofline**: Every run reports bandwidth utilization and bottleneck type.
- **Results log**: `results.tsv` (untracked by git).

## Baseline Results (March 2026)

| Backend | Precision | tok/s | Latency | Memory |
|---------|-----------|-------|---------|--------|
| **MLX 4-bit** | 4bit | **67.4** | 741ms | 2.5GB |
| MLX 8-bit | 8bit | 40.3 | 1241ms | 0.9GB |
| MLX fp16 | fp16 | 22.7 | 2198ms | 3.7GB |
| Transformers | fp16 | 13.7 | 3653ms | - |

**Best: MLX 4-bit at 67.4 tok/s (125% above baseline)**

## Files

```
benchmark.py    — inference harness + accuracy + roofline (AGENT MODIFIES THIS)
program.md      — autonomous agent protocol (HUMAN MODIFIES THIS)
results.tsv     — experiment log (untracked)
pyproject.toml  — dependencies
README.md       — project overview
CLAUDE.md       — this file
```

## Running

```bash
uv run benchmark.py
```

## Known Issues

1. **Transformers + Qwen3.5**: Qwen3.5 uses a nested `text_config` structure that current transformers versions don't fully support. The model loads but generation fails with config attribute errors.

2. **vLLM**: Requires specific torch versions not compatible with current Mac setup. vLLM-Metal plugin is available but experimental.

3. **SGLang**: Apple Silicon support is in development (MLX backend planned for 2026 Q1).

## Current baseline

Qwen3.5-2B 8-bit via MLX: ~40 tok/s, ~1900 MB memory.
