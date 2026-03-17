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

## Current baseline

Qwen3.5-2B 8-bit via MLX: ~40 tok/s, ~1900 MB memory.
