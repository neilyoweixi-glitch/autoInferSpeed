# autoInferSpeed

Autonomous inference optimization for **Qwen3.5-2B 8-bit** on Apple Silicon via **MLX**.

## Quick reference

- **Agent instructions**: See `program.md` for the full autonomous experiment protocol.
- **Human overview**: See `README.md` for project context and quick start.
- **Agent modifies**: `inference.py` only (may also create kernel `.py` files).
- **Backend**: MLX only. No Transformers, vLLM, SGLang, or Ollama.
- **Model**: Qwen3.5-2B 8-bit (`mlx-community/Qwen3.5-2B-8bit`) — fixed, do not change.
- **Metric**: `tokens_per_sec` — higher is better.
- **Constraint**: `batch_size=1` — always fixed.
- **Correctness**: Every run must pass accuracy verification vs baseline reference outputs.
- **Roofline**: Every run reports bandwidth utilization and bottleneck type.
- **Results log**: `results.tsv` (untracked by git).

## Baseline (March 2026)

MLX 8-bit: ~40.3 tok/s, TTFT ~1241ms, ~0.9 GB memory.

## Files

```
inference.py    — MLX inference harness + accuracy + roofline (AGENT MODIFIES THIS)
program.md      — autonomous agent protocol (HUMAN MODIFIES THIS)
results.tsv     — experiment log (untracked)
pyproject.toml  — dependencies
README.md       — project overview
CLAUDE.md       — this file
```

## Running

```bash
uv run inference.py
```
