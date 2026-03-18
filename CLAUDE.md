# autoInferSpeed

Autonomous inference optimization for **Qwen3.5-2B 8-bit** on Apple Silicon via **MLX**.

## Quick reference

- **Agent instructions**: See `program.md` for the full autonomous experiment protocol.
- **Human overview**: See `README.md` for project context and quick start.
- **Agent modifies**: `inference.py` only (may also create kernel `.py` or `.metal` files).
- **Agent MUST NOT modify**: `prepare.py` — test data, accuracy verification, data classes.
- **Backend**: MLX only. No Transformers, vLLM, SGLang, or Ollama.
- **Model**: Qwen3.5-2B 8-bit (`mlx-community/Qwen3.5-2B-8bit`) — fixed, do not change.
- **Forbidden**: Speculative decoding, early exit, layer skipping, algorithmic shortcuts.
- **Metrics**: prefill tok/s + MFU (compute-bound) and decode tok/s + bandwidth_util (memory-bound).
- **Correctness**: Every run must pass accuracy verification vs baseline reference outputs.
- **Roofline**: Every run reports MFU (prefill) and bandwidth utilization (decode).
- **Results log**: `results.tsv` (untracked by git).

## Baseline (March 2026)

- **Prefill**: MFU ~85% at seq_len=2048, ~1150 prefill tok/s
- **Decode**: ~49 tok/s at 20.4 ms/tok (batch=1, context=256)
- **Accuracy**: 100%

## Files

```
inference.py    — benchmark harness + optimization flags (AGENT MODIFIES THIS)
prepare.py      — test data, accuracy, hardware detection (DO NOT MODIFY)
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
