# InferenceSpeed - Mac Mini Edition

Autonomous LLM inference optimization research. Goal: maximize tokens/sec for batch_size=1 using Qwen3.5-2B on Apple Silicon.

## Baseline Results (March 2026)

| Backend | Precision | tok/s | Latency | Memory |
|---------|-----------|-------|---------|--------|
| **MLX 4-bit** | 4bit | **67.4** | 741ms | 2.5GB |
| MLX 8-bit | 8bit | 40.3 | 1241ms | 0.9GB |
| MLX fp16 | fp16 | 22.7 | 2198ms | 3.7GB |
| Transformers | fp16 | ❌ | - | - |

**Best: MLX 4-bit at 67.4 tok/s (125% above baseline)**

## Known Issues

1. **Transformers + Qwen3.5**: Qwen3.5 uses a nested `text_config` structure that current transformers versions don't fully support. The model loads but generation fails with config attribute errors.

2. **vLLM**: Requires specific torch versions not compatible with current Mac setup. vLLM-Metal plugin is available but experimental.

3. **SGLang**: Apple Silicon support is in development (MLX backend planned for 2026 Q1).

## Running Benchmarks

```bash
python benchmark.py
```

## Things to Experiment With

- Different model sizes (Qwen3.5-0.5B, Qwen3.5-1.8B, Qwen3.5-3B)
- Custom quantization (3-bit, 6-bit)
- KV cache optimization
- Prompt caching
- Speculative decoding
- Batch size = 1 specific optimizations

## Goals

- Target: >60 tok/s ✓ (achieved with MLX 4-bit)
- Minimize latency (<500ms)
- Reduce memory footprint
