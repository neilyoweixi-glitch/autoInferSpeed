# InferenceSpeed - Mac Mini Edition

Autonomous LLM inference optimization research. Goal: maximize tokens/sec for batch_size=1 using Qwen models on Apple Silicon.

## How It Works

1. **Agent modifies** `benchmark.py` - backends, quantization, model selection
2. **Runs benchmark** - tests different configurations
3. **Metric: tokens_per_sec** - higher is better
4. **Logs to** `results.tsv`

## Files

- `benchmark.py` - Benchmark runner (AGENT MODIFIES THIS)
- `results.tsv` - Experiment log
- `CLAUDE.md` - This file

## Running Benchmarks

```bash
python benchmark.py
```

## Baselines (from research)

| Backend | Expected tok/s | Notes |
|---------|----------------|-------|
| Transformers MPS | 20-40 | Baseline |
| MLX | 40-80 | 2x improvement |
| MLX 4-bit | 60-100 | Quantization speedup |
| llama.cpp | 40-70 | GGUF format |

Source: [MLX benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1rs059a/mlx_is_not_faster_i_benchmarked_mlx_vs_llamacpp/), [Qwen on Mac](https://dev.to/thefalkedguy/installing-qwen-35-on-apple-silicon-using-mlx-for-2x-performance-37ma)

## Things to Experiment With

- Model size (Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B)
- Precision (fp32, fp16, bf16)
- Quantization (4-bit, 8-bit, GGUF)
- Backend (transformers, MLX, llama.cpp)
- KV cache settings
- Batch size effects (though we fix at 1)
- Prompt length effects

## Goals

- Target: >60 tok/s on Mac M1/M2
- Minimize TTFT (time to first token)
- Minimize memory usage
