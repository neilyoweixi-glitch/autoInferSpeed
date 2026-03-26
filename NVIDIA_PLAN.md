# NVIDIA Inference Optimization Plan

## Server: 192.168.1.3

## Objective
Port autoInferSpeed to NVIDIA GPUs and explore performance optimization opportunities specific to CUDA architecture.

---

## Phase 0: Setup & Baseline (Day 1)

### 0.1 Server Access
```bash
# User needs to provide SSH credentials
ssh user@192.168.1.3
```

### 0.2 Environment Check
```bash
# GPU info
nvidia-smi

# Expected output needs:
# - GPU model (e.g., RTX 4090, A100, H100)
# - Memory size
# - CUDA version
# - Driver version
```

### 0.3 Dependencies
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo
git clone https://github.com/neilyoweixi-glitch/autoInferSpeed
cd autoInferSpeed
git checkout autoinfer/mar19
```

### 0.4 Backend Decision

| Option | Pros | Cons |
|--------|------|------|
| **PyTorch + CUDA** | Standard, well-supported | Higher overhead |
| **vLLM** | Production-grade, PagedAttention | Complex setup |
| **TensorRT-LLM** | Best performance | Requires conversion |
| **SGLang** | Fast, RadixAttention | Newer, less tested |

**Recommendation**: Start with PyTorch + CUDA for baseline, then try vLLM.

---

## Phase 1: Baseline Measurements (Day 1-2)

### 1.1 Port inference.py to PyTorch
Create `inference_cuda.py`:
- Use `transformers` + `bitsandbytes` for 8-bit loading
- Or use `AutoModelForCausalLM.from_pretrained(..., load_in_8bit=True)`
- Benchmark same metrics: prefill MFU, decode roofline

### 1.2 Baseline Metrics to Capture
| Metric | How to Measure |
|--------|----------------|
| Prefill tok/s | tokens / prefill_time |
| Decode tok/s | tokens / decode_time |
| Memory bandwidth util | actual_bw / peak_bw |
| GPU utilization | nvidia-smi dmon |
| Memory usage | torch.cuda.memory_allocated() |

### 1.3 Hardware Roofline for Common GPUs

| GPU | Memory BW | TFLOPS (FP16) | Roofline (2.47GB model) |
|-----|-----------|---------------|-------------------------|
| RTX 4090 | 1008 GB/s | 165 | 408 tok/s |
| RTX 4080 | 717 GB/s | 97 | 290 tok/s |
| A100 80GB | 2039 GB/s | 624 | 825 tok/s |
| A10 | 600 GB/s | 125 | 243 tok/s |
| H100 | 3352 GB/s | 989 | 1357 tok/s |

---

## Phase 2: NVIDIA-Specific Optimizations (Day 2-4)

### 2.1 Flash Attention 2
```python
# Install
pip install flash-attn --no-build-isolation

# Use in model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
```
**Expected gain**: 2-4x on long sequences

### 2.2 PagedAttention (vLLM)
```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-1.5B",
    quantization="awq",  # or "gptq"
    dtype="float16"
)
```
**Expected gain**: 2-3x batch throughput

### 2.3 CUDA Graphs
```python
# For fixed-shape decode
torch.cuda.make_graphed_callables(model, (sample_input,))
```
**Expected gain**: 10-20% reduce kernel launch overhead

### 2.4 Quantization Options

| Method | Bits | Speed | Quality |
|--------|------|-------|---------|
| FP16 | 16 | Baseline | 100% |
| INT8 (dynamic) | 8 | +20% | 99% |
| INT4 (GPTQ/AWQ) | 4 | +40% | 97-99% |
| FP8 (H100) | 8 | +50% | 99% |

### 2.5 KV Cache Optimization
```python
# Use cache in generate
outputs = model.generate(
    input_ids,
    use_cache=True,
    pad_token_id=tokenizer.eos_token_id
)
```

### 2.6 Continuous Batching (vLLM/SGLang)
```python
# vLLM handles this automatically
outputs = llm.generate(prompts, use_tqdm=True)
```

---

## Phase 3: Comparison & Analysis (Day 4-5)

### 3.1 Metrics Comparison Template

| Metric | Apple M4 (MLX) | NVIDIA GPU (PyTorch) | NVIDIA GPU (vLLM) |
|--------|----------------|---------------------|-------------------|
| Decode roofline % | 99.6% | ? | ? |
| 16K prefill MFU | 78% | ? | ? |
| Peak MFU | 85% | ? | ? |
| Max batch size | 1 | ? | ? |
| Latency (20ms constraint) | Exceeded | ? | ? |

### 3.2 Analysis Questions
1. Does separate GPU memory help or hurt for long sequences?
2. How does Flash Attention compare to MLX attention?
3. What's the optimal batch size for each GPU?
4. How does quantization affect NVIDIA vs Apple Silicon?

---

## Phase 4: Custom CUDA Kernels (Optional, Day 5+)

If baseline optimizations aren't sufficient:

### 4.1 Triton Kernels
```python
import triton
import triton.language as tl

@triton.jit
def fused_silu_mul_kernel(...):
    # Custom fused kernel
    pass
```

### 4.2 Custom Attention Kernel
- Fused QKV projection
- Fused softmax + scaling
- Optimized for specific batch/sequence sizes

---

## File Structure (Proposed)

```
autoInferSpeed/
├── inference.py          # MLX (Apple Silicon)
├── inference_cuda.py     # PyTorch (NVIDIA) - NEW
├── inference_vllm.py     # vLLM (NVIDIA) - NEW
├── kernels/
│   ├── mlx_kernels.py    # MLX Metal kernels
│   └── cuda_kernels.py   # CUDA/Triton kernels - NEW
├── results/
│   ├── m4_results.tsv    # Apple M4 results
│   └── nvidia_results.tsv # NVIDIA results - NEW
└── NVIDIA_PLAN.md        # This file - NEW
```

---

## Quick Start Commands

```bash
# SSH to server
ssh user@192.168.1.3

# Setup
git clone https://github.com/neilyoweixi-glitch/autoInferSpeed
cd autoInferSpeed
uv sync

# Run baseline
uv run inference_cuda.py

# Run with vLLM
pip install vllm
uv run inference_vllm.py
```

---

## Questions to Answer

1. **Memory architecture impact**: How does separate GPU memory affect long-sequence prefill?
2. **Batch efficiency**: Can we achieve >100 tok/s with batch decoding?
3. **Flash Attention benefit**: Does FA2 help at 16K on NVIDIA?
4. **Quantization tradeoffs**: INT4 vs INT8 vs FP16 on NVIDIA?
5. **Cost efficiency**: tok/s per dollar, tok/s per watt?

---

## Success Criteria

| Metric | Target | Stretch |
|--------|--------|---------|
| Decode roofline | >90% | >95% |
| 16K prefill MFU | >80% | >85% |
| Batch throughput | >200 tok/s (batch=8) | >500 tok/s |
| Latency (20ms) | Met with batch≥2 | Met with batch≥4 |

---

## Next Steps

1. [ ] Provide SSH credentials for 192.168.1.3
2. [ ] Run `nvidia-smi` to get GPU specs
3. [ ] Install dependencies
4. [ ] Run baseline PyTorch inference
5. [ ] Try Flash Attention 2
6. [ ] Try vLLM for batching
7. [ ] Document results
