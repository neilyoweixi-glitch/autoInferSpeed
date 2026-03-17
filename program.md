# autoInferSpeed

This is an experiment to have the LLM autonomously maximize inference speed for **Qwen3.5-2B 8-bit** on Apple Silicon, while proving that optimizations produce correct outputs and approach the hardware's theoretical limits.

## Fixed constraints

- **Model**: Qwen3.5-2B 8-bit quantized (`mlx-community/Qwen3.5-2B-8bit`). Do not switch models. All optimizations must target this exact model.
- **Backend**: MLX only. Apple Silicon native. No Transformers, vLLM, SGLang, or Ollama.
- **Correctness**: Every optimization must pass the accuracy verification suite. Speed without correctness is worthless.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar17`). The branch `autoinfer/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoinfer/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `inference.py` — the file you modify. Backend implementations, benchmark harness, accuracy verification, and configuration.
   - `pyproject.toml` — dependencies. Do not add new packages.
4. **Verify dependencies**: Run `uv sync` if needed. Check that `uv run python -c "import mlx.core as mx; print(mx.metal.is_available())"` returns True.
5. **Establish the baseline**: Run `inference.py` unmodified to record the baseline. This first run serves two purposes:
   - Records baseline tok/s for Qwen3.5-2B 8-bit.
   - Captures **baseline reference outputs** — the exact token sequences the unmodified model produces for each test prompt. All future optimizations are compared against these reference outputs for correctness.
6. **Initialize results.tsv**: Create `results.tsv` with the header row. The baseline will be the first data row.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Baseline accuracy verification

The benchmark must include a correctness verification suite that runs **after every speed benchmark**. This is non-negotiable — an optimization that breaks output quality is rejected regardless of speed gains.

### Reference model

The unmodified Qwen3.5-2B 8-bit model (loaded via `mlx-community/Qwen3.5-2B-8bit` with default settings, `temperature=0`, greedy decoding) is the **ground truth**. The first run captures its outputs as the reference.

### Test prompts

Use a fixed set of test prompts that exercise different capabilities:

```python
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
```

### Verification method

For each test prompt, generate with `temperature=0` (greedy decoding), `max_tokens=50`:

1. **Token-level exact match**: Compare the generated token IDs against the baseline reference. Report the fraction of tokens that match exactly (e.g. `48/50 = 96%`).
2. **First-divergence position**: Report at which token position the output first diverges from the reference (if at all). Early divergence (position < 5) is a red flag.
3. **Semantic coherence check**: For prompts with verifiable answers (arithmetic, factual), check if the core answer is correct regardless of exact token match. "512" vs " 512" is fine; "512" vs "513" is a failure.

### Pass/fail criteria

- **PASS**: All prompts produce >=90% token match with reference AND all verifiable answers are semantically correct.
- **WARN**: Token match is 80-90% but verifiable answers are correct. The optimization is acceptable but should be noted.
- **FAIL**: Any verifiable answer is wrong OR token match drops below 80%. The optimization is rejected — discard and revert.

### Output format

The accuracy check should print:

```
ACCURACY VERIFICATION (vs baseline)
  "The capital of France is"     tokens: 50/50 (100%)  first_div: none   semantic: PASS
  "What is 127 + 385?..."       tokens: 48/50 (96%)   first_div: 42     semantic: PASS
  "Write a Python function..."   tokens: 50/50 (100%)  first_div: none   semantic: N/A
  "If all roses are flowers..."  tokens: 45/50 (90%)   first_div: 38     semantic: PASS
  "Explain photosynthesis..."    tokens: 47/50 (94%)   first_div: 35     semantic: N/A
OVERALL: PASS (avg token match: 96.0%)
```

## Experimentation scope

Each experiment runs on Apple Silicon. You launch it simply as: `uv run inference.py`.

**What you CAN do:**
- Modify `inference.py` — this is the only file you edit. You may also create new `.py` files if you need custom Metal kernels or helper modules, but `inference.py` remains the entry point.

**What you CANNOT do:**
- Install new packages or add dependencies beyond what's in `pyproject.toml`. Use `uv sync` to install, never `pip`.
- Change the model (must remain Qwen3.5-2B 8-bit).
- Skip or weaken the accuracy verification.
- Modify any other files in the repo (except `results.tsv` for logging).

**The goal: maximize prefill and decode performance for Qwen3.5-2B 8-bit, while maintaining output correctness, and report how close you are to the hardware roofline for each stage.**

### Prefill vs decode: two separate optimization targets

Inference has two distinct stages with different computational profiles. The benchmark MUST measure and optimize them separately.

**Prefill stage** (processing the input prompt):
- All input tokens are processed in parallel in a single forward pass.
- This is **compute-bound**: arithmetic intensity = `2 * params * seq_len / params_bytes` = `2 * seq_len` (for 8-bit weights). Even at seq_len=64, this is well above the ridge point on Apple Silicon.
- The metric is **prefill tokens/sec** (input tokens processed per second) or equivalently **time-to-first-token (TTFT)**.
- Must be benchmarked at multiple sequence lengths to understand scaling:
  - Short: 32, 64 tokens (chatbot greetings, quick queries)
  - Medium: 256, 512 tokens (typical conversations, code snippets)
  - Long: 1024, 2048 tokens (long documents, RAG contexts)
- The roofline for prefill is FLOP/s-based, not bandwidth-based. Report MFU (achieved FLOP/s vs peak GPU TFLOP/s).
- Optimizations: fused kernels, tiling for Metal SIMD width, `mx.compile()`, larger `prefill_step_size`, chunked prefill to balance latency.

**Decode stage** (generating output tokens autoregressively):
- Each token step reads the full model weights + KV cache. One token out per step.
- At **batch_size=1**: purely **memory-bandwidth-bound**. The metric is **decode tokens/sec**.
- At **higher batch sizes**: arithmetic intensity increases. The critical batch size where decode shifts from memory-bound to compute-bound is: `batch_crit = peak_bandwidth_bytes / (2 * params_bytes * peak_flops)`. For M4 with 8-bit Qwen3.5-2B: `batch_crit ≈ 120 GB/s / (2 * 2.47 GB * 5.4 TFLOP/s) ≈ ~4-5`. Above this, you become compute-bound.
- Benchmark decode at multiple batch sizes: **1, 2, 4, 8** (if memory permits).
- For batch_size > 1, report both **throughput** (total tok/s across all sequences) and **per-request latency** (ms per token per sequence). There is a tradeoff: higher batch = more throughput but worse per-request latency.
- Latency constraint modes:
  - **Interactive** (batch=1): minimize per-token latency. Target < 25ms/token.
  - **Throughput** (batch=4-8): maximize total tok/s. Accept up to 100ms/token per request.
- Optimizations: KV cache quantization, continuous batching, memory-aligned weight layout, weight prefetching.

**Benchmark output format** (replaces the current unified format):

```
PREFILL BENCHMARK
  seq_len=32:    TTFT=12ms   prefill_tok/s=2667   MFU=74%
  seq_len=256:   TTFT=48ms   prefill_tok/s=5333   MFU=89%
  seq_len=1024:  TTFT=185ms  prefill_tok/s=5535   MFU=92%
  seq_len=2048:  TTFT=390ms  prefill_tok/s=5252   MFU=88%

DECODE BENCHMARK
  batch=1:  tok/s=40.3   ms/tok=24.8   bandwidth_util=99%  bottleneck=MEMORY_BOUND
  batch=2:  tok/s=72.1   ms/tok=27.7   bandwidth_util=89%  bottleneck=MEMORY_BOUND
  batch=4:  tok/s=125.0  ms/tok=32.0   bandwidth_util=77%  bottleneck=COMPUTE_BOUND
  batch=8:  tok/s=180.2  ms/tok=44.4   bandwidth_util=56%  bottleneck=COMPUTE_BOUND
```

### Hardware roofline analysis

Every benchmark run should report how close the achieved throughput is to the theoretical maximum. This tells us whether we're leaving performance on the table or hitting hardware limits.

**What to measure and report:**

1. **Memory bandwidth utilization**: Apple Silicon is memory-bandwidth-bound for batch_size=1 inference. Calculate:
   - Model size in bytes (e.g. ~2.5 GB for 2B params at 8-bit)
   - Theoretical peak memory bandwidth of the chip (e.g. M1: 68.25 GB/s, M2: 100 GB/s, M3: 100 GB/s, M4: 120 GB/s)
   - Roofline tok/s = `peak_bandwidth / bytes_per_token_step` where `bytes_per_token_step ≈ model_size_bytes` (each token generation reads the full model weights once)
   - Achieved fraction = `actual_tok_s / roofline_tok_s`
   - Report: `bandwidth_util: 67% (actual 40.3 tok/s vs roofline 60.1 tok/s on M2)`

2. **GPU vs CPU utilization**: Report whether the GPU and CPU are both being utilized or if one is idle. Use `Activity Monitor` metrics or `powermetrics` if available. At minimum, measure wall-clock time breakdown:
   - Time spent in GPU compute (matrix multiplications, attention)
   - Time spent in CPU overhead (tokenization, scheduling, memory copies)
   - Time spent idle/waiting

3. **Bottleneck identification**: Based on the above, identify the current bottleneck:
   - `MEMORY_BOUND`: bandwidth utilization > 70%, compute utilization low → limited by how fast we can read weights
   - `COMPUTE_BOUND`: GPU compute utilization high, bandwidth available → limited by arithmetic throughput
   - `OVERHEAD_BOUND`: neither GPU nor memory saturated → limited by framework overhead, kernel launch latency, CPU-side work
   - `CPU_BOUND`: CPU is the bottleneck (tokenization, pre/post-processing)

**Roofline output format** (separate for prefill and decode):

```
ROOFLINE ANALYSIS
  chip: Apple M4 (10-core GPU, 120 GB/s bandwidth, 5.4 TFLOPS)
  model_size: 2.47 GB (8-bit quantized)

  PREFILL ROOFLINE (compute-bound regime)
    peak_flops: 5.4 TFLOPS
    seq_len=256: achieved 4.8 TFLOPS (MFU=89%)
    seq_len=1024: achieved 5.0 TFLOPS (MFU=92%)
    bottleneck: COMPUTE_BOUND

  DECODE ROOFLINE (memory-bandwidth-bound at batch=1)
    roofline_tok_s: 48.6 tok/s (120 GB/s / 2.47 GB)
    batch=1: actual 40.3 tok/s, bandwidth_util=83%
    batch=4: actual 125.0 tok/s, shifted to COMPUTE_BOUND
    batch_critical: ~5 (crossover point)
```

### Ideas to try — kernel and hardware level

The following are ordered roughly from easiest to hardest. Focus on what moves the needle given the current bottleneck.

**Reducing framework overhead (if OVERHEAD_BOUND):**
- Minimize Python-side work between token generations
- Pre-allocate KV cache to avoid repeated memory allocation
- Use MLX's `mx.compile()` or `mx.disable_compile()` to find optimal JIT behavior
- Reduce tokenizer overhead (cache tokenized prompts, avoid re-encoding)
- Eliminate unnecessary memory copies between CPU and GPU

**Memory bandwidth optimizations (if MEMORY_BOUND):**
- KV cache quantization (quantize cached keys/values to 4-bit while keeping weights at 8-bit)
- Grouped-query attention exploitation (if model supports it — fewer KV heads = less memory traffic)
- Paged attention / block-sparse attention to reduce cache reads
- Prefetch optimization — overlap weight loading with computation
- Ensure memory alignment for optimal bus utilization

**Compute optimizations (if COMPUTE_BOUND):**
- Custom Metal kernels for fused operations (e.g. fused QKV projection, fused gate+up projection)
- Custom quantized matmul kernels optimized for 8-bit on Metal
- Fused softmax + attention score computation
- Fused RMSNorm + residual addition
- SIMD-width-aware tiling for Metal GPU cores

**CPU+GPU co-execution:**
- Offload embedding lookup to CPU while GPU runs attention
- Run tokenizer decode on CPU in parallel with next-token GPU inference
- Pipeline CPU pre-processing of next prompt chunk while GPU generates current token
- Use Accelerate framework for CPU-side matrix ops on efficiency cores
- Parallelize KV cache management on CPU while GPU computes

**Speculative and algorithmic:**
- Self-speculative decoding (use early layers to predict, full model to verify)
- Prompt-lookup decoding for repetitive outputs
- Dynamic early exit (skip later layers for high-confidence tokens)
- Continuous batching of KV cache updates

**MLX-specific:**
- `mx.metal.start_capture()` / `mx.metal.stop_capture()` for Metal profiling
- Experiment with `mx.stream()` for async execution
- Test `mx.compile()` with different optimization levels
- Investigate MLX's internal kernel selection — some shapes may trigger suboptimal kernels

## Output format

The benchmark script should print a combined report:

```
============================================================
BENCHMARK: Qwen3.5-2B 8-bit
============================================================

SPEED
  tokens_per_sec: 40.3
  time_to_first_token_ms: 45.2
  total_latency_ms: 1241
  tokens_generated: 50
  memory_mb: 1900

ROOFLINE ANALYSIS
  chip: Apple M2 (8-core GPU, 100 GB/s bandwidth)
  model_size: 2.47 GB
  roofline_tok_s: 40.5
  actual_tok_s: 40.3
  bandwidth_util: 99.5%
  bottleneck: MEMORY_BOUND

ACCURACY VERIFICATION (vs baseline)
  prompt_1: 50/50 (100%)  semantic: PASS
  prompt_2: 48/50 (96%)   semantic: PASS
  prompt_3: 50/50 (100%)  semantic: N/A
  prompt_4: 45/50 (90%)   semantic: PASS
  prompt_5: 47/50 (94%)   semantic: N/A
  OVERALL: PASS (96.0%)
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 12 columns:

```
commit	stage	seq_len_or_batch	tok_sec	ttft_ms	ms_per_tok	memory_mb	mfu_or_bw_util	bottleneck	accuracy	accuracy_pct	status	description
```

1. git commit hash (short, 7 chars)
2. stage: `prefill` or `decode`
3. seq_len (for prefill) or batch_size (for decode)
4. tokens_per_sec (prefill: input tok/s, decode: output tok/s)
5. TTFT in ms (prefill only, `-` for decode)
6. ms per token (decode: per-request latency, `-` for prefill)
7. memory in MB
8. MFU fraction (prefill) or bandwidth_util fraction (decode)
9. bottleneck type: `MEMORY_BOUND`, `COMPUTE_BOUND`, `OVERHEAD_BOUND`, `CPU_BOUND`
10. accuracy result: `PASS`, `WARN`, `FAIL`
11. average token match percentage (e.g. 96.0)
12. status: `keep`, `discard`, or `crash`
13. short text description of what this experiment tried

Example:

```
commit	stage	seq_or_batch	tok_sec	ttft_ms	ms_per_tok	memory_mb	mfu_or_bw	bottleneck	accuracy	accuracy_pct	status	description
a1b2c3d	prefill	256	5333	48	-	1900	0.89	COMPUTE_BOUND	PASS	100.0	keep	baseline prefill
a1b2c3d	prefill	1024	5535	185	-	1900	0.92	COMPUTE_BOUND	PASS	100.0	keep	baseline prefill
a1b2c3d	decode	1	40.3	-	24.8	1900	0.83	MEMORY_BOUND	PASS	100.0	keep	baseline decode batch=1
a1b2c3d	decode	4	125.0	-	32.0	2100	0.77	COMPUTE_BOUND	PASS	100.0	keep	baseline decode batch=4
b2c3d4e	decode	1	43.5	-	23.0	1950	0.90	MEMORY_BOUND	PASS	98.0	keep	fused RMSNorm + KV cache quant
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoinfer/mar17`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Check the roofline analysis from the last run. Look at **both** prefill and decode results to identify the current bottleneck for each stage.
3. Choose an optimization and which stage it targets. Some optimizations (e.g. fused kernels) help both; others are stage-specific (e.g. KV cache quantization only helps decode, chunked prefill only helps prefill).
4. Modify `inference.py` (and optionally create kernel files) with the optimization.
5. `git commit` the change.
6. Run the experiment: `uv run inference.py > run.log 2>&1` (redirect everything — do NOT let output flood your context).
7. Read out the results: `grep "prefill\|decode\|tok_sec\|MFU\|bandwidth_util\|bottleneck\|OVERALL\|Error" run.log` or `tail -n 50 run.log`.
8. **Check accuracy first**: If accuracy is FAIL, the optimization is immediately rejected — discard and revert.
9. If the output shows errors or no results, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
10. Record the results in the TSV — one row per (stage, seq_len/batch) combination. (NOTE: do not commit results.tsv, leave it untracked by git).
11. If the targeted metric improved AND accuracy is PASS or WARN, you "advance" the branch, keeping the git commit.
12. If no improvement, or accuracy is FAIL, you `git reset` back to where you started.

**Bottleneck-driven optimization**: After each run, the roofline analysis tells you where the bottleneck is. Use this to guide your next experiment:
- If `OVERHEAD_BOUND` → reduce Python overhead, improve kernel launch, pre-allocate buffers
- If `MEMORY_BOUND` → try KV cache quantization, weight prefetching, reduce memory traffic
- If `COMPUTE_BOUND` → write custom Metal kernels, fuse operations, optimize tiling
- If `CPU_BOUND` → offload work to GPU, parallelize CPU tasks, reduce tokenizer overhead

**Crashes**: If a run crashes (OOM, import error, etc.), use your judgment: If it's something dumb and easy to fix, fix and re-run. If fundamentally broken, skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the roofline data, profile the bottleneck, try combining approaches, write custom kernels. The loop runs until the human interrupts you, period.

**Convergence**:
- If **decode** bandwidth utilization exceeds 95% at batch=1, you are near the memory-bandwidth limit. Shift to: (a) higher batch sizes to move into compute-bound territory, (b) prefill optimization, or (c) CPU+GPU co-execution.
- If **prefill** MFU exceeds 90%, you are near the compute limit. Shift to: (a) decode optimization, (b) chunked prefill for latency, or (c) explore whether longer sequences reveal new bottlenecks.
- If both stages are converged, focus on combined end-to-end latency for realistic workloads (e.g. 256-token prompt + 50-token generation). But do not stop — keep trying creative approaches.
