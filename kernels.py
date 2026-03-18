"""
Custom Metal kernels for inference optimization.
"""
import mlx.core as mx
import numpy as np


# Fused SiLU + element-wise multiply kernel
# Used in SwiGLU: output = silu(gate) * up
SILU_MUL_KERNEL = """
uint elem = thread_position_in_grid.x;
T g = gate[elem];
T u = up[elem];
// SiLU: x * sigmoid(x)
T sigmoid_g = 1.0 / (1.0 + metal::exp(-g));
out[elem] = g * sigmoid_g * u;
"""


# Build the kernel once
_silu_mul_kernel = mx.fast.metal_kernel(
    name="silu_mul",
    input_names=["gate", "up"],
    output_names=["out"],
    source=SILU_MUL_KERNEL,
)


def fused_silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    """Fused SiLU activation + element-wise multiplication.

    Computes: silu(gate) * up = gate * sigmoid(gate) * up

    Args:
        gate: Gate tensor for SiLU
        up: Up projection tensor

    Returns:
        Element-wise product of silu(gate) and up
    """
    if gate.shape != up.shape:
        raise ValueError(f"Shape mismatch: gate={gate.shape}, up={up.shape}")

    outputs = _silu_mul_kernel(
        inputs=[gate, up],
        template=[("T", gate.dtype)],
        grid=(gate.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
    )
    return outputs[0]


# Fused RMSNorm + residual addition kernel
# Computes: output = x + rms_norm(hidden) * weight
RMSNORM_ADD_KERNEL = """
uint elem = thread_position_in_grid.x;
uint hidden_size = weight_shape[0];

// Compute RMS over the last dimension
uint row_start = (elem / hidden_size) * hidden_size;
float sum_sq = 0.0;
for (uint i = 0; i < hidden_size; i++) {
    float h = hidden[row_start + i];
    sum_sq += h * h;
}
float rms = metal::sqrt(sum_sq / float(hidden_size) + eps);

// Normalize and add residual
uint idx = elem % hidden_size;
float normalized = hidden[elem] / rms;
out[elem] = x[elem] + normalized * weight[idx];
"""


_rmsnorm_add_kernel = mx.fast.metal_kernel(
    name="rmsnorm_add",
    input_names=["hidden", "weight", "x"],
    output_names=["out"],
    source=RMSNORM_ADD_KERNEL,
)


def fused_rmsnorm_add(hidden: mx.array, weight: mx.array, residual: mx.array, eps: float = 1e-6) -> mx.array:
    """Fused RMSNorm + residual addition.

    Computes: output = residual + rms_norm(hidden) * weight

    This fuses the normalization and residual addition into a single kernel,
    reducing memory bandwidth.

    Args:
        hidden: Input hidden states
        weight: RMSNorm weight
        residual: Residual connection to add
        eps: Epsilon for numerical stability

    Returns:
        Normalized hidden states + residual
    """
    if hidden.shape != residual.shape:
        raise ValueError(f"Shape mismatch: hidden={hidden.shape}, residual={residual.shape}")

    B, L, D = hidden.shape
    outputs = _rmsnorm_add_kernel(
        inputs=[hidden.flatten(), weight, residual.flatten()],
        template=[("T", hidden.dtype)],
        grid=(B * L * D, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[hidden.shape],
        output_dtypes=[hidden.dtype],
        init_value=0.0,
    )
    return outputs[0].reshape(hidden.shape)


# Benchmark the kernels
def benchmark_silu_mul():
    """Benchmark fused SiLU + multiply vs separate ops."""
    import time

    size = 1024 * 1024 * 16  # 16M elements
    gate = mx.random.normal(shape=(size,)).astype(mx.float16)
    up = mx.random.normal(shape=(size,)).astype(mx.float16)

    # Warmup
    for _ in range(3):
        _ = mx.sigmoid(gate) * gate * up
        _ = fused_silu_mul(gate, up)
    mx.eval(gate, up)

    # Benchmark separate ops
    times_sep = []
    for _ in range(10):
        start = time.perf_counter()
        result_sep = mx.sigmoid(gate) * gate * up
        mx.eval(result_sep)
        times_sep.append(time.perf_counter() - start)

    # Benchmark fused kernel
    times_fused = []
    for _ in range(10):
        start = time.perf_counter()
        result_fused = fused_silu_mul(gate, up)
        mx.eval(result_fused)
        times_fused.append(time.perf_counter() - start)

    avg_sep = sum(times_sep) / len(times_sep) * 1000
    avg_fused = sum(times_fused) / len(times_fused) * 1000

    # Verify correctness
    mx.eval(result_sep, result_fused)
    close = mx.allclose(result_sep, result_fused, rtol=1e-2, atol=1e-2)

    print(f"SiLU + Multiply ({size/1e6:.1f}M elements):")
    print(f"  Separate: {avg_sep:.2f} ms")
    print(f"  Fused:    {avg_fused:.2f} ms")
    print(f"  Speedup:  {avg_sep/avg_fused:.2f}x")
    print(f"  Correct:  {close}")


if __name__ == "__main__":
    print("Benchmarking custom Metal kernels...\n")
    benchmark_silu_mul()
