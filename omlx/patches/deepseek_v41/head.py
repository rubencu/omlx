# SPDX-License-Identifier: MIT
"""Vocabulary projection with BF16 storage and FP32 accumulation/output."""

from functools import cache

import mlx.core as mx

_SOURCE = r"""
    const uint row = thread_position_in_grid.x / KL;
    if (row >= N) return;
    const uint lane = thread_index_in_simdgroup % KL;
    const uint query = threadgroup_position_in_grid.y;
    float sum = 0;
    for (uint k = lane * 4; k < K; k += KL * 4) {
        const size_t offset = size_t(row) * K + k;
        const float4 a = float4(w[offset], w[offset + 1],
                                w[offset + 2], w[offset + 3]);
        const float4 b = float4(x[query * K + k], x[query * K + k + 1], x[query * K + k + 2], x[query * K + k + 3]);
        sum += dot(a, b);
    }
    for (uint offset = KL / 2; offset > 0; offset /= 2) {
        sum += simd_shuffle_down(sum, offset);
    }
    if (lane == 0) y[query * N + row] = sum;
"""

# The same per-row dot/reduction order as _SOURCE, sharing each weight load
# across the short verification block. Closely related BF16 rows 2..5 code
# predates this work in tacos8me/m5-ultra (4e4d5a81, head.py). Rows above five
# retain the stock GEMM fallback, whose reduction differs from this kernel.
_REUSE_SOURCE = r"""
    const uint row = thread_position_in_grid.x / KL;
    if (row >= N) return;
    const uint lane = thread_index_in_simdgroup % KL;
    float sums[M];
    for (uint q = 0; q < M; ++q) sums[q] = 0.0f;
    for (uint k = lane * 4; k < K; k += KL * 4) {
        const size_t offset = size_t(row) * K + k;
        const float4 a = float4(w[offset], w[offset + 1],
                               w[offset + 2], w[offset + 3]);
        for (uint q = 0; q < M; ++q) {
            const float4 b = float4(x[q * K + k], x[q * K + k + 1],
                                   x[q * K + k + 2], x[q * K + k + 3]);
            sums[q] += dot(a, b);
        }
    }
    for (uint offset = KL / 2; offset > 0; offset /= 2) {
        for (uint q = 0; q < M; ++q)
            sums[q] += simd_shuffle_down(sums[q], offset);
    }
    if (lane == 0) {
        for (uint q = 0; q < M; ++q) y[q * N + row] = sums[q];
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="v41_bf16_head_fp32_output",
        input_names=["x", "w"],
        output_names=["y"],
        source=_SOURCE,
    )


@cache
def _reuse_kernel():
    return mx.fast.metal_kernel(
        name="v41_bf16_weight_reuse_head_fp32_output",
        input_names=["x", "w"],
        output_names=["y"],
        source=_REUSE_SOURCE,
    )


def project_logits(x, weight, *, reuse_weights=False):
    """Keep the full logits contract for prefill and multi-token verification."""
    rows, width = weight.shape
    queries = x.size // width if width else 0
    if (
        reuse_weights
        and weight.dtype == mx.bfloat16
        and x.shape[-1] == width
        and 2 <= queries <= 5
        and width % 4 == 0
        and width >= 256
        and rows >= 4096
        and mx.default_device() != mx.cpu
    ):
        lanes = 16
        return _reuse_kernel()(
            inputs=[x.astype(mx.float32), weight],
            template=[("N", rows), ("K", width), ("KL", lanes), ("M", queries)],
            grid=(((rows * lanes + 63) // 64) * 64, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(*x.shape[:-1], rows)],
            output_dtypes=[mx.float32],
        )[0]
    if (
        weight.dtype != mx.bfloat16
        or x.shape[-1] != width
        or not 1 <= x.size // width <= 5
        or width % 4
        or width < 256
        or rows < 4096
        or mx.default_device() == mx.cpu
    ):
        return x.astype(mx.float32) @ weight.astype(mx.float32).T
    lanes = 16
    return _kernel()(
        inputs=[x.astype(mx.float32), weight],
        template=[("N", rows), ("K", width), ("KL", lanes)],
        grid=(((rows * lanes + 63) // 64) * 64, x.size // width, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*x.shape[:-1], rows)],
        output_dtypes=[mx.float32],
    )[0]
