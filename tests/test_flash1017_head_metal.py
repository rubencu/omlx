# SPDX-License-Identifier: Apache-2.0
"""Real Metal head checks, including the production hidden width and tail rows.

Rows 2-5 compare with the normal batched head. Row 6 retains the stock GEMM
fallback, and is compared directly with that same fallback.
This is a projection test, not an answer-quality evaluation.
"""

import mlx.core as mx
import pytest

from omlx.patches.deepseek_v41.head import project_logits


@pytest.mark.parametrize("width", [256, 5120])
@pytest.mark.parametrize("rows", [2, 5, 6])
def test_reuse_head_matches_original_row_arithmetic(rows, width):
    if not mx.metal.is_available():
        pytest.skip("Apple Metal is required")
    with mx.stream(mx.gpu):
        mx.random.seed(17 + rows + width)
        weights = mx.random.normal((4099, width)).astype(mx.bfloat16)
        inputs = mx.random.normal((1, rows, width)).astype(mx.bfloat16)
        actual = project_logits(inputs, weights, reuse_weights=True)
        expected = project_logits(inputs, weights)
        mx.eval(actual, expected)
        assert actual.dtype == mx.float32
        assert actual.shape == (1, rows, 4099)
        assert mx.array_equal(actual, expected).item()
