# SPDX-License-Identifier: Apache-2.0
"""Exercise the production prefetch admission and cleanup with a tiny reader."""

import numpy as np
import pytest

from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding, EngramPrefetch


class Reader(DiskEngramEmbedding):
    def __init__(self):
        self._resident = None
        self._prefetched = None

    def selected_bytes(self, count):
        return 32 * 2**20

    def _read_rows(self, ids):
        return ids.copy()


def test_prefetch_budget_is_local_and_default_preserved():
    original, expanded = EngramPrefetch(), EngramPrefetch(max_bytes=64 * 2**20)
    a, b = Reader(), Reader()
    try:
        original.submit(a, np.array([7]))
        expanded.submit(b, np.array([7]))
        assert original.budget_bytes == 16 * 2**20
        assert original._pending is None and a._prefetched is None
        assert expanded._pending is not None
        assert expanded._pending[1].result().tolist() == [7]
        expanded.drain()
        assert b._prefetched is None and expanded._pending is None
    finally:
        original.close()
        expanded.close()


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, 64 * 2**20 + 1])
def test_invalid_budget_rejected_before_executor_creation(budget):
    with pytest.raises(ValueError, match="budget"):
        EngramPrefetch(max_bytes=budget)
