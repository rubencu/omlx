# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in Flash1017 runtime; importing this package does not load MLX."""

from .runtime import Controller, install

__all__ = ["Controller", "install"]
