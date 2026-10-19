# SPDX-License-Identifier: Apache-2.0
"""Explicit, portable configuration for the frozen Flash-1017 experiment."""

import copy
import hashlib
from pathlib import Path

MODEL_REPOSITORY = "mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp"
MODEL_REVISION = "78456115d301eecf9fadcdcc81c0cf6f8b2c8eab"
PROFILE_VERSION = "flash1017-v1"
MAX_CONTEXT = 131072
PREFILL_STEP = 8192
METADATA_SHA256 = {
    "config.json": "74259bebe816bb248eb075a9c285c64eb705d9ced63777ef5b0a9263b06793c6",
    "model.safetensors.index.json": "cf5d05c50ce04258ece0b1d47f9abb9f8ff0eeb62665088042b3088a866714d0",
    "tokenizer.json": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
    "tokenizer_config.json": "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
}


def validate_checkpoint(path):
    """Verify pinned metadata before allocation; this does not hash weight shards."""
    path = Path(path).expanduser().resolve()
    for name, expected in METADATA_SHA256.items():
        file = path / name
        if (
            not file.is_file()
            or hashlib.sha256(file.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(
                f"Flash-1017 requires {MODEL_REPOSITORY} at {MODEL_REVISION}; "
                f"checkpoint metadata differs: {name}"
            )
    return path


def check_load_headroom():
    """Keep the historical per-shard floor without a global loader patch."""
    import psutil

    if psutil.virtual_memory().available < 12 * 2**30:
        raise MemoryError("Flash-1017 loading requires at least 12 GiB system headroom")


def prepare_memory():
    """Set this process's MLX guidelines, retaining the measured OS reserve."""
    import mlx.core as mx
    import psutil

    device = mx.device_info()
    reserve = 32 * 2**30
    limit = min(
        device["max_recommended_working_set_size"], device["memory_size"] - reserve
    )
    if device["memory_size"] < 256 * 2**30 or limit < 210 * 2**30:
        raise MemoryError(
            "Flash-1017 requires a Mac with at least 256 GiB unified memory"
        )
    if psutil.virtual_memory().available < 210 * 2**30:
        raise MemoryError(
            "Flash-1017 needs at least 210 GiB available before loading; unload other models first"
        )
    check_load_headroom()
    mx.synchronize()
    previous = mx.set_wired_limit(limit)
    mx.set_memory_limit(limit)
    mx.set_cache_limit(8 * 2**30)
    return {
        "wired_limit_bytes": limit,
        "previous_wired_limit_bytes": previous,
        "allocator_guideline_bytes": limit,
        "cache_limit_bytes": 8 * 2**30,
        "load_available_floor_bytes": 12 * 2**30,
        "physical_memory_reserve_bytes": reserve,
    }


def context_limit(maximum=None):
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError("Flash-1017 context limit must be a positive integer")
    return min(maximum if maximum is not None else MAX_CONTEXT, MAX_CONTEXT)


def apply_model_settings(settings):
    """Return an effective copy; don't silently rewrite the user's saved settings."""
    if getattr(settings, "deepseek_v41_flash1017_enabled", False) is not True:
        raise ValueError("Flash-1017 requires an explicit boolean opt-in")
    if getattr(settings, "moe_expert_offload_enabled", False):
        raise ValueError("Flash-1017 has not qualified expert offload")
    if getattr(settings, "turboquant_kv_enabled", False):
        raise ValueError("Flash-1017 has not qualified quantized KV caches")
    if getattr(settings, "specprefill_enabled", False):
        raise ValueError("Flash-1017 cannot combine with SpecPrefill")
    if getattr(settings, "external_engine", None):
        raise ValueError("Flash-1017 requires the native oMLX engine")
    if any(
        getattr(settings, name, False) for name in ("dflash_enabled", "vlm_mtp_enabled")
    ):
        raise ValueError("Flash-1017 cannot combine with another speculative decoder")
    result = copy.copy(settings)
    result.deepseek_v41_engram_ssd_offload = True
    result.deepseek_v41_ced_prefill_enabled = False
    result.mtp_enabled = True
    result.mtp_fixed_depth = 5
    result.mtp_adaptive_max_depth = 5
    maximum = getattr(settings, "max_context_window", None)
    result.max_context_window = context_limit(maximum)
    return result


def configure_scheduler(config):
    """Apply model-local limits and separate caches from the normal arithmetic."""
    config.prefill_step_size = PREFILL_STEP
    config.max_num_seqs = 1
    config.completion_batch_size = 1
    if config.paged_ssd_cache_dir:
        path = Path(config.paged_ssd_cache_dir)
        if path.name != PROFILE_VERSION:
            config.paged_ssd_cache_dir = str(path / PROFILE_VERSION)
    return config
