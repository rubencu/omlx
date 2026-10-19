# SPDX-License-Identifier: Apache-2.0
"""Pool-boundary contracts for the explicit Flash1017 profile.

No checkpoint allocation or inference: residency and loader boundaries are
mocked, while the actual setting normalization and refusal paths are exercised.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from omlx import engine_pool as pool_module
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_settings import ModelSettings
from omlx.patches.deepseek_v41.flash1017 import profile


@pytest.fixture
def pool_and_entry(monkeypatch):
    pool = EnginePool.__new__(EnginePool)
    entry = EngineEntry(
        model_id="flash1017",
        model_path="/fixture/flash1017",
        model_type="vlm",
        engine_type="vlm",
        estimated_size=1000,
        config_model_type="deepseek_v41",
        model_context_length=1048576,
    )
    pool._entries = {entry.model_id: entry}
    pool._settings_manager = None
    pool._cluster_registry = None
    pool._lock = asyncio.Lock()
    monkeypatch.setattr(
        pool, "_distributed_deployment_for_entry", Mock(return_value=None)
    )
    return pool, entry


def requested(**kwargs):
    return ModelSettings(deepseek_v41_flash1017_enabled=True, **kwargs)


def test_effective_profile_precedes_admission_without_mutating_saved_settings(
    pool_and_entry,
):
    pool, entry = pool_and_entry
    saved = requested(
        mtp_enabled=False,
        max_context_window=1048576,
        deepseek_v41_ced_prefill_enabled=True,
    )
    effective = pool._effective_flash1017_model_settings(entry, saved)
    assert effective is not saved
    assert effective.mtp_enabled and effective.mtp_fixed_depth == 5
    assert effective.mtp_adaptive_max_depth == 5
    assert effective.deepseek_v41_engram_ssd_offload
    assert not effective.deepseek_v41_ced_prefill_enabled
    assert effective.max_context_window == profile.MAX_CONTEXT
    assert not saved.mtp_enabled and saved.deepseek_v41_ced_prefill_enabled
    assert saved.max_context_window == 1048576


def test_non_profile_settings_are_returned_unmodified(pool_and_entry, monkeypatch):
    pool, entry = pool_and_entry
    entry.config_model_type = "qwen3"
    entry.engine_type = "batched"
    saved = ModelSettings(mtp_enabled=False)
    checked = Mock(side_effect=AssertionError("Non-profile checkpoint validation"))
    monkeypatch.setattr(profile, "validate_checkpoint", checked)
    assert (
        pool._effective_flash1017_model_settings(
            entry, saved, force_lm=True, validate_checkpoint=True
        )
        is saved
    )
    assert pool._effective_flash1017_model_settings(entry, None) is None
    checked.assert_not_called()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("config_model_type", "qwen3", "VLM engine"),
        ("engine_type", "batched", "VLM engine"),
        ("source_type", "cluster", "single-machine"),
    ],
)
def test_other_model_engine_or_cluster_refused(pool_and_entry, field, value, message):
    pool, entry = pool_and_entry
    setattr(entry, field, value)
    with pytest.raises(ValueError, match=message):
        pool._effective_flash1017_model_settings(entry, requested())


def test_distributed_deployment_refused(pool_and_entry):
    pool, entry = pool_and_entry
    pool._distributed_deployment_for_entry.return_value = object()
    with pytest.raises(ValueError, match="single-machine"):
        pool._effective_flash1017_model_settings(entry, requested())


@pytest.mark.parametrize(
    "options",
    [
        {"external_engine": "splash"},
        {"dflash_enabled": True},
        {"vlm_mtp_enabled": True},
        {"specprefill_enabled": True},
        {"moe_expert_offload_enabled": True},
        {"turboquant_kv_enabled": True},
    ],
)
def test_conflicting_runtime_modes_refused_at_pool_boundary(pool_and_entry, options):
    pool, entry = pool_and_entry
    with pytest.raises(ValueError, match="Flash-1017"):
        pool._effective_flash1017_model_settings(entry, requested(**options))


def test_metadata_checked_for_load_but_not_ready_engine_resolution(
    pool_and_entry, monkeypatch
):
    pool, entry = pool_and_entry
    checked = Mock()
    monkeypatch.setattr(profile, "validate_checkpoint", checked)
    pool._effective_flash1017_model_settings(entry, requested())
    checked.assert_not_called()
    pool._effective_flash1017_model_settings(
        entry, requested(), validate_checkpoint=True
    )
    checked.assert_called_once_with(entry.model_path)


def test_residency_receives_the_effective_profile(pool_and_entry, monkeypatch):
    pool, entry = pool_and_entry
    saved = requested(mtp_enabled=False)
    seen = []
    monkeypatch.setattr(pool, "_entry_resident_size", lambda entry: 1000)
    monkeypatch.setattr(
        pool, "_qwen4_ple_offload_status", lambda *a: (False, False, None)
    )

    def residency(entry, settings):
        seen.append(settings)
        return (
            settings.deepseek_v41_engram_ssd_offload,
            False,
            SimpleNamespace(mmap_bytes=100, resident_bytes=1000),
        )

    monkeypatch.setattr(pool, "_deepseek_v41_engram_offload_status", residency)
    monkeypatch.setattr(pool_module, "_qwen35_cpu_share_estimated_bytes", lambda *a: 0)
    monkeypatch.setattr(pool_module, "dflash_draft_for", lambda *a: None)
    assert pool._entry_runtime_resident_size(entry, saved) == 100
    assert seen[0].mtp_enabled and seen[0].deepseek_v41_engram_ssd_offload
    assert not saved.mtp_enabled and not saved.deepseek_v41_engram_ssd_offload


def test_requested_and_effective_signature_match(pool_and_entry, monkeypatch):
    pool, entry = pool_and_entry
    monkeypatch.setattr(
        pool, "_qwen4_ple_offload_status", lambda *a: (False, False, None)
    )
    monkeypatch.setattr(
        pool,
        "_deepseek_v41_engram_offload_status",
        lambda entry, settings: (settings.deepseek_v41_engram_ssd_offload, False, None),
    )
    saved = requested(
        mtp_enabled=False, mtp_fixed_depth=2, deepseek_v41_ced_prefill_enabled=True
    )
    effective = profile.apply_model_settings(saved)
    first = pool._engine_runtime_signature(entry.model_id, saved)
    assert first == pool._engine_runtime_signature(entry.model_id, effective)
    values = dict(first)
    assert values["deepseek_v41_flash1017_enabled"] == "True"
    assert values["mtp_enabled"] == "True"
    assert values["mtp_fixed_depth"] == "5"
    assert values["deepseek_v41_ced_prefill_enabled"] == "False"
    assert values["flash1017_max_total_tokens"] == "131072"
    assert first == pool._engine_runtime_signature(
        entry.model_id, requested(max_context_window=1048576)
    )
    assert first != pool._engine_runtime_signature(
        entry.model_id, requested(max_context_window=8192)
    )
    assert first != pool._engine_runtime_signature(entry.model_id, ModelSettings())
    assert "flash1017_max_total_tokens" not in dict(
        pool._engine_runtime_signature(entry.model_id, ModelSettings())
    )


@pytest.mark.asyncio
async def test_force_lm_rejected_before_family_dispatch_or_load(
    pool_and_entry, monkeypatch
):
    pool, entry = pool_and_entry
    dispatch = Mock(side_effect=AssertionError("Profile was silently coerced"))
    monkeypatch.setattr(pool, "_has_mlx_lm_path", dispatch)
    with pytest.raises(ValueError, match="force_lm"):
        await pool.get_engine(
            entry.model_id, force_lm=True, runtime_settings=requested()
        )
    dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_direct_load_rejects_metadata_before_memory_or_launcher(
    pool_and_entry, monkeypatch
):
    pool, entry = pool_and_entry
    pool._check_splash_launcher = AsyncMock()
    memory = Mock(
        side_effect=AssertionError("Read allocator before profile validation")
    )
    monkeypatch.setattr(pool_module.mx, "get_active_memory", memory)
    monkeypatch.setattr(
        profile, "validate_checkpoint", Mock(side_effect=ValueError("wrong checkpoint"))
    )
    with pytest.raises(ValueError, match="wrong checkpoint"):
        await pool._load_engine(entry.model_id, runtime_settings=requested())
    assert not entry.is_loading
    pool._check_splash_launcher.assert_not_awaited()
    memory.assert_not_called()


@pytest.mark.asyncio
async def test_profile_concurrency_stays_single_without_clamping_other_engines(
    pool_and_entry,
):
    pool, entry = pool_and_entry
    pool._scheduler_config = SimpleNamespace(max_num_seqs=8, completion_batch_size=8)
    profile_config = SimpleNamespace(max_num_seqs=1, completion_batch_size=1)
    regular_config = SimpleNamespace(max_num_seqs=8, completion_batch_size=8)
    entry.engine = SimpleNamespace(
        _scheduler_config=profile_config, _model_settings=requested()
    )
    pool._entries["other"] = SimpleNamespace(
        engine=SimpleNamespace(
            _scheduler_config=regular_config, _model_settings=ModelSettings()
        )
    )
    await pool.apply_max_concurrent_requests(16)
    assert pool._scheduler_config.max_num_seqs == 16
    assert regular_config.max_num_seqs == 16
    assert profile_config.max_num_seqs == profile_config.completion_batch_size == 1
