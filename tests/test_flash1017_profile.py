# SPDX-License-Identifier: Apache-2.0
"""Profile wiring and portable CLI contracts; no checkpoint inference."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from omlx import flash1017_cli
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.patches.deepseek_v41.flash1017 import profile
from omlx.settings import GlobalSettings


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint"
    path.mkdir()
    contents = {
        "config.json": b'{"model_type":"deepseek_v41"}',
        "tokenizer.json": b"{}",
    }
    for name, content in contents.items():
        (path / name).write_bytes(content)
    monkeypatch.setattr(
        profile,
        "METADATA_SHA256",
        {
            name: hashlib.sha256(content).hexdigest()
            for name, content in contents.items()
        },
    )
    return path


def test_checkpoint_metadata_checked_before_configuration(checkpoint, tmp_path):
    assert profile.validate_checkpoint(checkpoint) == checkpoint
    (checkpoint / "tokenizer.json").write_text("changed")
    base = tmp_path / "server"
    with pytest.raises(ValueError, match="tokenizer.json"):
        flash1017_cli.configure(checkpoint, base)
    assert not base.exists()


def test_explicit_profile_preserves_original_settings():
    original = ModelSettings(
        deepseek_v41_flash1017_enabled=True, max_context_window=1_000_000
    )
    effective = profile.apply_model_settings(original)
    assert not original.mtp_enabled
    assert original.max_context_window == 1_000_000
    assert effective.deepseek_v41_engram_ssd_offload
    assert not effective.deepseek_v41_ced_prefill_enabled
    assert effective.mtp_enabled and effective.mtp_fixed_depth is None
    assert effective.mtp_adaptive_max_depth == 3
    assert effective.max_context_window == 131072


@pytest.mark.parametrize(
    "field",
    [
        "moe_expert_offload_enabled",
        "turboquant_kv_enabled",
        "specprefill_enabled",
        "dflash_enabled",
        "vlm_mtp_enabled",
        "external_engine",
    ],
)
def test_incompatible_runtime_rejected(field):
    settings = SimpleNamespace(deepseek_v41_flash1017_enabled=True)
    setattr(settings, field, True)
    with pytest.raises(ValueError):
        profile.apply_model_settings(settings)


@pytest.mark.parametrize("maximum", [0, -1, True, "131072", 1.5])
def test_invalid_context_rejected(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        profile.context_limit(maximum)


def test_scheduler_limits_and_cache_namespace():
    settings = SimpleNamespace(
        prefill_step_size=2048,
        max_num_seqs=8,
        completion_batch_size=8,
        paged_ssd_cache_dir="/cache",
    )
    profile.configure_scheduler(settings)
    profile.configure_scheduler(settings)
    assert (
        settings.prefill_step_size,
        settings.max_num_seqs,
        settings.completion_batch_size,
    ) == (2048, 1, 1)
    assert settings.paged_ssd_cache_dir == "/cache/flash1017-v2"


def test_prepare_memory_limits_and_small_machine_refusal(monkeypatch):
    import mlx.core as mx

    calls = []
    monkeypatch.setattr(
        profile, "check_load_headroom", lambda: calls.append("headroom")
    )
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(available=240 * 2**30)
    )
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {
            "memory_size": 256 * 2**30,
            "max_recommended_working_set_size": 240 * 2**30,
        },
    )
    monkeypatch.setattr(mx, "synchronize", lambda: calls.append("sync"))
    for name in ("set_wired_limit", "set_memory_limit", "set_cache_limit"):
        monkeypatch.setattr(
            mx, name, lambda value, name=name: calls.append((name, value))
        )
    receipt = profile.prepare_memory()
    assert receipt["wired_limit_bytes"] == 224 * 2**30
    assert calls == [
        "headroom",
        "sync",
        ("set_wired_limit", 224 * 2**30),
        ("set_memory_limit", 224 * 2**30),
        ("set_cache_limit", 8 * 2**30),
    ]
    calls.clear()
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {
            "memory_size": 128 * 2**30,
            "max_recommended_working_set_size": 120 * 2**30,
        },
    )
    with pytest.raises(MemoryError):
        profile.prepare_memory()
    assert not calls
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {
            "memory_size": 256 * 2**30,
            "max_recommended_working_set_size": 240 * 2**30,
        },
    )
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(available=200 * 2**30)
    )
    with pytest.raises(MemoryError, match="available"):
        profile.prepare_memory()
    assert not calls


def test_cli_check_is_read_only(checkpoint, tmp_path, capsys):
    base = tmp_path / "server"
    from omlx.cli import main

    main(["flash1017", "--model", str(checkpoint), "--base-path", str(base), "--check"])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["metadata_verified"] and not receipt["model_loaded"]
    assert not base.exists()


def test_configuration_is_separate_and_repeatable(checkpoint, tmp_path):
    base = flash1017_cli.configure(checkpoint, tmp_path / "server")
    settings = GlobalSettings.load(base_path=base)
    saved = ModelSettingsManager(base).get_settings(flash1017_cli.MODEL_ID)
    assert saved.deepseek_v41_flash1017_enabled and saved.mtp_fixed_depth is None
    assert saved.max_context_window == 131072 and saved.enable_thinking is False
    assert settings.server.host == "127.0.0.1" and settings.server.port == 8001
    assert not settings.huggingface.hf_cache_enabled
    assert settings.scheduler.max_concurrent_requests == 1
    assert (base / "models" / flash1017_cli.MODEL_ID).resolve() == checkpoint
    settings.auth.api_key = "local-test-key"
    settings.save()
    saved.max_context_window = 16384
    saved.temperature = 0.5
    ModelSettingsManager(base).set_settings(flash1017_cli.MODEL_ID, saved)
    flash1017_cli.configure(checkpoint, base, port=8002)
    assert GlobalSettings.load(base_path=base).auth.api_key == "local-test-key"
    saved = ModelSettingsManager(base).get_settings(flash1017_cli.MODEL_ID)
    assert saved.max_context_window == 16384 and saved.temperature == 0.5


def test_configuration_refuses_existing_unrelated_files(checkpoint, tmp_path):
    base = tmp_path / "normal-omlx"
    base.mkdir()
    (base / "settings.json").write_text('{"existing":true}')
    with pytest.raises(ValueError, match="empty directory"):
        flash1017_cli.configure(checkpoint, base)
    assert (base / "settings.json").read_text() == '{"existing":true}'


def test_launch_uses_normal_server(checkpoint, tmp_path):
    args = SimpleNamespace(
        model=checkpoint,
        base_path=tmp_path / "server",
        port=8002,
        check=False,
        configure_only=False,
    )
    with patch("omlx.cli.main") as serve:
        flash1017_cli.run(args)
    assert serve.call_args.args[0] == [
        "serve",
        "--base-path",
        str(args.base_path),
        "--host",
        "127.0.0.1",
        "--port",
        "8002",
        "--max-concurrent-requests",
        "1",
    ]


def test_engine_profile_does_not_mutate_shared_scheduler():
    from omlx.engine.vlm import VLMBatchedEngine

    shared = SimpleNamespace(max_num_seqs=8)
    engine = VLMBatchedEngine(
        "unused",
        scheduler_config=shared,
        model_settings=ModelSettings(deepseek_v41_flash1017_enabled=True),
    )
    engine._scheduler_config.max_num_seqs = 1
    assert shared.max_num_seqs == 8


@pytest.mark.parametrize(
    "maximum,expected", [(None, 131072), (1_000_000, 131072), (16384, 16384)]
)
def test_api_context_cap(maximum, expected):
    import omlx.server as server

    settings = ModelSettings(
        deepseek_v41_flash1017_enabled=True, max_context_window=maximum
    )
    with (
        patch.object(server, "get_model_settings_for_request", return_value=settings),
        patch.object(server, "resolve_model_id", return_value="model"),
    ):
        assert server._configured_max_context_window("model") == expected


def test_api_context_profile_requires_literal_opt_in():
    from unittest.mock import MagicMock

    import omlx.server as server

    settings = MagicMock(max_context_window=4096)
    with (
        patch.object(server, "get_model_settings_for_request", return_value=settings),
        patch.object(server, "resolve_model_id", return_value="model"),
    ):
        assert server._configured_max_context_window("model") == 4096


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_partial_start_closes_model_on_worker(monkeypatch, failure):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import mlx.core as mx

    from omlx.engine.vlm import VLMBatchedEngine

    calls = []
    main_thread = threading.get_ident()

    def close():
        calls.append(threading.get_ident())
        if failure:
            raise RuntimeError("teardown error")

    monkeypatch.setattr(mx, "synchronize", lambda: None)
    engine = VLMBatchedEngine("unused")
    engine._vlm_model = SimpleNamespace(
        _flash1017_runtime_controller=object(), close=close
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr("omlx.engine_core.get_mlx_executor", lambda: executor)
        await engine.stop()
    assert len(calls) == 1 and calls[0] != main_thread
    assert engine._vlm_model is None
