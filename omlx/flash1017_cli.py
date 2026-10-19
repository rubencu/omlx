# SPDX-License-Identifier: Apache-2.0
"""Portable, isolated launch workflow for the opt-in Flash-1017 profile."""

import json
from pathlib import Path

from .patches.deepseek_v41.flash1017.profile import (
    MAX_CONTEXT,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PROFILE_VERSION,
    apply_model_settings,
    validate_checkpoint,
)

MODEL_ID = "DeepSeek-V4.1-Flash-1017"
MARKER = "flash1017-profile.json"


def configure(model, base_path, port=8001):
    """Prepare a separate normal oMLX configuration; never copy model weights."""
    from .model_settings import ModelSettings, ModelSettingsManager
    from .settings import GlobalSettings

    model = validate_checkpoint(model)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    base = Path(base_path).expanduser().resolve()
    marker = base / MARKER
    identity = {"profile": PROFILE_VERSION, "checkpoint": str(model)}
    if base == model or base in model.parents or model in base.parents:
        raise ValueError("The server directory must be separate from the checkpoint")
    if (
        base.exists()
        and any(base.iterdir())
        and (not marker.is_file() or json.loads(marker.read_text()) != identity)
    ):
        raise ValueError(
            "Choose an empty directory or this checkpoint's existing Flash-1017 directory"
        )
    link = base / "models" / MODEL_ID
    if (link.exists() or link.is_symlink()) and (
        not link.is_symlink() or link.resolve() != model
    ):
        raise ValueError(f"Refusing to replace a different model at {link}")

    settings = (
        GlobalSettings.load(base_path=base)
        if marker.exists()
        else GlobalSettings(base_path=base)
    )
    settings.server.host = "127.0.0.1"
    settings.server.port = port
    settings.model.model_dirs = [str(base / "models")]
    settings.model.model_dir = None
    settings.huggingface.hf_cache_enabled = False
    settings.scheduler.max_concurrent_requests = 1
    settings.cache.hot_cache_max_size = "4GB"
    settings.cache.ssd_cache_max_size = "20GB"
    settings.cache.ssd_cache_dir = str(base / "cache")
    settings.sampling.max_context_window = MAX_CONTEXT
    settings.ensure_directories()
    if not link.is_symlink():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(model, target_is_directory=True)
    settings.save()
    manager = ModelSettingsManager(base)
    if marker.exists():
        model_settings = manager.get_settings(MODEL_ID)
        model_settings.deepseek_v41_flash1017_enabled = True
    else:
        model_settings = ModelSettings(
            deepseek_v41_flash1017_enabled=True,
            deepseek_v41_engram_ssd_offload=True,
            deepseek_v41_ced_prefill_enabled=False,
            mtp_enabled=True,
            mtp_fixed_depth=5,
            mtp_adaptive_max_depth=5,
            max_context_window=MAX_CONTEXT,
            enable_thinking=False,
            temperature=1.0,
            top_p=1.0,
            is_default=True,
        )
    manager.set_settings(MODEL_ID, apply_model_settings(model_settings))
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    return base


def run(args):
    model = validate_checkpoint(args.model)
    if args.check:
        print(
            json.dumps(
                {
                    "profile": PROFILE_VERSION,
                    "checkpoint": str(model),
                    "repository": MODEL_REPOSITORY,
                    "revision": MODEL_REVISION,
                    "metadata_verified": True,
                    "weight_shards_hashed": False,
                    "model_loaded": False,
                    "max_total_context_tokens": MAX_CONTEXT,
                    "max_concurrent_requests": 1,
                    "minimum_unified_memory_gib": 256,
                },
                indent=2,
            )
        )
        return
    base = configure(model, args.base_path, args.port)
    print(f"Flash-1017 settings: {base}")
    if args.configure_only:
        return
    from .cli import main

    main(
        [
            "serve",
            "--base-path",
            str(base),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--max-concurrent-requests",
            "1",
        ]
    )
