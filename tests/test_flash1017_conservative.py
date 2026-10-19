# SPDX-License-Identifier: Apache-2.0
"""CPU construction/control checks for the native-arithmetic profile.

No checkpoint, generation or Metal kernels. Real tiny LanguageModel classes
and parameter trees exercise the non-wrapping boundary; storage lifetime/bytes
are independently covered by test_flash1017_runtime's actual row controller.
"""

import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches import mlx_lm_mtp
from omlx.patches.deepseek_v41 import language
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.flash1017 import runtime
from omlx.patches.deepseek_v41.mtp import AcceptanceDepthController


@pytest.fixture(autouse=True)
def cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class StorageStub:
    """Exclude IO from this real-class/parameter-identity construction test."""

    def __init__(self, model, **kwargs):
        assert kwargs['own_model_close'] is False
        self._lock = threading.RLock()
        self.enabled = kwargs['enabled']
        self.status = 'active'

    def _reject_reader_control(self):
        pass

    def _reject_forward_control(self):
        pass

    def _validate_restore(self):
        pass

    def set_enabled(self, enabled):
        self.enabled = enabled

    def snapshot(self):
        return {'status': self.status, 'enabled': self.enabled, 'retained_bytes': 0}

    def close(self):
        self.status, self.enabled = 'detached', False


def test_native_model_types_parameters_and_acceptance_policy_are_untouched(monkeypatch):
    monkeypatch.setattr(mlx_lm_mtp, '_MTP_ACTIVE', True)
    monkeypatch.setattr(mlx_lm_mtp, '_MTP_DEPTH', 3)
    monkeypatch.setattr(mlx_lm_mtp, '_MTP_DEPTH_FIXED', False)
    config = ModelConfig(
        vocab_size=64, dim=32, moe_inter_dim=32, n_layers=5, n_heads=2,
        q_lora_rank=32, head_dim=32, rope_head_dim=8, o_groups=2,
        o_lora_rank=16, index_n_heads=8, index_head_dim=32, index_topk=3,
        n_routed_experts=4, n_activated_experts=2, window_size=4,
        dtype='bf16', expert_dtype=None, n_mtp_layers=3, max_seq_len=64,
        max_batch_size=1, hc_sinkhorn_iters=3, candidate_source_layer=3,
        candidate_block_size=2, candidate_topk_blocks=2,
        index_source_layers=(1, 3, 4), preserve_mtp=True, dspark_block_size=5,
        dspark_noise_token_id=2, dspark_target_layer_ids=(2, 3, 4),
        dspark_n_routed_experts=2, dspark_n_activated_experts=1,
        dspark_markov_rank=32, compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
        flash1017_enabled=True, attention_output_quantization=False,
    )
    target = language.LanguageModel(config)
    target._engram_prefetch = SimpleNamespace(_closed=False, _pending=None)
    outer = SimpleNamespace(language_model=target, parameters=target.parameters, close=lambda: None)
    objects = [target, *target.layers, *target.mtp]
    objects += [block.attn.wkv for block in target.layers]
    objects += [stage.attn.wkv for stage in target.mtp]
    types_before = [type(obj) for obj in objects]
    parameters_before = runtime.parameter_identity(outer)
    hooks_before = runtime.native_hooks()
    monkeypatch.setattr(runtime, '_install_rows', StorageStub)
    monkeypatch.setattr(runtime, '_install_cone', lambda *a, **kw: pytest.fail('Unexpected replay installation'))
    owner = runtime.install(outer, replay=False)
    try:
        assert [type(obj) for obj in objects] == types_before
        assert runtime.parameter_identity(outer) == parameters_before
        assert runtime.native_hooks() == hooks_before
        for name in ('_forward', '_omlx_prefill', 'dspark_append_context', '_flash1017_cone_owner', '_flash1017_experiment'):
            assert name not in vars(target) and name not in target
        assert target._omlx_mtp_depth == 3
        assert target._omlx_mtp_depth_fixed is False
        assert target._omlx_dspark_decode_enabled is True
        assert config.attention_output_quantization is False and config.ced_prefill is False
        controller = target.make_mtp_depth_controller(target._omlx_mtp_depth)
        assert type(controller) is AcceptanceDepthController and controller.cur == 3
        for used, accepted, expected in [(3, 0, 1), (1, 1, 2), (2, 2, 3), (3, 1, 2)]:
            controller.observe(used, accepted, cycle_ms=1_000_000)
            assert controller.cur == expected and not controller.should_exit()
        assert owner.snapshot()['cone'] is None
    finally:
        owner.close()
    assert [type(obj) for obj in objects] == types_before
    assert runtime.parameter_identity(outer) == parameters_before
    assert runtime.native_hooks() == hooks_before
    assert owner.snapshot()['model_references_released']
