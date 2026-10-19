# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the opt-in inference profile; no checkpoint loads."""

import json
import math
import sys
from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import dspark, head, language, loading
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.flash1017 import policy
from omlx.patches.deepseek_v41.quantization import quantize_activation
from omlx.patches.mlx_lm_mtp import batch_generator as bg


@pytest.fixture(autouse=True)
def cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def test_profile_cannot_be_activated_by_checkpoint_metadata():
    config = ModelConfig.from_dict({"flash1017_enabled": True})
    assert not config.flash1017_enabled
    assert not config.attention_output_quantization
    config = ModelConfig.from_dict(
        {
            "text_config": {"flash1017_enabled": True},
            "omlx_deepseek_v41": {"attention_output_quantization": True},
        }
    )
    assert not config.flash1017_enabled
    assert config.attention_output_quantization


@pytest.mark.parametrize("value", ["false", 1, None])
def test_attention_rounding_metadata_requires_a_boolean(value):
    with pytest.raises(TypeError, match="boolean"):
        ModelConfig.from_dict(
            {"omlx_deepseek_v41": {"attention_output_quantization": value}}
        )


def test_causal_policy_never_peeks_beyond_first_nonimproving_prefix():
    def stream():
        yield -10.0
        raise AssertionError("Read future candidate-dependent confidence")

    assert policy.PrefixPolicy((10, 11, 12)).choose(stream()).depth == 0
    assert policy.PrefixPolicy((10, 15, 6)).choose((0, 100)).depth == 0


def test_policy_includes_bonus_and_conditional_prefix_survival():
    result = policy.PrefixPolicy((10, 11, 12)).choose((math.log(4), math.log(3)))
    assert result.depth == 2
    assert result.expected_tokens == pytest.approx(2.4)
    assert result.predicted_tokens_per_ms == pytest.approx(0.2)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_confidence_keeps_only_prior_admitted_prefix(bad):
    assert policy.PrefixPolicy((10, 11, 12)).choose((10, bad)).depth == 1


@pytest.mark.parametrize(
    "costs,temperatures",
    [
        ((), ()),
        ((1,), ()),
        ((1, 0), ()),
        ((1, math.inf), ()),
        ((1, 2), (0,)),
        ((1, 2), (math.nan,)),
        ((1, 2, 3), (1,)),
    ],
)
def test_invalid_calibration_rejected(costs, temperatures):
    with pytest.raises(ValueError):
        policy.PrefixPolicy(costs, temperatures)


def test_calibration_and_counters_are_independent_per_model():
    a, b = policy.ConfidenceController(), policy.ConfidenceController()
    before = a.select([0.5] * 5, 5)
    for _ in range(100):
        a.record(verified=before, accepted=0, copied=False, cycle_ms=1e99)
    assert a.select([0.5] * 5, 5) == before
    assert a.cycles == 100 and b.cycles == 0
    assert len(a.selected_depths) == 6
    assert sum(a.selected_depths) == 2 and sum(b.selected_depths) == 0
    assert not hasattr(a, "events")
    with pytest.raises(FrozenInstanceError):
        a.policy.cycle_ms = (1, 2)


def test_policy_configuration_requires_opt_in_and_full_width():
    host = SimpleNamespace(_config=ModelConfig(), configure_mtp=lambda *_: None)
    with pytest.raises(ValueError, match="opt-in"):
        policy.configure(host)
    host._config.flash1017_enabled = True
    with pytest.raises(ValueError, match="five"):
        policy.configure(host)
    host._config.dspark_block_size = 5
    calls = []
    host.configure_mtp = lambda *args: calls.append(args)
    controller = policy.configure(host)
    assert calls == [(True, 5)]
    assert host._omlx_mtp_depth_fixed
    assert host._flash1017_experiment is controller


class _Tensor:
    """Routing sentinel: never allocates an MLX array or invokes a kernel."""

    def __init__(self, shape, dtype=mx.bfloat16):
        self.shape, self.dtype = shape, dtype
        self.size = math.prod(shape)

    def astype(self, dtype):
        return _Tensor(self.shape, dtype)

    @property
    def T(self):  # noqa: N802 - tensor transpose API
        return _Tensor(self.shape[::-1], self.dtype)

    def __matmul__(self, other):
        return "fallback"


@pytest.mark.parametrize("rows", [1, 2, 5, 6, 7])
@pytest.mark.parametrize("enabled", [False, True])
def test_head_dispatch_is_opt_in_and_bounded(monkeypatch, rows, enabled):
    calls = []

    def kernel(kind):
        def run(**kwargs):
            calls.append((kind, kwargs))
            return [kind]

        return run

    monkeypatch.setattr(head, "_kernel", lambda: kernel("original"))
    monkeypatch.setattr(head, "_reuse_kernel", lambda: kernel("reuse"))
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    result = head.project_logits(
        _Tensor((1, rows, 256)), _Tensor((4099, 256)), reuse_weights=enabled
    )
    expected = (
        "reuse"
        if enabled and 2 <= rows <= 5
        else ("original" if rows <= 5 else "fallback")
    )
    assert result == expected
    if calls:
        assert calls[0][1]["output_shapes"] == [(1, rows, 4099)]
        assert calls[0][1]["output_dtypes"] == [mx.float32]


@pytest.mark.parametrize(
    "shape,weight_shape,dtype,device",
    [
        ((1, 3, 256), (4099, 256), mx.bfloat16, mx.cpu),
        ((1, 3, 256), (4099, 256), mx.float32, mx.gpu),
        ((1, 3, 260), (4099, 264), mx.bfloat16, mx.gpu),
        ((1, 3, 257), (4099, 257), mx.bfloat16, mx.gpu),
        ((1, 3, 128), (4099, 128), mx.bfloat16, mx.gpu),
        ((1, 3, 256), (4095, 256), mx.bfloat16, mx.gpu),
    ],
)
def test_head_retains_existing_ineligible_fallbacks(
    monkeypatch, shape, weight_shape, dtype, device
):
    monkeypatch.setattr(mx, "default_device", lambda: device)
    monkeypatch.setattr(head, "_kernel", lambda: pytest.fail("Unexpected kernel"))
    monkeypatch.setattr(head, "_reuse_kernel", lambda: pytest.fail("Unexpected reuse"))
    assert (
        head.project_logits(
            _Tensor(shape), _Tensor(weight_shape, dtype), reuse_weights=True
        )
        == "fallback"
    )


@pytest.mark.parametrize("kind", ["target", "draft"])
def test_reference_rounding_occurs_after_inverse_rope_for_both_paths(monkeypatch, kind):
    config = ModelConfig(
        dim=32,
        n_heads=1,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=0,
        o_groups=1,
        o_lora_rank=32,
        compress_ratios=(0,),
        kv_source_layers=(),
        index_source_layers=(),
    )
    module = language if kind == "target" else dspark
    cls = language.Attention if kind == "target" else dspark.DSparkAttention
    attention = cls(config, 0)
    attention.wo_a.weight = mx.eye(32)
    attention.wo_b.weight = mx.eye(32)
    out = mx.linspace(-1.7, 2.3, 64).reshape(1, 2, 1, 32)
    monkeypatch.setattr(
        module, "rope", lambda x, *a, inverse=False: x + 0.037 if inverse else x
    )
    if kind == "target":
        monkeypatch.setattr(language, "packed_sparse_attention", lambda *a: out)

        def call():
            return attention(mx.ones((1, 2, 32)), [None] * 7, {}, 0)

    else:
        monkeypatch.setattr(
            mx.fast,
            "scaled_dot_product_attention",
            lambda *a, **k: out.transpose(0, 2, 1, 3),
        )

        def call():
            return attention(
                mx.ones((1, 2, 32)),
                SimpleNamespace(offset=0, keys=mx.zeros((1, 1, 0, 32))),
            )

    unrounded = call()
    config.attention_output_quantization = True
    rounded = call()
    expected = quantize_activation((out + 0.037).reshape(1, 2, 1, 32)).reshape(1, 2, 32)
    np.testing.assert_array_equal(rounded, expected)
    np.testing.assert_array_equal(unrounded, (out + 0.037).reshape(1, 2, 32))
    assert not mx.array_equal(rounded, unrounded).item()


def _draft_fixture(enabled=True):
    calls = {"widths": [], "previous": [], "confidence": [], "append": []}

    def forward(hidden, anchor, cache, *, draft_length):
        calls["widths"].append(draft_length)
        return mx.zeros((1, draft_length, 16)), mx.ones((1, draft_length, 2))

    def markov(previous):
        calls["previous"].append(previous.item())
        bias = mx.where(mx.arange(16)[None] == previous[:, None] + 1, 100, -100)
        return bias.astype(mx.float32), previous[:, None].astype(mx.float32)

    def confidence(hidden, embeddings):
        calls["confidence"].append(embeddings.reshape(-1).tolist())
        return mx.zeros(embeddings.shape[:2])

    host = SimpleNamespace(
        _omlx_dspark_decode_enabled=True,
        args=SimpleNamespace(dspark_block_size=5),
        dspark_forward=forward,
        dspark_markov=markov,
        dspark_append_context=lambda *args: calls["append"].append(args),
        mtp=[SimpleNamespace(confidence_head=confidence)],
    )
    if enabled:
        host._flash1017_experiment = policy.ConfidenceController()
    batch = SimpleNamespace(
        model=host,
        uids=[7],
        samplers=[None],
        logits_processors=None,
        fallback_sampler=lambda lp: mx.argmax(lp, axis=-1).astype(mx.uint32),
    )
    state = bg._MtpState(uid=7, chain=True, depth=3, mtp_cache=[])
    return host, batch, state, calls


def test_draft_confidence_uses_previous_token_and_frozen_width():
    host, batch, state, calls = _draft_fixture()
    bg._dspark_next_drafts(batch, state, mx.zeros((1, 1, 2)), mx.array([2]), None)
    assert state.drafts.tolist() == [3, 4, 5, 6, 7]
    assert calls["widths"] == [5]
    assert calls["previous"] == [2, 3, 4, 5, 6]
    assert calls["confidence"] == [[2.0, 3.0, 4.0, 5.0, 6.0]]
    assert state._flash1017_confidence.shape == (5,)


def test_disabled_policy_preserves_native_width_and_omits_confidence():
    host, batch, state, calls = _draft_fixture(False)
    bg._dspark_next_drafts(batch, state, mx.zeros((1, 1, 2)), mx.array([2]), None)
    assert state.drafts.tolist() == [3, 4, 5]
    assert calls["widths"] == [3] and calls["confidence"] == []
    assert not hasattr(state, "_flash1017_confidence")


def test_context_copy_appends_context_without_drafting_or_stale_confidence():
    host, batch, state, calls = _draft_fixture()
    state._flash1017_confidence = mx.array([100.0] * 5)
    bg._dspark_next_drafts(
        batch, state, mx.zeros((1, 2, 2)), mx.array([2, 3]), None, depth=0
    )
    assert state.hist_offset == 2 and state.drafts.size == 0
    assert state._flash1017_confidence is None
    assert len(calls["append"]) == 1 and calls["widths"] == []


def test_singleton_guard_precedes_draft_or_batched_verify():
    host, batch, state, calls = _draft_fixture()
    batch.uids = [7, 8]
    with pytest.raises(ValueError, match="one active"):
        bg._dspark_next_drafts(batch, state, None, None, None)
    with pytest.raises(ValueError, match="one active"):
        bg._run_verify_cycle_batched(batch, None)
    assert calls["widths"] == []


def test_confidence_clips_distributions_before_any_backbone_verify(monkeypatch):
    host, batch, state, calls = _draft_fixture()
    state.next_main = mx.array([2])
    state.drafts = mx.array([3, 4, 5, 6, 7])
    state.draft_lps = [object()] * 5
    state.draft_accept_lps = [object()] * 5
    state._flash1017_confidence = mx.array([-100.0] * 5)

    class StopBeforeBackboneError(Exception):
        pass

    def stop(_):
        raise StopBeforeBackboneError

    monkeypatch.setattr(bg, "_resolve_sampler", stop)
    with pytest.raises(StopBeforeBackboneError):
        bg._run_verify_cycle_chain(batch, state)
    assert state.drafts.size == 0
    assert state.draft_lps == state.draft_accept_lps == []
    assert state._flash1017_confidence is None
    assert host._flash1017_experiment.selected_depths[0] == 1


def _loader_fixture(tmp_path, monkeypatch):
    events, models = [], []
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "dspark_block_size": 5,
                "n_layers": 6,
                "compress_ratios": [0, 2, 2, 1, 1, 1, 0],
                "ced_prefill": True,
                "omlx_deepseek_v41": {"version": 1, "preserve_mtp": True},
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')

    class FakeModel:
        def __init__(self, config):
            events.append("model")
            models.append(self)
            self.config = config
            self.language_model = SimpleNamespace(
                _config=config,
                layers=[],
                configure_mtp=lambda *a: events.append(("mtp", a)),
            )

        def parameters(self):
            return {}

        def eval(self):
            events.append("eval")

        def close(self):
            events.append("close")

        def load_weights(self, *args, **kwargs):
            pass

    monkeypatch.setattr(loading, "Model", FakeModel)
    monkeypatch.setattr(
        loading,
        "PreTrainedTokenizerFast",
        SimpleNamespace(from_pretrained=lambda _: SimpleNamespace()),
    )
    monkeypatch.setattr(loading, "Processor", lambda *a: "processor")
    profile = ModuleType("omlx.patches.deepseek_v41.flash1017.profile")
    profile.validate_checkpoint = lambda path: events.append("validate")
    profile.prepare_memory = lambda: events.append("memory") or {"prepared": True}
    profile.check_load_headroom = lambda: events.append("headroom")
    runtime = ModuleType("omlx.patches.deepseek_v41.flash1017.runtime")
    runtime.install = lambda model, **kw: events.append(("install", kw))
    monkeypatch.setitem(sys.modules, profile.__name__, profile)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    # The package may already expose real submodules from another test.
    from omlx.patches.deepseek_v41 import flash1017

    monkeypatch.setattr(flash1017, "profile", profile, raising=False)
    monkeypatch.setattr(flash1017, "runtime", runtime, raising=False)
    return events, models, profile, runtime


def test_load_validation_precedes_allocation_and_installs_after_weights(
    tmp_path, monkeypatch
):
    events, models, _, _ = _loader_fixture(tmp_path, monkeypatch)
    model, processor = loading.load(
        tmp_path, flash1017_enabled=True, ced_prefill=True, preserve_mtp=False
    )
    assert processor == "processor"
    assert events == [
        "validate",
        "memory",
        "model",
        "eval",
        ("install", {"replay": False}),
    ]
    assert model._flash1017_memory_receipt == {"prepared": True}
    assert model.config.flash1017_enabled
    assert not model.config.attention_output_quantization
    assert model.config.preserve_mtp
    assert not hasattr(model.language_model, "_flash1017_experiment")
    assert not model.config.ced_prefill


def test_conservative_profile_overrides_extra_rounding_before_construction(tmp_path, monkeypatch):
    events, _, _, _ = _loader_fixture(tmp_path, monkeypatch)
    path = tmp_path / "config.json"
    raw = json.loads(path.read_text())
    raw["omlx_deepseek_v41"]["attention_output_quantization"] = True
    path.write_text(json.dumps(raw))
    model, _ = loading.load(
        tmp_path, flash1017_enabled=True, attention_output_quantization=True
    )
    assert model.config.attention_output_quantization is False
    assert events[-1] == ("install", {"replay": False})
    assert not any(isinstance(event, tuple) and event[0] == "mtp" for event in events)
    assert not hasattr(model.language_model, "_flash1017_experiment")


def test_disabled_load_never_installs_or_checks_profile(tmp_path, monkeypatch):
    events, models, _, _ = _loader_fixture(tmp_path, monkeypatch)
    model, _ = loading.load(tmp_path)
    assert events == ["model", "eval"]
    assert not model.config.flash1017_enabled
    assert not model.config.attention_output_quantization
    assert not hasattr(model.language_model, "_flash1017_experiment")


def test_install_failure_closes_model(tmp_path, monkeypatch):
    events, _, _, runtime = _loader_fixture(tmp_path, monkeypatch)

    def fail(model, *, replay):
        assert replay is False
        raise RuntimeError("installation failed")

    runtime.install = fail
    with pytest.raises(RuntimeError, match="installation failed"):
        loading.load(tmp_path, flash1017_enabled=True)
    assert events[-1] == "close"


def test_validation_failure_allocates_nothing(tmp_path, monkeypatch):
    events, models, profile, _ = _loader_fixture(tmp_path, monkeypatch)

    def fail(path):
        raise ValueError("wrong checkpoint")

    profile.validate_checkpoint = fail
    with pytest.raises(ValueError, match="wrong checkpoint"):
        loading.load(tmp_path, flash1017_enabled=True)
    assert not models


def test_expert_offload_is_rejected_before_allocation(tmp_path, monkeypatch):
    _, models, _, _ = _loader_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="expert offload"):
        loading.load(
            tmp_path, flash1017_enabled=True, moe_expert_offload_resident_fraction=0.5
        )
    assert not models


@pytest.mark.parametrize(
    "kwargs",
    [
        {"flash1017_enabled": "false"},
        {"flash1017_enabled": 1},
        {"attention_output_quantization": "true"},
    ],
)
def test_load_flags_are_strict_booleans(tmp_path, kwargs):
    with pytest.raises(TypeError, match="boolean"):
        loading.load(tmp_path, **kwargs)


def test_load_guard_surrounds_each_normal_shard_read(tmp_path, monkeypatch):
    events, _, _, _ = _loader_fixture(tmp_path, monkeypatch)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"missing": "one.safetensors"}}'
    )
    monkeypatch.setattr(
        loading, "_load_shard", lambda path: events.append("shard") or {}
    )
    with pytest.raises(ValueError, match="Incomplete"):
        loading.load(tmp_path, flash1017_enabled=True)
    assert events == [
        "validate",
        "memory",
        "model",
        "headroom",
        "shard",
        "headroom",
        "close",
    ]
