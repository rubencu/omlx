# SPDX-License-Identifier: Apache-2.0
"""CPU-only runtime contracts: fake tensors, actual ownership/cache controllers.

Run with `python -m unittest discover -s tests -p test_flash1017_runtime.py`.
This intentionally bypasses the GPU initialization in the general conftest.
These fixtures establish control flow and ownership, not MLX numerical parity.
"""

import gc
import hashlib
import importlib
import subprocess
import sys
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

from omlx.patches.deepseek_v41.flash1017 import cone as candidate
from omlx.patches.deepseek_v41.flash1017 import rows, runtime


class Tensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.ndim, self.dtype = tuple(shape), len(shape), dtype

    def __getitem__(self, key):
        shape = list(self.shape)
        for axis, part in enumerate(key if isinstance(key, tuple) else (key,)):
            if isinstance(part, slice):
                shape[axis] = len(range(*part.indices(shape[axis])))
        return Tensor(shape, self.dtype)


class Module(dict):
    """Match MLX Module's Python dict-subclass layout, not its computation."""

    pass


class Projection(Module):
    def __init__(self, *, quantize_input=True):
        self.quantize_input, self.mode, self.bits, self.group_size = (
            quantize_input,
            "mxfp8",
            8,
            32,
        )
        self.input_dims = 5120
        self.weight = Tensor((512, 1280))
        self.calls = []
        self.fail = False

    def project_quantized(self, x, indices=None, sorted_indices=False, block_plan=None):
        self.calls.append(x.shape[1])
        if self.fail:
            raise ValueError("projection failure")
        return Tensor((*x.shape[:2], 512), x.dtype)

    def __call__(self, x, indices=None, sorted_indices=False, block_plan=None):
        return self.project_quantized(x, indices, sorted_indices, block_plan)


def hc_mixes(*args):
    return "native-hc"


class Gate(Module):
    def __init__(self, config):
        self._config = config
        self.rows = []

    def __call__(self, x, image_mask):
        self.rows.append(x.shape[1])
        return Tensor((*x.shape[:2], 6)), Tensor((*x.shape[:2], 6))


class MoE(Module):
    def __init__(self, config):
        self.gate = Gate(config)

    def __call__(self, x, image_mask):
        self.gate(x, image_mask)
        return x


class Block(Module):
    def __init__(self, config, i):
        self._config, self.layer = config, i
        self.attn = SimpleNamespace(
            wkv=Projection(),
            indexer=SimpleNamespace(weights_proj=Projection(quantize_input=False)),
        )
        self.ffn = MoE(config)

    def __call__(self, h, pre, cache, shared, start, image_mask, ced_tail=None):
        hc_mixes(h, Tensor((24, 20480), "f32"), None, None, self._config)
        if ced_tail is not None:
            h, pre = h[:, -ced_tail:], pre[:, -ced_tail:]
        self.attn.wkv(h)
        if self.layer in (20, 24, 28, 32, 36):
            self.attn.indexer.weights_proj(h)
        return self.ffn(h, image_mask), pre


class Cache:
    def __init__(self, start=0):
        self.offset, self.left_padding, self.lengths = start, None, None

    def size(self):
        return self.offset


class LanguageModel(Module):
    def __init__(self):
        self._config = SimpleNamespace(
            n_layers=40,
            window_size=128,
            kv_source_layers=(2, 8, 14, 20),
            index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36),
            engram_layer_ids=(1, 14),
            dspark_target_layer_ids=(37, 38, 39),
            candidate_source_layer=20,
            candidate_block_size=8,
            candidate_topk_blocks=2048,
            ced_prefill=False,
            ced_layout_supported=lambda: True,
            n_routed_experts=384,
            n_activated_experts=6,
            hc_mult=4,
        )
        self.layers = [Block(self._config, i) for i in range(40)]
        self.mtp = [
            SimpleNamespace(attn=SimpleNamespace(wkv=Projection())) for _ in range(3)
        ]
        self._omlx_dspark_decode_enabled = True
        self.calls, self.appends = [], []
        self.fail = self.swallow = self.omit_prime = self.reenter = False
        self.enter_event = self.exit_event = None

    def _omlx_prefill(self, input_ids, cache=None, **kwargs):
        return self(input_ids, cache=cache, _ced_prefill=True, **kwargs)

    def __call__(
        self, input_ids, cache=None, inputs_embeds=None, token_types=None, **kwargs
    ):
        if self.enter_event:
            self.enter_event.set()
            assert self.exit_event.wait(2)
        if self.reenter:
            return self(input_ids, cache=cache)
        if self.fail:
            raise ValueError("forward failure")
        cache = cache or [Cache() for _ in range(40)]
        n, start = input_ids.shape[1], cache[0].size()
        self.calls.append(
            (n, self._config.ced_prefill, inputs_embeds, token_types, dict(kwargs))
        )
        h = pre = Tensor((1, n, 5120))
        shared = {}
        for i, block in enumerate(self.layers):
            ced = (
                self._config.ced_prefill
                and kwargs.get("_ced_prefill", False)
                and i >= 20
            )
            absolute_start = start + n - 128 if ced and i > 20 else start
            h, pre = block(
                h,
                pre,
                cache[i],
                shared,
                absolute_start,
                None,
                ced_tail=128 if ced else None,
            )
            cache[i].offset = start + n
        if self._omlx_dspark_decode_enabled and not self.omit_prime:
            try:
                self.dspark_append_context(
                    Tensor((1, h.shape[1], 15360)),
                    [Cache() for _ in range(3)],
                    start_offset=start + n - h.shape[1],
                )
            except Exception:
                if not self.swallow:
                    raise
        return Tensor((1, h.shape[1], 129280))

    def dspark_append_context(self, hidden, cache, *, start_offset=None):
        self.appends.append((hidden.shape[1], start_offset))
        for stage in self.mtp:
            out = stage.attn.wkv(Tensor((1, hidden.shape[1], 5120)))
            assert out.shape[1] == hidden.shape[1]
        return "native-append"

    def forward_spec(self, ids, main_hidden, cache, **kwargs):
        return self.mtp[0].attn.wkv(main_hidden)

    def configure_mtp(self, enabled, depth=5):
        self._omlx_dspark_decode_enabled = bool(enabled)


class Adapter:
    def __init__(self, model):
        self.model = model

    def _omlx_prefill(self, ids, cache=None, **kwargs):
        # Mirrors VLMModelAdapter, which bypasses inner _omlx_prefill.
        return self.model(ids, cache=cache, _ced_prefill=True, **kwargs)


def bindings():
    def concat(arrays, axis=1):
        shape = list(arrays[0].shape)
        shape[axis] = sum(x.shape[axis] for x in arrays)
        return Tensor(shape, arrays[0].dtype)

    return SimpleNamespace(
        mx=SimpleNamespace(
            concatenate=concat,
            zeros=lambda shape, dtype: Tensor(shape, dtype),
            bfloat16="bf16",
            float32="f32",
        ),
        language=SimpleNamespace(
            LanguageModel=LanguageModel,
            Block=Block,
            Gate=Gate,
            MoE=MoE,
            hc_mixes=hc_mixes,
        ),
        hyper_connection=SimpleNamespace(),
        QuantizedProjection=Projection,
    )


def cache(start=0):
    return [Cache(start) for _ in range(40)]


@contextmanager
def fake_arithmetic():
    """Import normal arithmetic modules with fake dependencies, then restore."""
    from omlx.patches import deepseek_v41 as package
    from omlx.patches.deepseek_v41 import flash1017

    b = bindings()
    fake_mlx = ModuleType("mlx")
    fake_mlx.core = b.mx
    modules = {
        "mlx": fake_mlx,
        "mlx.core": b.mx,
        "omlx.patches.deepseek_v41.language": b.language,
        "omlx.patches.deepseek_v41.hyper_connection": b.hyper_connection,
        "omlx.patches.deepseek_v41.quantization": SimpleNamespace(
            QuantizedProjection=b.QuantizedProjection
        ),
    }
    imported = []
    with ExitStack() as stack:
        stack.enter_context(mock.patch.dict(sys.modules, modules))
        stack.enter_context(
            mock.patch.object(package, "language", b.language, create=True)
        )
        stack.enter_context(
            mock.patch.object(
                package, "hyper_connection", b.hyper_connection, create=True
            )
        )
        for name in ("cone_arithmetic", "shape_arithmetic", "prime_arithmetic"):
            full = flash1017.__name__ + "." + name
            previous = sys.modules.get(full)
            module = importlib.import_module(full)
            if previous is None:
                imported.append((full, name))
            for key in ("mx", "language", "hyper_connection", "QuantizedProjection"):
                if hasattr(module, key):
                    stack.enter_context(mock.patch.object(module, key, getattr(b, key)))
        try:
            yield b
        finally:
            # Do not leave fake modules behind for other test files.
            for full, name in imported:
                sys.modules.pop(full, None)
                if hasattr(flash1017, name):
                    delattr(flash1017, name)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.enterContext(fake_arithmetic())

    def install(self, model=None, **kwargs):
        model = model or LanguageModel()
        owner = candidate.install(
            model, bindings=bindings(), record_events=True, **kwargs
        )
        self.addCleanup(owner.close)
        return model, owner

    def test_import_provenance_no_mlx_and_source_pins(self):
        self.assertEqual(len(candidate.provenance()), 3)
        code = "import sys; import omlx.patches.deepseek_v41.flash1017; assert 'mlx.core' not in sys.modules"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_direct_and_adapter_prefill_rows_endpoints(self):
        model, owner = self.install()
        for entry, n, start in (
            (model._omlx_prefill, 8192, 0),
            (Adapter(model)._omlx_prefill, 8191, 24576),
        ):
            result = entry(Tensor((1, n)), cache=cache(start))
            self.assertEqual(result.shape, (1, 128, 129280))
            trace = owner.snapshot()["trace"][-1]
            self.assertEqual(trace["cone"][0]["end"], start + n)
            event = trace["prime"][0]
            self.assertTrue(event["append_complete"])
            self.assertEqual(event["append_start"], start + n - 128)
            self.assertEqual([x["stage"] for x in event["stages"]], [0, 1, 2])
            for stage in model.mtp:
                self.assertEqual(stage.attn.wkv.calls[-1], n)
            self.assertFalse(model._config.ced_prefill)

    def test_disabled_native_toggle_and_ordinary_decode(self):
        model, owner = self.install(enabled=False)
        self.assertEqual(
            Adapter(model)._omlx_prefill(Tensor((1, 8192)), cache=cache()).shape[1],
            8192,
        )
        owner.set_enabled(True)
        self.assertEqual(
            model._omlx_prefill(Tensor((1, 8192)), cache=cache()).shape[1], 128
        )
        self.assertEqual(model(Tensor((1, 3)), cache=cache()).shape[1], 3)
        self.assertEqual(model.mtp[-1].attn.wkv.calls[-1], 3)
        self.assertEqual(
            model.forward_spec(None, Tensor((1, 5, 5120)), None).shape[1], 5
        )

    def test_short_vision_unknown_missing_cache_fallbacks(self):
        model, owner = self.install()
        for n, kwargs, c in (
            (2667, {}, cache()),
            (8192, {"inputs_embeds": Tensor((1, 8192, 5120))}, cache()),
            (8192, {"token_types": Tensor((1, 8192))}, cache()),
            (8192, {"new_option": True}, cache()),
            (8192, {}, None),
        ):
            self.assertEqual(
                model._omlx_prefill(Tensor((1, n)), cache=c, **kwargs).shape[1], n
            )
        self.assertEqual(owner.snapshot()["cone_counts"].get("cone_calls", 0), 0)

    def test_sequential_thread_migration(self):
        model, owner = self.install()
        with ThreadPoolExecutor(1) as a, ThreadPoolExecutor(1) as b:
            ids = {
                a.submit(threading.get_ident).result(),
                b.submit(threading.get_ident).result(),
            }
            self.assertEqual(len(ids), 2)
            for worker in (a, b, a, b):
                self.assertEqual(
                    worker.submit(model._omlx_prefill, Tensor((1, 8192)), cache=cache())
                    .result()
                    .shape[1],
                    128,
                )

    def test_overlap_and_mutation_rejected_before_native_call(self):
        model, owner = self.install()
        model.enter_event, model.exit_event = threading.Event(), threading.Event()
        with ThreadPoolExecutor(1) as worker:
            future = worker.submit(
                model._omlx_prefill, Tensor((1, 8192)), cache=cache()
            )
            self.assertTrue(model.enter_event.wait(2))
            for action in (
                lambda: model(Tensor((1, 1)), cache=cache()),
                lambda: owner.set_enabled(False),
                owner.close,
                owner.snapshot,
                lambda: model.forward_spec(None, Tensor((1, 2, 5120)), None),
            ):
                with self.assertRaises(RuntimeError):
                    action()
            model.exit_event.set()
            self.assertEqual(future.result().shape[1], 128)

    def test_reentrant_call_failure_cleanup_and_recovery(self):
        model, owner = self.install()
        model.reenter = True
        with self.assertRaises(RuntimeError):
            model._omlx_prefill(Tensor((1, 8192)), cache=cache())
        model.reenter = False
        self.assertFalse(model._config.ced_prefill)
        self.assertEqual(
            model._omlx_prefill(Tensor((1, 8192)), cache=cache()).shape[1], 128
        )

    def test_swallowed_prime_error_and_missing_capture_fail_visible(self):
        model, owner = self.install()
        model.swallow, model.mtp[1].attn.wkv.fail = True, True
        with self.assertRaisesRegex(RuntimeError, "priming failed"):
            model._omlx_prefill(Tensor((1, 8192)), cache=cache())
        model.mtp[1].attn.wkv.fail = False
        model.omit_prime = True
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            model._omlx_prefill(Tensor((1, 8192)), cache=cache())
        model.omit_prime = False
        self.assertEqual(
            model._omlx_prefill(Tensor((1, 8192)), cache=cache()).shape[1], 128
        )

    def test_no_dspark_allowed_and_native_direct_append(self):
        model, owner = self.install()
        model.configure_mtp(False)
        self.assertEqual(
            model._omlx_prefill(Tensor((1, 8192)), cache=cache()).shape[1], 128
        )
        model.dspark_append_context(
            Tensor((1, 128, 15360)), [Cache() for _ in range(3)], start_offset=0
        )
        self.assertEqual(model.mtp[0].attn.wkv.calls[-1], 128)

    def test_unrelated_model_and_class_globals_untouched(self):
        before = (
            LanguageModel.__call__,
            Block.__call__,
            Gate.__call__,
            MoE.__call__,
            Projection.__call__,
            Projection.project_quantized,
            hc_mixes,
        )
        first, owner = self.install()
        second, other = self.install()
        unrelated = LanguageModel()
        self.assertEqual(
            unrelated._omlx_prefill(Tensor((1, 8192)), cache=cache()).shape[1], 8192
        )
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(m._omlx_prefill, Tensor((1, 8192)), cache=cache())
                for m in (first, second)
            ]
            self.assertEqual([f.result().shape[1] for f in futures], [128, 128])
        after = (
            LanguageModel.__call__,
            Block.__call__,
            Gate.__call__,
            MoE.__call__,
            Projection.__call__,
            Projection.project_quantized,
            hc_mixes,
        )
        self.assertEqual(before, after)
        self.assertNotIn("_flash1017_cone_owner", first)

    def test_class_restore_parameter_identity_and_bounded_trace(self):
        model, owner = self.install()
        weight = model.layers[20].attn.wkv.weight
        for _ in range(20):
            model(Tensor((1, 1)), cache=cache())
        self.assertEqual(len(owner.snapshot()["trace"]), 16)
        owner.snapshot(clear=True)
        self.assertEqual(owner.snapshot()["trace"], [])
        owner.close()
        self.assertIs(type(model), LanguageModel)
        self.assertIs(type(model.layers[20]), Block)
        self.assertIs(type(model.layers[20].attn.wkv), Projection)
        self.assertIs(weight, model.layers[20].attn.wkv.weight)
        self.assertFalse(hasattr(model, "_flash1017_cone_owner"))
        self.assertEqual(
            model._omlx_prefill(Tensor((1, 8192)), cache=cache()).shape[1], 8192
        )

    def test_layout_guard_duplicate_and_native_ced_guard(self):
        model, owner = self.install()
        with self.assertRaises(RuntimeError):
            candidate.install(model, bindings=bindings())
        model._config.ced_prefill = True
        with self.assertRaises(RuntimeError):
            model(Tensor((1, 1)), cache=cache())
        model._config.ced_prefill = False
        invalid = LanguageModel()
        invalid._config.window_size = 64
        with self.assertRaises(ValueError):
            candidate.install(invalid, bindings=bindings())

    def test_foreign_class_refuses_without_partial_teardown(self):
        model, owner = self.install()
        installed = type(model)
        foreign = type("Foreign", (installed,), {"__slots__": ()})
        object.__setattr__(model, "__class__", foreign)
        before = [type(obj) for obj, _, _ in owner._saved]
        with self.assertRaisesRegex(RuntimeError, "class changed"):
            owner.close()
        self.assertEqual(before, [type(obj) for obj, _, _ in owner._saved])
        self.assertFalse(owner.closed)
        object.__setattr__(model, "__class__", installed)
        owner.close()

    def test_strict_bool_options(self):
        model, owner = self.install()
        for invalid in (0, 1, None, "false"):
            with self.assertRaises(TypeError):
                owner.set_enabled(invalid)
            with self.assertRaises(TypeError):
                candidate.install(LanguageModel(), enabled=invalid, bindings=bindings())

    def test_failure_cleanup_preserves_request_cache_boundary(self):
        model, owner = self.install()
        model.fail = True
        failed_cache = cache(1000)
        with self.assertRaises(ValueError):
            model._omlx_prefill(Tensor((1, 8192)), cache=failed_cache)
        self.assertEqual(failed_cache[0].size(), 1000)
        self.assertIsNone(owner.prime.frame)
        self.assertIsNone(owner.shape.frame)
        self.assertIsNone(owner.cone.frame)
        self.assertFalse(model._config.ced_prefill)
        model.fail = False
        self.assertEqual(
            model._omlx_prefill(Tensor((1, 8192)), cache=cache(29000)).shape[1], 128
        )

    def test_batched_padded_limited_capture_verify_fallback(self):
        model, owner = self.install()
        padded = cache()
        padded[0].left_padding = SimpleNamespace(
            reshape=lambda _: SimpleNamespace(tolist=lambda: [1])
        )
        limited = cache()
        limited[0].lengths = Tensor((1,))
        for ids, c, kw in (
            (Tensor((2, 8192)), cache(), {}),
            (Tensor((1, 8192)), padded, {}),
            (Tensor((1, 8192)), limited, {}),
            (Tensor((1, 8192)), cache(), {"return_hidden": True}),
            (Tensor((1, 8192)), cache(), {"return_dspark_hidden": True}),
            (Tensor((1, 8192)), cache(), {"target_verify": True}),
        ):
            self.assertEqual(model._omlx_prefill(ids, cache=c, **kw).shape[1], 8192)
        self.assertEqual(owner.cone.counts["cone_calls"], 0)

    def test_close_releases_model_and_keeps_plain_diagnostics(self):
        model, owner = self.install()
        model._omlx_prefill(Tensor((1, 8192)), cache=cache())
        reference = weakref.ref(model)
        owner.close()
        del model
        gc.collect()
        self.assertIsNone(reference())
        summary = owner.snapshot()
        self.assertTrue(summary["closed"])
        self.assertTrue(summary["model_references_released"])
        self.assertEqual(summary["prime_counts"]["padded_appends"], 1)
        self.assertEqual(summary["owned_instances"], 0)

    def test_prefill_trace_survives_decode_and_reports_eviction(self):
        model, owner = self.install()
        for _ in range(17):
            model._omlx_prefill(Tensor((1, 2668)), cache=cache())
        for _ in range(20):
            model(Tensor((1, 1)), cache=cache())
        summary = owner.snapshot()
        self.assertEqual(len(summary["prefill_trace"]), 16)
        self.assertEqual(summary["counts"]["prefill_trace_evictions"], 1)
        self.assertTrue(
            all(t["prime"][0]["append_complete"] for t in summary["prefill_trace"])
        )
        owner.snapshot(clear=True)
        self.assertEqual(owner.snapshot()["prefill_trace"], [])


class Prefetch:
    def __init__(self):
        self._closed = False
        self._pending = None
        self.close_count = self.drain_count = 0
        self.fail_close = self.fail_drain = False

    @contextmanager
    def forward(self):
        try:
            yield
        finally:
            self.drain()

    def submit(self, *args, **kwargs):
        return None

    def drain(self):
        self.drain_count += 1
        if self.fail_drain:
            raise OSError("prefetch drain failure")

    def close(self):
        self.close_count += 1
        self._closed = True
        if self.fail_close:
            raise OSError("prefetch close failure")


class Embedding:
    def __init__(self, salt=0):
        self._lock = threading.RLock()
        self._closed = False
        self._resident = self._prefetched = None
        self._weight_key, self._scale_key, self._bias_key = "weight", None, None
        self.values = np.arange(128, dtype=np.uint32).reshape(32, 4) + salt
        self._weights = self._scales = SimpleNamespace(
            header={"weight": {"shape": self.values.shape}}
        )
        self.read_count = self.close_count = 0
        self.fail_close = False

    def _read_rows(self, host):
        with self._lock:
            if self._closed:
                raise RuntimeError("table closed")
            self.read_count += 1
            return [(self.values[np.asarray(host).reshape(-1)].copy(), "U32")]

    def selected_bytes(self, n):
        return int(n) * self.values.shape[1] * 4

    def close(self):
        with self._lock:
            self.close_count += 1
            self._closed = True
            if self.fail_close:
                raise OSError("table close failure")

    def make_resident(self):
        self._resident = self.values


class Outer:
    def __init__(self):
        self.language_model = LanguageModel()
        self.language_model._engram_prefetch = Prefetch()
        for i in (1, 14):
            self.language_model.layers[i].engram = SimpleNamespace(embed=Embedding(i))
        self.close_count = 0
        self.fail_close = False

    @property
    def tables(self):
        return [self.language_model.layers[i].engram.embed for i in (1, 14)]

    @property
    def prefetch(self):
        return self.language_model._engram_prefetch

    def close(self):
        self.close_count += 1
        if self.fail_close:
            raise OSError("outer close failure")
        try:
            self.prefetch.close()
        finally:
            for table in self.tables:
                table.close()


def fixture_hooks():
    return (
        LanguageModel.__call__,
        Block.__call__,
        Projection.__call__,
        Projection.project_quantized,
        Gate.__call__,
        MoE.__call__,
        hc_mixes,
        Embedding._read_rows,
        Embedding.close,
        Outer.close,
    )


def fixture_parameters(model):
    return tuple(id(block.attn.wkv.weight) for block in model.language_model.layers)


class CombinedLifecycle(unittest.TestCase):
    def setUp(self):
        self.enterContext(fake_arithmetic())
        self.enterContext(mock.patch.object(runtime, "native_hooks", fixture_hooks))
        self.enterContext(
            mock.patch.object(runtime, "parameter_identity", fixture_parameters)
        )
        self.enterContext(
            mock.patch.object(
                runtime,
                "_install_cone",
                lambda target, **kw: candidate.install(
                    target, bindings=bindings(), **kw
                ),
            )
        )

        def small_rows(model, **kw):
            # Bound fixture allocations, retaining the production ownership path.
            kw["budget_per_table"] = 4096
            return rows.install(model, embedding_type=Embedding, **kw)

        self.enterContext(mock.patch.object(runtime, "_install_rows", small_rows))

    def install(self, **kwargs):
        model = Outer()
        owner = runtime.install(model, **kwargs)
        self.addCleanup(owner.close)
        return model, owner

    def test_combined_toggle_and_detach_preserve_native_model(self):
        model, owner = self.install(enabled=False, record_events=True)
        self.assertFalse(owner.snapshot()["enabled"])
        owner.enable()
        self.assertEqual(
            model.language_model._omlx_prefill(Tensor((1, 8192)), cache=cache()).shape[
                1
            ],
            128,
        )
        self.assertEqual(owner.snapshot()["cone"]["prime_counts"]["padded_appends"], 1)
        owner.set_enabled(False)
        self.assertFalse(owner.snapshot()["rows"]["enabled"])
        owner.close()
        owner.close()
        self.assertTrue(owner.snapshot()["closed"])
        self.assertIsNone(owner.model)
        self.assertIsNone(owner.target)
        self.assertIs(type(model.language_model), LanguageModel)
        self.assertNotIn("close", vars(model))
        self.assertFalse(model.prefetch._closed)
        model.close()
        self.assertTrue(all(table._closed for table in model.tables))

    def test_normal_outer_close_detaches_both_owners(self):
        model, owner = self.install()
        model.close()
        summary = owner.snapshot()
        self.assertTrue(owner.closed)
        self.assertEqual(summary["rows"]["status"], "closed")
        self.assertEqual(summary["rows"]["retained_bytes"], 0)
        self.assertTrue(model.prefetch._closed)
        self.assertTrue(all(table._closed for table in model.tables))
        self.assertIs(type(model.language_model), LanguageModel)
        self.assertFalse(hasattr(model, runtime._MARKER))
        self.assertFalse(hasattr(model.language_model, "_flash1017_cone_owner"))
        self.assertEqual(fixture_hooks(), owner.original_hooks or fixture_hooks())

    def test_shutdown_failure_still_closes_every_table_and_releases(self):
        for failure in ("outer", "prefetch", "table"):
            with self.subTest(failure=failure):
                model, owner = self.install()
                if failure == "outer":
                    model.fail_close = True
                if failure == "prefetch":
                    model.prefetch.fail_close = True
                if failure == "table":
                    model.tables[0].fail_close = True
                with self.assertRaises(OSError):
                    model.close()
                self.assertTrue(owner.closed)
                self.assertTrue(model.prefetch._closed)
                self.assertTrue(all(table._closed for table in model.tables))
                self.assertEqual(owner.snapshot()["rows"]["status"], "closed")
                self.assertIs(type(model.language_model), LanguageModel)

    def test_drain_error_releases_caches_and_classes_on_detach(self):
        model, owner = self.install()
        model.prefetch.fail_drain = True
        with self.assertRaisesRegex(OSError, "drain failure"):
            owner.close()
        self.assertTrue(owner.closed)
        self.assertEqual(owner.snapshot()["rows"]["status"], "detached")
        self.assertIs(type(model.language_model), LanguageModel)

    def test_foreign_outer_wrapper_refuses_before_any_partial_detach(self):
        model, owner = self.install()
        installed = model.close
        object.__setattr__(model, "close", lambda: None)
        with self.assertRaisesRegex(RuntimeError, "runtime attribute"):
            owner.close()
        self.assertFalse(owner.closed)
        self.assertEqual(owner.rows.status, "active")
        self.assertFalse(owner.cone.closed)
        object.__setattr__(model, "close", installed)
        owner.close()

    def test_foreign_table_wrapper_refuses_before_any_partial_detach(self):
        model, owner = self.install()
        table = model.tables[0]
        installed = table._read_rows
        object.__setattr__(table, "_read_rows", lambda ids: [])
        with self.assertRaisesRegex(RuntimeError, "instance attribute"):
            model.close()
        self.assertFalse(owner.closed)
        self.assertFalse(model.prefetch._closed)
        self.assertFalse(owner.cone.closed)
        object.__setattr__(table, "_read_rows", installed)
        model.close()

    def test_row_install_failure_restores_cone_classes(self):
        model = Outer()
        with (
            mock.patch.object(
                runtime, "_install_rows", side_effect=ValueError("row install failure")
            ),
            self.assertRaisesRegex(ValueError, "row install failure"),
        ):
            runtime.install(model)
        self.assertIs(type(model.language_model), LanguageModel)
        self.assertNotIn("close", vars(model))
        self.assertFalse(hasattr(model.language_model, "_flash1017_cone_owner"))
        model.close()

    def test_row_callback_and_forward_close_refuse_before_detach(self):
        model, owner = self.install()
        owner.rows._read_local.depth = 1
        try:
            with self.assertRaisesRegex(RuntimeError, "row-read callback"):
                model.close()
        finally:
            owner.rows._read_local.depth = 0
        self.assertFalse(owner.cone.closed)
        self.assertEqual(owner.rows.status, "active")
        self.assertTrue(hasattr(model, runtime._MARKER))
        with (
            model.prefetch.forward(),
            self.assertRaisesRegex(RuntimeError, "drained forward boundary"),
        ):
            model.close()
        self.assertFalse(owner.cone.closed)
        self.assertFalse(model.prefetch._closed)
        model.close()

    def test_enable_failure_restores_every_owner(self):
        model = Outer()
        model.prefetch.fail_drain = True
        with self.assertRaisesRegex(OSError, "drain failure"):
            runtime.install(model)
        self.assertIs(type(model.language_model), LanguageModel)
        self.assertNotIn("close", vars(model))
        self.assertFalse(hasattr(model, runtime._MARKER))
        self.assertFalse(hasattr(model.tables[0], rows._MARKER))

    def test_active_call_refuses_unload_and_policy_change(self):
        model, owner = self.install()
        target = model.language_model
        target.enter_event, target.exit_event = threading.Event(), threading.Event()
        with ThreadPoolExecutor(1) as worker:
            future = worker.submit(
                target._omlx_prefill, Tensor((1, 8192)), cache=cache()
            )
            self.assertTrue(target.enter_event.wait(2))
            try:
                for action in (
                    model.close,
                    owner.close,
                    owner.snapshot,
                    lambda: owner.enable(False),
                ):
                    with self.assertRaisesRegex(RuntimeError, "Overlapping"):
                        action()
                self.assertFalse(model.prefetch._closed)
            finally:
                target.exit_event.set()
            self.assertEqual(future.result().shape[1], 128)
        model.close()

    def test_cached_duplicate_rows_remain_byte_identical(self):
        model, owner = self.install()
        table = model.tables[0]
        ids = np.array([7, 3, 7, 1, 3], dtype=np.int64)
        for _ in range(4):
            actual = table._read_rows(ids)
            self.assertEqual(actual[0][1], "U32")
            self.assertEqual(actual[0][0].tobytes(), table.values[ids].tobytes())
        self.assertGreater(
            owner.snapshot()["rows"]["live_tables"][0]["counts"][
                "duplicate_rows_elided"
            ],
            0,
        )
        self.assertLessEqual(owner.snapshot()["rows"]["retained_bytes"], 8192)

    def test_closed_controller_does_not_retain_model(self):
        model, owner = self.install()
        reference = weakref.ref(model)
        model.close()
        del model
        gc.collect()
        self.assertIsNone(reference())
        summary = owner.snapshot()
        self.assertTrue(summary["model_references_released"])
        summary["closed"] = False
        self.assertTrue(owner.snapshot()["closed"])

    def test_duplicate_install_rejected_and_plain_hooks_unchanged(self):
        before = fixture_hooks()
        model, owner = self.install()
        with self.assertRaisesRegex(RuntimeError, "already"):
            runtime.install(model)
        self.assertEqual(fixture_hooks(), before)
        model.close()
        self.assertEqual(fixture_hooks(), before)


class StorageOnlyLifecycle(unittest.TestCase):
    setUp = CombinedLifecycle.setUp

    def install(self, **kwargs):
        return CombinedLifecycle.install(self, replay=False, **kwargs)

    def test_no_inference_class_or_method_is_wrapped(self):
        model = Outer()
        target = model.language_model
        objects = [target, *target.layers, *target.mtp]
        objects += [block.attn.wkv for block in target.layers]
        objects += [stage.attn.wkv for stage in target.mtp]
        original_types = [type(obj) for obj in objects]
        original_parameters = fixture_parameters(model)
        native_call = type(target).__call__
        native_prefill = target._omlx_prefill.__func__
        native_append = target.dspark_append_context.__func__
        with mock.patch.object(runtime, "_install_cone", side_effect=AssertionError("Replay installed")):
            owner = runtime.install(model, replay=False)
        self.addCleanup(owner.close)
        self.assertEqual([type(obj) for obj in objects], original_types)
        self.assertIs(type(target).__call__, native_call)
        self.assertIs(target._omlx_prefill.__func__, native_prefill)
        self.assertIs(target.dspark_append_context.__func__, native_append)
        self.assertEqual(fixture_parameters(model), original_parameters)
        self.assertFalse(hasattr(target, "_flash1017_cone_owner"))
        self.assertFalse(hasattr(target, "_flash1017_experiment"))
        self.assertEqual(target._omlx_prefill(Tensor((1, 8192)), cache=cache()).shape[1], 8192)
        snapshot = owner.snapshot(clear_cone=True)
        self.assertFalse(snapshot["replay"])
        self.assertIsNone(snapshot["cone"])
        self.assertTrue(snapshot["rows"]["enabled"])
        owner.set_enabled(False)
        self.assertFalse(owner.snapshot()["enabled"])
        self.assertFalse(owner.snapshot()["rows"]["enabled"])
        owner.set_enabled(True)
        model.close()
        self.assertIsNone(owner.snapshot()["cone"])
        self.assertEqual([type(obj) for obj in objects], original_types)

    def test_storage_only_shutdown_releases_after_native_errors(self):
        for failure in ("outer", "prefetch", "table"):
            with self.subTest(failure=failure):
                model, owner = self.install()
                if failure == "outer":
                    model.fail_close = True
                elif failure == "prefetch":
                    model.prefetch.fail_close = True
                else:
                    model.tables[0].fail_close = True
                with self.assertRaises(OSError):
                    model.close()
                self.assertTrue(owner.closed)
                self.assertTrue(model.prefetch._closed)
                self.assertTrue(all(table._closed for table in model.tables))
                self.assertEqual(owner.snapshot()["rows"]["retained_bytes"], 0)
                self.assertTrue(owner.snapshot()["model_references_released"])

    def test_storage_controls_reject_reader_and_same_forward_before_mutation(self):
        model, owner = self.install()
        owner.rows._read_local.depth = 1
        try:
            for action in (model.close, owner.close, owner.snapshot, lambda: owner.set_enabled(False)):
                with self.assertRaisesRegex(RuntimeError, "row-read callback"):
                    action()
        finally:
            owner.rows._read_local.depth = 0
        with model.prefetch.forward():
            for action in (model.close, owner.close, owner.snapshot, lambda: owner.set_enabled(False)):
                with self.assertRaisesRegex(RuntimeError, "drained forward boundary"):
                    action()
        self.assertFalse(owner.closed)
        self.assertFalse(model.prefetch._closed)
        model.close()

    def test_storage_foreign_wrapper_refuses_before_partial_release(self):
        model, owner = self.install()
        original = model.tables[0]._read_rows
        object.__setattr__(model.tables[0], "_read_rows", lambda ids: [])
        try:
            with self.assertRaisesRegex(RuntimeError, "instance attribute"):
                model.close()
            self.assertFalse(owner.closed)
            self.assertFalse(model.prefetch._closed)
            self.assertTrue(hasattr(model, runtime._MARKER))
        finally:
            object.__setattr__(model.tables[0], "_read_rows", original)
        model.close()

    def test_reader_refusal_precedes_waiting_for_controller_lock(self):
        model, owner = self.install()

        def callback(action):
            owner.rows._read_local.depth = 1
            try:
                with self.assertRaisesRegex(RuntimeError, "row-read callback"):
                    action()
            finally:
                owner.rows._read_local.depth = 0

        with ThreadPoolExecutor(1) as worker:
            for action in (model.close, owner.close, owner.snapshot, lambda: owner.set_enabled(False)):
                # A shutdown can hold this lock while its drain waits for the
                # reader. The callback must refuse without waiting for it.
                with owner._lock:
                    future = worker.submit(callback, action)
                    future.result(timeout=2)
        model.close()

    def test_storage_install_failure_does_not_leave_ownership(self):
        model = Outer()
        with mock.patch.object(runtime, "_install_rows", side_effect=ValueError("rows failed")):
            with self.assertRaisesRegex(ValueError, "rows failed"):
                runtime.install(model, replay=False)
        self.assertNotIn("close", vars(model))
        self.assertFalse(hasattr(model, runtime._MARKER))
        model.close()

    def test_replay_requires_a_literal_boolean(self):
        for value in (None, 0, 1, "false"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                runtime.install(Outer(), replay=value)

    test_cached_duplicate_rows_remain_byte_identical = CombinedLifecycle.test_cached_duplicate_rows_remain_byte_identical
    test_closed_controller_does_not_retain_model = CombinedLifecycle.test_closed_controller_does_not_retain_model
    test_duplicate_install_rejected_and_plain_hooks_unchanged = CombinedLifecycle.test_duplicate_install_rejected_and_plain_hooks_unchanged
    test_drain_error_releases_caches_and_classes_on_detach = CombinedLifecycle.test_drain_error_releases_caches_and_classes_on_detach


class PackageBoundary(unittest.TestCase):
    def test_hot_rows_preserved_verbatim(self):
        self.assertEqual(
            hashlib.sha256(
                Path(rows.__file__).with_name("hot_rows.py").read_bytes()
            ).hexdigest(),
            rows.UPSTREAM_SHA256,
        )
        self.assertTrue(Path(rows.__file__).with_name("HOT_ROWS_LICENSE").is_file())

    def test_runtime_has_no_external_lab_import_or_dynamic_execution(self):
        import ast

        package = Path(runtime.__file__).parent
        names = (
            "runtime.py",
            "cone.py",
            "rows.py",
            "cone_arithmetic.py",
            "shape_arithmetic.py",
            "prime_arithmetic.py",
        )
        for name in names:
            source = (package / name).read_text()
            tree = ast.parse(source)
            calls = [
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            ]
            self.assertNotIn("exec", calls)
            self.assertNotIn("eval", calls)
            self.assertNotIn("work/", source)
            self.assertNotIn("from evidence", source)


if __name__ == "__main__":
    unittest.main()
