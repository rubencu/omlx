# SPDX-License-Identifier: Apache-2.0
"""Opt-in, per-model dependency-cone and shape-restoration controller.

Installed before exposing the model to inference. No process-wide class or
module methods change. The integration owner drains the worker before unload.
"""

import copy
import sys
import threading
from collections import Counter, deque
from contextlib import contextmanager
from types import FunctionType, SimpleNamespace


def provenance():
    # Static provenance of the arithmetic port; no laboratory files are needed.
    return {
        "cone": "cone_arithmetic.py",
        "shape": "shape_arithmetic.py",
        "prime": "prime_arithmetic.py",
    }


def _bindings():
    """Resolve modules the caller has already loaded; never import MLX here."""
    names = [
        "mlx.core",
        "omlx.patches.deepseek_v41.language",
        "omlx.patches.deepseek_v41.hyper_connection",
        "omlx.patches.deepseek_v41.quantization",
    ]
    try:
        mx, language, hc, quant = [sys.modules[name] for name in names]
    except KeyError as exc:
        raise RuntimeError("Install only after the native model has loaded") from exc
    return SimpleNamespace(
        mx=mx,
        language=language,
        hyper_connection=hc,
        QuantizedProjection=quant.QuantizedProjection,
    )


def _helpers(bindings):
    from .cone_arithmetic import ConeExperiment
    from .prime_arithmetic import PrimeStabilization
    from .shape_arithmetic import ShapeStabilization

    return {
        "ConeExperiment": ConeExperiment,
        "ShapeStabilization": ShapeStabilization,
        "PrimeStabilization": PrimeStabilization,
    }


class Controller:
    def __init__(self, model, *, enabled=True, record_events=False, bindings=None):
        if type(enabled) is not bool or type(record_events) is not bool:
            raise TypeError("enabled and record_events must be bool")
        self.bindings = bindings or _bindings()
        language, qp = self.bindings.language, self.bindings.QuantizedProjection
        if not isinstance(model, language.LanguageModel):
            raise TypeError("install expects the inner DeepSeek V4.1 language_model")
        if getattr(model, "_flash1017_cone_owner", None) is not None:
            raise RuntimeError("This language model already has a cone owner")
        if model._config.ced_prefill:
            raise ValueError("Configure native CED off before attaching the exact cone")
        self.model = model
        self.enabled, self.record_events, self.closed = (
            bool(enabled),
            bool(record_events),
            False,
        )
        self._lock = threading.Lock()
        self._owner_thread = None
        self._saved = []
        self._trace = deque(maxlen=16)
        self._prefill_trace = deque(maxlen=16)
        self._closed_snapshot = None
        self.counts = Counter()
        self.sources = provenance()
        helpers = _helpers(self.bindings)
        self.cone = helpers["ConeExperiment"](model, record_events=record_events)
        if self.cone.layout_reason:
            raise ValueError(
                "Unsupported production layout: " + self.cone.layout_reason
            )
        self.shape = helpers["ShapeStabilization"](
            model,
            layers=(20, 24, 28, 32, 36),
            gate=True,
            hc=True,
            local_wkv=True,
            moe_min256=True,
            record_events=record_events,
        )
        self.prime = helpers["PrimeStabilization"](model, record_events=record_events)
        self.prime.in_prefill = False
        # Ownership belongs to the current admitted call, never installation.
        self.cone._assert_owner = self._assert_owner
        self.shape._owner = self._assert_owner
        self.prime._owner = self._assert_owner
        self.native_call = type(model).__call__
        self.native_append = type(model).dspark_append_context
        self.native_block = language.Block.__call__
        self.native_project = qp.project_quantized
        self.native_projection_call = qp.__call__
        self.native_gate, self.native_moe = (
            language.Gate.__call__,
            language.MoE.__call__,
        )
        # Only this model's blocks see the stabilized HC binding. All other
        # globals and the native function bytecode/closure are retained.
        private_globals = dict(self.native_block.__globals__)
        private_globals["hc_mixes"] = self.shape._hc
        private_block = FunctionType(
            self.native_block.__code__,
            private_globals,
            self.native_block.__name__,
            self.native_block.__defaults__,
            self.native_block.__closure__,
        )
        private_block.__kwdefaults__ = self.native_block.__kwdefaults__
        self.cone._original_call, self.cone._original_block = (
            self.native_call,
            private_block,
        )
        original_reason = self.cone._fallback_reason

        def reason(ids, cache, embeds, token_types, kwargs):
            if cache is None:
                return "missing-cache"
            return original_reason(ids, cache, embeds, token_types, kwargs)

        self.cone._fallback_reason = reason
        self.shape.original_call = self.cone._call
        self.shape.original_projection = self.native_project
        self.shape.original_gate, self.shape.original_moe = (
            self.native_gate,
            self.native_moe,
        )
        self.shape.original_hc = language.hc_mixes
        self.prime.original_prefill = self._prefill_bridge
        self.prime.original_append = self.native_append
        self.prime.original_projection = self.native_projection_call
        try:
            replacements = {
                "__call__": lambda obj, *a, **kw: self._call(obj, *a, **kw),
                "dspark_append_context": lambda obj, *a, **kw: self._append(
                    obj, *a, **kw
                ),
            }
            # These public phases can reach the same projection objects without
            # target __call__. Nested same-thread native phase calls are valid.
            for name in (
                "forward_spec",
                "dspark_forward",
                "dspark_markov",
                "mtp_forward",
                "mtp_take_primed",
                "mtp_partial_rollback",
                "configure_mtp",
            ):
                if hasattr(type(model), name):
                    replacements[name] = self._phase_wrapper(getattr(type(model), name))
            self._attach(model, replacements)
            for block in model.layers:
                self._attach(
                    block, {"__call__": self._leaf(self.cone._block, self.native_block)}
                )
            for block in model.layers[20:]:
                self._attach(
                    block.ffn.gate,
                    {"__call__": self._leaf(self.shape._gate, self.native_gate)},
                )
                self._attach(
                    block.attn.wkv,
                    {
                        "project_quantized": self._leaf(
                            self.shape._projection, self.native_project
                        )
                    },
                )
            for i in (20, 24, 28, 32, 36):
                self._attach(
                    model.layers[i].attn.indexer.weights_proj,
                    {
                        "project_quantized": self._leaf(
                            self.shape._projection, self.native_project
                        )
                    },
                )
            self._attach(
                model.layers[-1].ffn,
                {"__call__": self._leaf(self.shape._moe, self.native_moe)},
            )
            for stage in model.mtp:
                self._attach(
                    stage.attn.wkv,
                    {
                        "__call__": self._leaf(
                            self.prime._projection, self.native_projection_call
                        )
                    },
                )
            object.__setattr__(model, "_flash1017_cone_owner", self)
        except BaseException:
            self._restore()
            raise

    def _attach(self, obj, methods):
        if any(saved[0] is obj for saved in self._saved):
            raise ValueError("Owned modules must not alias across adapter roles")
        base = type(obj)
        derived = type(
            base.__name__ + "_Flash1017Cone",
            (base,),
            dict(methods, __slots__=(), __module__=base.__module__),
        )
        object.__setattr__(obj, "__class__", derived)
        self._saved.append((obj, base, derived))

    def _assert_owner(self):
        if self._owner_thread != threading.get_ident():
            raise RuntimeError("Cone operation outside its admitted model call")

    @contextmanager
    def _admit(self, *, nested=False):
        if nested and self._owner_thread == threading.get_ident():
            yield
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "Overlapping/reentrant calls on one cone model are unsupported"
            )
        try:
            if self.closed:
                raise RuntimeError("Cone controller is closed")
            self._owner_thread = threading.get_ident()
            yield
        finally:
            self._owner_thread = None
            self._lock.release()

    def _leaf(self, corrected, native):
        def call(obj, *args, **kwargs):
            if self._owner_thread is None:
                # Direct leaf use outside a model phase retains native behavior.
                return native(obj, *args, **kwargs)
            self._assert_owner()
            return (corrected if self.enabled else native)(obj, *args, **kwargs)

        return call

    def _phase_wrapper(self, native):
        def phase(obj, *args, **kwargs):
            with self._admit(nested=True):
                return native(obj, *args, **kwargs)

        return phase

    def _prefill_bridge(self, obj, ids, cache=None, **kwargs):
        return self.shape._call(obj, ids, cache=cache, _ced_prefill=True, **kwargs)

    def _call(
        self, obj, ids, cache=None, inputs_embeds=None, token_types=None, **kwargs
    ):
        with self._admit():
            self.counts["model_calls"] += 1
            if obj._config.ced_prefill:
                raise RuntimeError("Native CED changed while cone owner was attached")
            for scope in (self.cone, self.shape, self.prime):
                scope.events.clear()
            try:
                if not self.enabled:
                    self.counts["disabled_calls"] += 1
                    return self.native_call(
                        obj,
                        ids,
                        cache=cache,
                        inputs_embeds=inputs_embeds,
                        token_types=token_types,
                        **kwargs,
                    )
                options = dict(
                    kwargs, inputs_embeds=inputs_embeds, token_types=token_types
                )
                if options.pop("_ced_prefill", False):
                    before = self.cone.counts["cone_calls"]
                    appended = self.prime.counts["padded_appends"]
                    result = self.prime._prefill(obj, ids, cache=cache, **options)
                    if (
                        self.cone.counts["cone_calls"] > before
                        and getattr(obj, "_omlx_dspark_decode_enabled", False)
                        and self.prime.counts["padded_appends"] != appended + 1
                    ):
                        # Includes native capture returning early without an
                        # append. Never silently continue with unprimed state.
                        raise RuntimeError(
                            "Eligible cone prefill did not complete DSpark priming"
                        )
                    return result
                return self.shape._call(obj, ids, cache=cache, **options)
            except BaseException:
                self.counts["call_failures"] += 1
                raise
            finally:
                if self.record_events:
                    event = {
                        name: copy.deepcopy(scope.events)
                        for name, scope in (
                            ("cone", self.cone),
                            ("shape", self.shape),
                            ("prime", self.prime),
                        )
                    }
                    self._trace.append(event)
                    if kwargs.get("_ced_prefill", False):
                        if len(self._prefill_trace) == self._prefill_trace.maxlen:
                            self.counts["prefill_trace_evictions"] += 1
                        self._prefill_trace.append(event)
                if (
                    self.cone.frame is not None
                    or self.shape.frame is not None
                    or self.prime.frame is not None
                    or self.prime.phase is not None
                ):
                    raise RuntimeError("Cone transient frame was not cleared")

    def _append(self, obj, *args, **kwargs):
        with self._admit(nested=True):
            return (self.prime._append if self.enabled else self.native_append)(
                obj, *args, **kwargs
            )

    def snapshot(self, *, clear=False):
        """CPU-only counters and bounded traces, only at a drained boundary."""
        if self.closed:
            return copy.deepcopy(self._closed_snapshot)
        with self._admit():
            out = self._snapshot()
            if clear:
                self._clear_admitted()
            return out

    def _clear_admitted(self):
        self._assert_owner()
        self.counts.clear()
        for scope in (self.cone, self.shape, self.prime):
            scope.counts.clear()
            scope.events.clear()
        self._trace.clear()
        self._prefill_trace.clear()

    def _snapshot(self):
        return {
            "enabled": self.enabled,
            "closed": self.closed,
            "counts": dict(self.counts),
            "cone_counts": dict(self.cone.counts),
            "shape_counts": dict(self.shape.counts),
            "prime_counts": dict(self.prime.counts),
            "trace": copy.deepcopy(list(self._trace)),
            "prefill_trace": copy.deepcopy(list(self._prefill_trace)),
            "sources": copy.deepcopy(self.sources),
            "owned_instances": len(self._saved),
        }

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        with self._admit():
            self.enabled = enabled

    def _validate_restore(self):
        # Validate the complete tree before mutating any class, so foreign
        # wrapping produces a recoverable refusal, never partial teardown.
        for obj, _original, installed in self._saved:
            if type(obj) is not installed:
                raise RuntimeError("Owned module class changed after cone installation")

    def _restore(self):
        self._validate_restore()
        for obj, original, _installed in reversed(self._saved):
            object.__setattr__(obj, "__class__", original)
        self._saved.clear()
        if getattr(self.model, "_flash1017_cone_owner", None) is self:
            object.__delattr__(self.model, "_flash1017_cone_owner")

    def close(self):
        """Restore instance classes, without unloading weights or outer Model.close.

        The integration owner must stop/drain its worker and outstanding MLX
        evaluation before detach; this lock covers Python calls, not GPU fences.
        """
        if self.closed:
            return
        with self._admit():
            self._close_admitted()

    def _close_admitted(self):
        """Detach while the combined controller holds this admission lock."""
        self._assert_owner()
        final = self._snapshot()
        self._restore()
        final.update(closed=True, owned_instances=0, model_references_released=True)
        self._closed_snapshot = final
        # Keep only plain diagnostics after close. Otherwise a retained
        # controller could keep this very large model alive after unload.
        self.model = self.cone = self.shape = self.prime = self.bindings = None
        self.native_call = self.native_append = self.native_block = None
        self.native_project = self.native_projection_call = None
        self.native_gate = self.native_moe = None
        self.closed = True

    detach = close


def install(model, *, enabled=True, record_events=False, bindings=None):
    """Attach to a fresh inner language model; bindings is for CPU fixtures."""
    return Controller(
        model, enabled=enabled, record_events=record_events, bindings=bindings
    )
