# SPDX-License-Identifier: Apache-2.0
"""One lifecycle owner for the opt-in Flash1017 cone and Engram row cache.

Importing this module does not import MLX. Install after the configured model
loads, before its worker is exposed. Controls require a drained request boundary;
the engine must stop/join its worker and finish MLX work before model unload.
The locks here exclude Python calls, not asynchronous GPU evaluation.
"""

import copy
import threading
from contextlib import nullcontext
from types import MethodType

_MARKER = "_flash1017_runtime_controller"
_MISSING = object()


def _install_cone(target, **kwargs):
    from .cone import install

    return install(target, **kwargs)


def _install_rows(model, **kwargs):
    from .rows import install

    return install(model, **kwargs)


def native_hooks():
    from .. import language, storage
    from ..quantization import QuantizedProjection

    return {
        "language_call": language.LanguageModel.__call__,
        "language_prefill": language.LanguageModel._omlx_prefill,
        "language_append": language.LanguageModel.dspark_append_context,
        "block": language.Block.__call__,
        "projection_call": QuantizedProjection.__call__,
        "projection": QuantizedProjection.project_quantized,
        "gate": language.Gate.__call__,
        "moe": language.MoE.__call__,
        "hc": language.hc_mixes,
        "disk_read": storage.DiskEngramEmbedding._read_rows,
        "disk_close": storage.DiskEngramEmbedding.close,
    }


def parameter_identity(model):
    from mlx.utils import tree_flatten

    return tuple(
        (key, id(value), tuple(value.shape), str(value.dtype))
        for key, value in tree_flatten(model.parameters())
    )


def drained(target):
    prefetch = getattr(target, "_engram_prefetch", None)
    if prefetch is not None and (prefetch._closed or prefetch._pending is not None):
        raise RuntimeError("Require an open drained prefetcher between model requests")


class Controller:
    """Own a fresh outer model's replay and cache, including normal close().

    close() detaches the runtime while keeping the model open. model.close()
    also shuts down native resources, even if an earlier shutdown step raises.
    Both release all controller references to the model. Repeated controller
    close is harmless. Foreign wrapping or an active call causes a refusal
    before mutation, so another owner's work is never silently discarded.
    """

    def __init__(self, model, *, enabled=True, record_events=False):
        if type(enabled) is not bool or type(record_events) is not bool:
            raise TypeError("enabled and record_events must be bool")
        if hasattr(model, _MARKER):
            raise RuntimeError("This model already has a Flash1017 runtime owner")
        self.model = model
        self.target = model.language_model
        self.original_hooks = native_hooks()
        self.original_parameters = parameter_identity(model)
        self.cone = self.rows = None
        self.closed = False
        self._lock = threading.RLock()
        self._patches = []
        self._closed_snapshot = None
        drained(self.target)
        try:
            self.cone = _install_cone(
                self.target, enabled=False, record_events=record_events
            )
            # The combined controller owns outer close. Keep the row owner's
            # robust prefetch/table shutdown routine without stacking its hook.
            self.rows = _install_rows(
                model,
                enabled=False,
                budget_per_table=256 * 2**20,
                own_model_close=False,
            )
            self._patch(model, _MARKER, self)
            self._patch(
                model, "close", MethodType(lambda obj: self._model_close(), model)
            )
            self.verify()
            self.set_enabled(enabled)
        except BaseException as install_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                install_error.add_note(
                    f"Flash1017 cleanup also failed: {cleanup_error!r}"
                )
            raise

    def _patch(self, obj, name, value):
        previous = vars(obj).get(name, _MISSING)
        object.__setattr__(obj, name, value)
        self._patches.append((obj, name, value, previous))

    def _validate_restore(self):
        for obj, name, value, _previous in self._patches:
            if vars(obj).get(name, _MISSING) is not value:
                raise RuntimeError(f"Owned runtime attribute was replaced: {name}")
        if self.rows is not None:
            # Reject callback/forward teardown before removing either owner's
            # hooks. The row controller must still be able to resume afterward.
            self.rows._reject_reader_control()
            self.rows._reject_forward_control()
            self.rows._validate_restore()
        if self.cone is not None:
            self.cone._validate_restore()

    def _restore(self):
        for obj, name, _value, previous in reversed(self._patches):
            if previous is _MISSING:
                object.__delattr__(obj, name)
            else:
                object.__setattr__(obj, name, previous)
        self._patches.clear()

    def verify(self):
        if native_hooks() != self.original_hooks:
            raise RuntimeError("Flash1017 changed a process-wide inference method")
        if parameter_identity(self.model) != self.original_parameters:
            raise RuntimeError("Flash1017 changed parameter names or tensor identity")

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        with self._lock:
            if self.closed:
                raise RuntimeError("Flash1017 controller is closed")
            with self.cone._admit():
                drained(self.target)
                self.rows.set_enabled(enabled)
                # Already own admission: calling cone.set_enabled would be
                # correctly rejected as a reentrant control operation.
                self.cone.enabled = enabled
        return self

    def enable(self, enabled=True):
        return self.set_enabled(enabled)

    def snapshot(self, *, clear_cone=False):
        if type(clear_cone) is not bool:
            raise TypeError("clear_cone must be bool")
        with self._lock:
            if self.closed:
                return copy.deepcopy(self._closed_snapshot)
            with self.cone._admit():
                result = {
                    "closed": False,
                    "enabled": self.cone.enabled,
                    "cone": self.cone._snapshot(),
                    "rows": self.rows.snapshot(),
                }
                if clear_cone:
                    self.cone._clear_admitted()
                return result

    def _finish(self):
        self._closed_snapshot = {
            "closed": True,
            "enabled": False,
            "cone": self.cone.snapshot() if self.cone is not None else None,
            "rows": self.rows.snapshot() if self.rows is not None else None,
            "model_references_released": True,
        }
        self.model = self.target = None
        self.original_parameters = self.original_hooks = None
        self.closed = True

    def _shutdown(self, *, unload):
        with self._lock:
            if self.closed:
                return
            admission = self.cone._admit() if self.cone is not None else nullcontext()
            with admission:
                # Validate every owner before restoring anything. In particular,
                # do not unload native resources underneath a foreign wrapper.
                self._validate_restore()
                error = None
                try:
                    self.verify()
                except BaseException as exc:
                    # Record invariant violations, but still release our own
                    # references during a failed load/unload.
                    error = exc
                self._restore()
                if unload:
                    # Detach classes before native close can clear parameters.
                    # The row owner runs all native shutdown actions despite
                    # failures and releases its cache in every normal case.
                    if self.cone is not None:
                        self.cone._close_admitted()
                    try:
                        if self.rows is not None:
                            self.rows._model_close()
                    except BaseException as exc:
                        error = error or exc
                else:
                    try:
                        if self.rows is not None:
                            self.rows.close()
                    except BaseException as exc:
                        error = error or exc
                    if self.cone is not None:
                        self.cone._close_admitted()
                self._finish()
                if error is not None:
                    raise error

    def close(self):
        """Detach at a drained boundary; preserve native model resources."""
        self._shutdown(unload=False)

    detach = close

    def _model_close(self):
        self._shutdown(unload=True)


def install(outer_model, enabled=True, record_events=False):
    """Install the complete opt-in runtime on a fresh configured outer model."""
    return Controller(outer_model, enabled=enabled, record_events=record_events)
