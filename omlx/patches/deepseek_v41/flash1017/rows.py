# SPDX-License-Identifier: Apache-2.0
"""Per-model, CPU-only Engram row caching for the pinned Flash1017 runtime.

Install after load/configure, before exposing a model to its inference worker.
Only owned *instance* methods change. No class, module, kernel or global budget
is patched. Control transitions drain the native prefetch executor before
changing policy; the native forward context owns the lifecycle lock throughout
its final drain. Background reads only acquire the native embedding RLock.

The controller is storage lifecycle protection, not a model invocation lock:
stop/join the inference worker before unload/detach. The opt-in cone serving
controller supplies fail-fast whole-model invocation exclusion separately.
"""

import hashlib
import threading
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import MethodType

import numpy as np

from .hot_rows import HotRows

MAX_TABLE_BYTES = 256 * 2**20
MAX_TOTAL_BYTES = 2 * MAX_TABLE_BYTES
UPSTREAM_SHA256 = "341791362213d565b1b5d5aac1b8c38057375477f62a7f8177529e455e8015fa"
_MARKER = "_flash1017_serving_row_cache"
_MISSING = object()
_HERE = Path(__file__).resolve().parent


def dependency_fingerprints():
    """Packaged source bytes for a qualification manifest."""
    names = ("rows.py", "hot_rows.py", "HOT_ROWS_LICENSE", "provenance.json")
    return {
        name: hashlib.sha256((_HERE / name).read_bytes()).hexdigest() for name in names
    }


class _Patch:
    def __init__(self, obj, name, value):
        self.obj, self.name, self.value = obj, name, value
        self.previous = vars(obj).get(name, _MISSING)
        # MLX Module.__setattr__ stores many values in its parameter mapping.
        # Bypass it: policy, CPU bytes and methods never enter that tree.
        object.__setattr__(obj, name, value)

    def restore(self):
        if vars(self.obj).get(self.name, _MISSING) is not self.value:
            raise RuntimeError(f"Owned instance attribute was replaced: {self.name}")
        if self.previous is _MISSING:
            object.__delattr__(self.obj, self.name)
        else:
            object.__setattr__(self.obj, self.name, self.previous)


class _Table:
    def __init__(self, embed, budget, index):
        self.embed, self.index = embed, index
        self.cache = HotRows(budget)
        self.counts = Counter()
        self.disabled = False
        self.active = True
        self.original_read = embed._read_rows
        self.original_close = embed.close
        self.original_resident = embed.make_resident

    def summary(self):
        return {
            "cache_id": self.index,
            "table": str(self.embed._weight_key),
            "budget": self.cache.budget,
            "nbytes": self.cache.nbytes,
            "closed": bool(self.embed._closed),
            "resident": self.embed._resident is not None,
            "cache_disabled_after_allocation_failure": self.disabled,
            "counts": dict(self.counts),
            "unique_hits": self.cache.hits,
            "unique_queries": self.cache.queries,
            "admitted": self.cache.admitted,
        }


class RowCacheController:
    """Two immutable disk tables, at most 512 MiB retained payload/tags/sketch.

    set_enabled(False) bypasses without clearing history. close()/detach()
    releases arrays and restores methods without unloading the model. Native
    model.close() additionally shuts down prefetch and closes both tables even
    if an earlier native close step raises. All operations support sequential
    migration between loader/inference/shutdown threads.
    """

    def __init__(
        self,
        model,
        *,
        embedding_type=None,
        enabled=True,
        budget_per_table=MAX_TABLE_BYTES,
        own_model_close=True,
    ):
        if type(own_model_close) is not bool:
            raise TypeError("own_model_close must be a bool")
        if type(enabled) is not bool:
            raise TypeError("enabled must be a bool")
        if (
            type(budget_per_table) is not int
            or not 0 <= budget_per_table <= MAX_TABLE_BYTES
        ):
            raise ValueError(
                "budget_per_table must be an integer between 0 and 256 MiB"
            )
        if hasattr(model, _MARKER):
            raise RuntimeError("This model already owns a row cache")
        language = getattr(model, "language_model", None)
        if language is None:
            raise TypeError("Install the row cache on the outer model")
        prefetch = getattr(language, "_engram_prefetch", None)
        if prefetch is None:
            raise TypeError("The model has no native Engram prefetch manager")
        tables = []
        for layer in language.layers:
            engram = getattr(layer, "engram", None)
            if engram is None:
                continue
            embed = engram.embed
            is_disk = (
                isinstance(embed, embedding_type)
                if embedding_type is not None
                else any(
                    cls.__name__ == "DiskEngramEmbedding"
                    and cls.__module__.endswith("deepseek_v41.storage")
                    for cls in type(embed).__mro__
                )
            )
            if not is_disk:
                raise TypeError("Every Engram table must be a native disk embedding")
            tables.append(embed)
        if len(tables) != 2 or tables[0] is tables[1]:
            raise ValueError("Exactly two distinct native disk tables are required")
        for obj in [prefetch, *tables]:
            if hasattr(obj, _MARKER):
                raise RuntimeError(
                    "An owned prefetch manager or table is already adapted"
                )
        if prefetch._closed or prefetch._pending is not None:
            raise RuntimeError("Install only on a fresh, drained, open model")
        for embed in tables:
            if (
                embed._closed
                or embed._resident is not None
                or embed._prefetched is not None
            ):
                raise RuntimeError("Install only on fresh, drained, open disk tables")
            for name in ("_read_rows", "close", "make_resident", "selected_bytes"):
                if not callable(getattr(embed, name, None)):
                    raise TypeError(f"Unsupported disk table: missing {name}")
        for name in ("forward", "submit", "drain", "close"):
            if not callable(getattr(prefetch, name, None)):
                raise TypeError(f"Unsupported prefetch manager: missing {name}")
        if not callable(getattr(model, "close", None)):
            raise TypeError("The outer model has no close method")
        # Installation is deliberately a fresh-load-only operation. There can
        # be no reliable retrospective test for an already running drain,
        # because native drain clears _pending before waiting for its Future.
        self._lock = threading.RLock()
        self._read_local = threading.local()
        self._forward_owner = None
        self._patches = []
        self._states = [_Table(e, budget_per_table, i) for i, e in enumerate(tables)]
        self._retired = []
        self._model, self._prefetch = model, prefetch
        self._native_prefetch = {
            name: getattr(prefetch, name)
            for name in ("forward", "submit", "drain", "close")
        }
        self._native_model_close = model.close
        self.budget_per_table = budget_per_table
        self.enabled = enabled
        self.status = "installing"
        try:
            for obj in [model, prefetch, *tables]:
                self._patch(obj, _MARKER, self)
            for state in self._states:
                self._method(
                    state.embed,
                    "_read_rows",
                    lambda obj, host, state=state: self._read(state, host),
                )
                self._method(
                    state.embed,
                    "close",
                    lambda obj, state=state: self._table_close(state),
                )
                self._method(
                    state.embed,
                    "make_resident",
                    lambda obj, state=state: self._make_resident(state),
                )
            self._method(prefetch, "forward", lambda obj: self._forward())
            self._method(
                prefetch,
                "submit",
                lambda obj, *a, **kw: self._prefetch_call("submit", *a, **kw),
            )
            self._method(prefetch, "drain", lambda obj: self._prefetch_call("drain"))
            self._method(prefetch, "close", lambda obj: self._prefetch_close())
            if own_model_close:
                self._method(model, "close", lambda obj: self._model_close())
            self.status = "active"
        except BaseException:
            self._release("failed_install")
            raise

    def _patch(self, obj, name, value):
        self._patches.append(_Patch(obj, name, value))

    def _method(self, obj, name, function):
        self._patch(obj, name, MethodType(function, obj))

    def _reject_reader_control(self):
        if getattr(self._read_local, "depth", 0):
            raise RuntimeError(
                "Lifecycle controls cannot run inside a row-read callback"
            )

    def _reject_forward_control(self):
        if self._forward_owner == threading.get_ident():
            raise RuntimeError("Lifecycle controls require a drained forward boundary")

    def _require_active(self):
        if self.status != "active":
            raise RuntimeError(f"Row-cache controller is {self.status}")

    def _table_locks(self):
        stack = ExitStack()
        try:
            for state in self._states:
                stack.enter_context(state.embed._lock)
        except BaseException:
            stack.close()
            raise
        return stack

    @contextmanager
    def _forward(self):
        self._reject_reader_control()
        with self._lock:
            self._require_active()
            if self._forward_owner is not None:
                raise RuntimeError("Nested native Engram forwards are unsupported")
            self._forward_owner = threading.get_ident()
            try:
                # Keep lifecycle lock until native finally/drain completes.
                # The executor's _read_rows deliberately never takes this lock.
                with self._native_prefetch["forward"]() as result:
                    yield result
            finally:
                self._forward_owner = None

    def _prefetch_call(self, name, *args, **kwargs):
        self._reject_reader_control()
        with self._lock:
            if self.status not in ("active", "closing"):
                raise RuntimeError(f"Row-cache controller is {self.status}")
            return self._native_prefetch[name](*args, **kwargs)

    def _prefetch_close(self):
        self._reject_reader_control()
        with self._lock:
            self._reject_forward_control()
            if self.status not in ("active", "closing"):
                raise RuntimeError(f"Row-cache controller is {self.status}")
            return self._native_prefetch["close"]()

    def set_enabled(self, enabled, *, clear=False):
        if type(enabled) is not bool or type(clear) is not bool:
            raise TypeError("enabled and clear must be bools")
        self._reject_reader_control()
        with self._lock:
            self._reject_forward_control()
            self._require_active()
            self._native_prefetch["drain"]()
            with self._table_locks():
                self.enabled = enabled
                if clear:
                    for state in self._states:
                        state.cache.clear()

    def snapshot(self):
        self._reject_reader_control()
        with self._lock, self._table_locks():
            live = [s.summary() for s in self._states]
            retained = sum(s["nbytes"] for s in live)
            if retained > 2 * self.budget_per_table or retained > MAX_TOTAL_BYTES:
                raise RuntimeError("Retained row cache exceeds its configured bound")
            return {
                "status": self.status,
                "enabled": self.enabled,
                "per_table_budget_bytes": self.budget_per_table,
                "max_retained_bytes": 2 * self.budget_per_table,
                "retained_bytes": retained,
                "live_tables": live,
                "retired_tables": list(self._retired),
                "scope": "Retained payload, full-ID tags and frequency sketch only; temporary CPU gather arrays are additional.",
            }

    def _native_read(self, state, host):
        state.counts["native_read_calls"] += 1
        state.counts["native_requested_rows"] += int(host.size)
        state.counts["native_selected_bytes"] += int(
            state.embed.selected_bytes(host.size)
        )
        try:
            return state.original_read(host)
        except BaseException:
            state.counts["read_failures"] += 1
            raise

    def _read(self, state, host):
        # All admitted direct/worker reads serialize with close on the native
        # embedding lock. Never acquire self._lock from the executor thread.
        self._read_local.depth = getattr(self._read_local, "depth", 0) + 1
        try:
            with state.embed._lock:
                if not state.active:
                    return state.original_read(host)  # stale saved callback
                state.counts["calls"] += 1
                if state.embed._closed:
                    return self._native_read(state, host)
                if (
                    not self.enabled
                    or not self.budget_per_table
                    or state.disabled
                    or state.embed._resident is not None
                    or not isinstance(host, np.ndarray)
                    or host.dtype.kind not in "iu"
                    or not host.size
                ):
                    if state.embed._resident is not None:
                        state.cache.clear()
                    state.counts["bypass_calls"] += 1
                    # Preserve native behavior for unsupported host types.
                    if not isinstance(host, np.ndarray):
                        return state.original_read(host)
                    return self._native_read(state, host)
                flat = host.reshape(-1)
                for reader, key in (
                    (state.embed._weights, state.embed._weight_key),
                    (state.embed._scales, state.embed._scale_key),
                    (state.embed._scales, state.embed._bias_key),
                ):
                    if key is not None:
                        size = int(reader.header[key]["shape"][0])
                        if flat.min() < 0 or flat.max() >= size:
                            raise IndexError("Engram row outside table")
                rows, inverse, counts = np.unique(
                    flat.astype(np.int64, copy=False),
                    return_inverse=True,
                    return_counts=True,
                )
                state.counts["cache_calls"] += 1
                state.counts["requested_rows"] += int(flat.size)
                state.counts["unique_rows"] += int(rows.size)
                state.counts["duplicate_rows_elided"] += int(flat.size - rows.size)
                hits, hit_data = state.cache.lookup(rows)
                missing = rows[~hits]
                state.counts["miss_rows"] += int(missing.size)
                if missing.size:
                    gathered = self._native_read(state, missing)
                    if hits.any():
                        if len(gathered) != len(hit_data):
                            raise RuntimeError("Engram cache component count changed")
                        merged = []
                        for (raw, dtype), (saved, saved_dtype) in zip(
                            gathered, hit_data
                        ):
                            if (
                                dtype != saved_dtype
                                or raw.dtype != saved.dtype
                                or raw.shape[1:] != saved.shape[1:]
                            ):
                                raise RuntimeError(
                                    "Engram cache component layout changed"
                                )
                            full = np.empty(
                                (len(rows), *raw.shape[1:]), dtype=raw.dtype
                            )
                            full[hits], full[~hits] = saved, raw
                            merged.append((full, dtype))
                        gathered = merged
                else:
                    gathered = list(hit_data)
                    state.counts["all_hit_calls"] += 1
                try:
                    state.cache.remember(rows, counts, gathered)
                except MemoryError:
                    state.cache.clear()
                    state.disabled = True
                    state.counts["cache_allocation_failures"] += 1
                return [(raw[inverse], dtype) for raw, dtype in gathered]
        finally:
            self._read_local.depth -= 1

    def _table_close(self, state):
        self._reject_reader_control()
        with self._lock:
            self._reject_forward_control()
            error = None
            try:
                self._native_prefetch["drain"]()
            except BaseException as exc:
                error = exc
            try:
                with state.embed._lock:
                    try:
                        result = state.original_close()
                    finally:
                        state.cache.clear()
            except BaseException as exc:
                error = error or exc
            if error is not None:
                raise error
            return result

    def _make_resident(self, state):
        self._reject_reader_control()
        with self._lock:
            self._reject_forward_control()
            self._require_active()
            self._native_prefetch["drain"]()
            with state.embed._lock:
                try:
                    return state.original_resident()
                finally:
                    state.cache.clear()

    def _validate_restore(self):
        for patch in self._patches:
            if vars(patch.obj).get(patch.name, _MISSING) is not patch.value:
                raise RuntimeError(
                    f"Owned instance attribute was replaced: {patch.name}"
                )

    def _release(self, status):
        # Caller owns lifecycle lock and has drained executor work. Also wait
        # direct reads before clearing/restoring; no native file resource closes
        # here. Preserve small summaries but drop references to the full model.
        error = None
        with self._table_locks():
            for state in self._states:
                state.active = False
                state.cache.clear()
                self._retired.append(state.summary())
            for patch in reversed(self._patches):
                try:
                    patch.restore()
                except BaseException as exc:
                    error = error or exc
            self._patches.clear()
            self._states.clear()
            self._model = self._prefetch = None
            self._native_prefetch.clear()
            self._native_model_close = None
            self.enabled = False
            self.status = status
        if error is not None:
            raise error

    def close(self):
        """Drain, release cache arrays and restore native methods; keep model open.

        Call after stopping/joining the model worker, then detach the cone
        controller, then call native model.close() for complete unload.
        """
        self._reject_reader_control()
        with self._lock:
            self._reject_forward_control()
            if self.status in ("detached", "closed", "failed_install"):
                return
            self._require_active()
            self._validate_restore()
            error = None
            try:
                self._native_prefetch["drain"]()
            except BaseException as exc:
                error = exc
            try:
                self._release("detached")
            except BaseException as exc:
                error = error or exc
            if error is not None:
                raise error

    detach = close

    def _model_close(self):
        self._reject_reader_control()
        with self._lock:
            self._reject_forward_control()
            self._require_active()
            self.status = "closing"
            error = None
            # Offload close can raise before native Model.close reaches Engram.
            # Explicitly shut down prefetch first, then execute original close,
            # and finally close every table regardless of any earlier error.
            for action in [
                self._native_prefetch["close"],
                self._native_model_close,
                *(state.original_close for state in self._states),
            ]:
                try:
                    action()
                except BaseException as exc:
                    error = error or exc
            try:
                self._release("closed")
            except BaseException as exc:
                error = error or exc
            if error is not None:
                raise error


def install(
    model,
    *,
    embedding_type=None,
    enabled=True,
    budget_per_table=MAX_TABLE_BYTES,
    own_model_close=True,
):
    """Fresh-load-only opt-in; no prefetch budget or inference arithmetic change."""
    return RowCacheController(
        model,
        embedding_type=embedding_type,
        enabled=enabled,
        budget_per_table=budget_per_table,
        own_model_close=own_model_close,
    )
