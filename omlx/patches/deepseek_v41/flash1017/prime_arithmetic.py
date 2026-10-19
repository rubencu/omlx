# SPDX-License-Identifier: Apache-2.0
"""Qualified Flash1017 arithmetic, installed only by the per-model controller.

Arithmetic bodies are retained from the frozen research implementation.
Prototype process-wide context managers are deliberately excluded. Shape
restoration limits known dispatch differences; it does not prove universal
floating-point or output equivalence.
"""

import threading
from collections import Counter

import mlx.core as mx

from .. import language
from ..quantization import QuantizedProjection


class PrimeStabilization:
    def __init__(self, model, *, record_events=True):
        if not isinstance(model, language.LanguageModel):
            raise TypeError("Pass the pinned V4.1 language_model")
        config = model._config
        if (
            config.n_layers != 40
            or config.window_size != 128
            or tuple(config.dspark_target_layer_ids) != (37, 38, 39)
            or len(model.mtp) != 3
        ):
            raise ValueError("DSpark padding requires the pinned 1017 layout")
        self.model, self.record_events = model, bool(record_events)
        self.stages = {}
        for i, stage in enumerate(model.mtp):
            p = stage.attn.wkv
            if (
                type(p) is not QuantizedProjection
                or not p.quantize_input
                or p.mode != "mxfp8"
                or p.bits != 8
                or p.group_size != 32
                or p.input_dims != 5120
                or p.weight.shape[-2] != 512
            ):
                raise ValueError("Unexpected pinned DSpark wkv projection")
            if id(p) in self.stages:
                raise ValueError("DSpark stages must own distinct projections")
            self.stages[id(p)] = i
        self.required = config.window_size + (config.n_layers // 2) * (
            config.window_size - 1
        )
        self.frame, self.phase, self.active = None, None, False
        self.events, self.counts = [], Counter()

    def _owner(self):
        if threading.get_ident() != self.thread:
            raise RuntimeError("DSpark stabilization is installation-thread scoped")

    def _reason(self, ids, cache, kwargs):
        if ids.ndim != 2 or ids.shape[0] != 1:
            return "not-singleton-text"
        if ids.shape[1] < self.required:
            return "short-chunk"
        if (
            kwargs.get("inputs_embeds") is not None
            or kwargs.get("token_types") is not None
        ):
            return "multimodal-or-token-types"
        if kwargs.get("return_hidden") or kwargs.get("return_dspark_hidden"):
            return "explicit-full-hidden"
        if (
            kwargs.get("target_verify")
            or kwargs.get("n_confirmed")
            or kwargs.get("mtp_verify_states") is not None
        ):
            return "verification"
        if any(
            k
            not in {
                "skip_lm_head",
                "return_hidden",
                "return_dspark_hidden",
                "target_verify",
                "n_confirmed",
                "mtp_verify_states",
                "inputs_embeds",
                "token_types",
            }
            for k in kwargs
        ):
            return "unknown-prefill-option"
        if cache is None or len(cache) != 40:
            return "missing-or-unsupported-cache"
        for item in cache:
            padding = getattr(item, "left_padding", None)
            if padding is not None and any(
                int(x) != 0 for x in padding.reshape(-1).tolist()
            ):
                return "padded-cache"
            if getattr(item, "lengths", None) is not None:
                return "length-limited-cache"
        return None

    def _prefill(self, instance, input_ids, cache=None, **kwargs):
        if instance is not self.model:
            return self.original_prefill(instance, input_ids, cache=cache, **kwargs)
        self._owner()
        if self.frame is not None or self.in_prefill:
            raise RuntimeError("Reentrant DSpark prefill stabilization")
        reason = self._reason(input_ids, cache, kwargs)
        event = {
            "kind": "prefill",
            "native_rows": int(input_ids.shape[1]),
            "reason": reason,
            "complete": False,
            "append_count": 0,
            "stages": [],
        }
        self.counts["prefill_calls"] += 1
        if reason is None:
            start = int(cache[0].size())
            event.update(start=start, end=start + int(input_ids.shape[1]))
            self.frame = event
            self.counts["eligible_prefills"] += 1
        else:
            self.counts["fallback:" + reason] += 1
        if self.record_events:
            self.events.append(event)
        self.in_prefill = True
        try:
            result = self.original_prefill(instance, input_ids, cache=cache, **kwargs)
            # Native capture catches append exceptions and silently discards
            # priming. Experimental failures must remain visible to callers.
            if "append_error" in event or (
                event["append_count"] and not event.get("append_complete")
            ):
                raise RuntimeError(
                    "DSpark priming failed during stabilized prefill: "
                    + event.get("append_error", "incomplete append")
                )
            event["complete"] = True
            return result
        except BaseException as exc:
            event["error"] = repr(exc)
            self.counts["prefill_failures"] += 1
            raise
        finally:
            self.phase, self.frame, self.in_prefill = None, None, False

    def _append(self, instance, hidden, cache, *, start_offset=None):
        if instance is not self.model:
            return self.original_append(
                instance, hidden, cache, start_offset=start_offset
            )
        self._owner()
        if self.frame is None or not self.model._config.ced_prefill:
            return self.original_append(
                instance, hidden, cache, start_offset=start_offset
            )
        frame = self.frame
        # Full capture is the native baseline/fallback and needs no correction.
        if hidden.ndim == 3 and hidden.shape[:2] == (1, frame["native_rows"]):
            self.counts["native_full_capture"] += 1
            return self.original_append(
                instance, hidden, cache, start_offset=start_offset
            )
        if (
            hidden.ndim != 3
            or hidden.shape != (1, 128, 15360)
            or len(cache) != 3
            or start_offset is None
            or int(start_offset) + 128 != frame["end"]
        ):
            frame["append_error"] = "Unexpected captured suffix or absolute endpoint"
            self.counts["append_failures"] += 1
            raise RuntimeError("Unexpected DSpark captured suffix or absolute endpoint")
        if self.phase is not None or frame["append_count"]:
            frame["append_error"] = "Duplicate/reentrant priming append"
            self.counts["append_failures"] += 1
            raise RuntimeError("Duplicate/reentrant DSpark priming append")
        frame["append_count"] += 1
        frame["append_start"] = int(start_offset)
        frame["append_complete"] = False
        self.phase = {"next_stage": 0}
        try:
            result = self.original_append(
                instance, hidden, cache, start_offset=start_offset
            )
            if self.phase["next_stage"] != 3:
                raise RuntimeError(
                    "Priming did not visit all three DSpark wkv projections"
                )
            frame["append_complete"] = True
            self.counts["padded_appends"] += 1
            return result
        except BaseException as exc:
            frame["append_error"] = repr(exc)
            self.counts["append_failures"] += 1
            raise
        finally:
            self.phase = None

    def _projection(
        self, instance, x, indices=None, sorted_indices=False, block_plan=None
    ):
        stage = self.stages.get(id(instance))
        if stage is None or self.phase is None:
            return self.original_projection(
                instance, x, indices, sorted_indices, block_plan
            )
        self._owner()
        if (
            self.frame is None
            or stage != self.phase["next_stage"]
            or x.ndim != 3
            or x.shape != (1, 128, 5120)
            or indices is not None
            or sorted_indices
            or block_plan is not None
        ):
            raise RuntimeError("Unexpected stage-wkv projection during priming")
        rows = self.frame["native_rows"]
        missing = rows - 128
        padded = mx.concatenate([mx.zeros((1, missing, 5120), x.dtype), x], axis=1)
        result = self.original_projection(instance, padded)
        if result.shape != (1, rows, 512):
            raise RuntimeError("Unexpected padded stage-wkv output shape")
        self.phase["next_stage"] += 1
        self.counts["wkv_pads"] += 1
        self.counts["stage:" + str(stage)] += 1
        self.frame["stages"].append(
            {"stage": stage, "input_rows": 128, "native_rows": rows, "output_rows": 128}
        )
        return result[:, missing:]
