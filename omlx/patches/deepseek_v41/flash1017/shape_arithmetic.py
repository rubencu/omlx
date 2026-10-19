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

from .. import hyper_connection, language
from ..quantization import QuantizedProjection


def quantized_split_partitions(rows, output_width, input_width, group_size):
    """MLXv0.32.4 nonbatched transposed QMM dispatch, for matrix-sized M."""
    tiles = ((int(rows) + 31) // 32) * ((int(output_width) + 31) // 32)
    alignment = max(int(group_size), 32)
    partitions = min(max(1, 512 // tiles), int(input_width) // alignment)
    while partitions > 1 and int(input_width) % (partitions * alignment):
        partitions -= 1
    return partitions


def padded_projection(projection, x, native_rows, *, original=None):
    """Recreate a native singleton query shape for one quantized projection."""
    if x.ndim != 3 or x.shape[0] != 1 or not 0 < x.shape[1] <= native_rows:
        raise ValueError("Expected a nonempty singleton suffix of native_rows")
    original = original or QuantizedProjection.project_quantized
    missing = int(native_rows) - int(x.shape[1])
    if not missing:
        return original(projection, x)
    padded = mx.concatenate([mx.zeros((1, missing, x.shape[-1]), x.dtype), x], axis=1)
    # Prefix padding places retained rows at their original native row offsets.
    return original(projection, padded)[:, missing:]


class ShapeStabilization:
    def __init__(
        self,
        model,
        *,
        layers=(20,),
        gate=False,
        hc=False,
        local_wkv=False,
        moe_min256=False,
        record_events=True,
    ):
        if not isinstance(model, language.LanguageModel):
            raise TypeError("Pass a V4.1 language_model")
        self.model = model
        self.layers = tuple(int(i) for i in layers)
        self.projections = {}
        for i in self.layers:
            projection = model.layers[i].attn.indexer.weights_proj
            if (
                not isinstance(projection, QuantizedProjection)
                or projection.quantize_input
            ):
                raise ValueError(
                    "This probe requires an unquantized-input QuantizedProjection"
                )
            self.projections[id(projection)] = i
        self.local_projections = {}
        if local_wkv:
            for i in range(model._config.n_layers // 2, model._config.n_layers):
                projection = model.layers[i].attn.wkv
                if not isinstance(projection, QuantizedProjection):
                    raise ValueError("Local-wkv probe requires QuantizedProjection")
                self.local_projections[id(projection)] = i
        self.final_moe = model.layers[-1].ffn if moe_min256 else None
        self.gates = (
            {
                id(block.ffn.gate): i
                for i, block in enumerate(model.layers)
                if i >= model._config.n_layers // 2
            }
            if gate
            else {}
        )
        self.hc = bool(hc)
        self.record_events = bool(record_events)
        self.events, self.counts = [], Counter()
        self.frame = None
        self.active = False

    def _owner(self):
        if threading.get_ident() != self.thread:
            raise RuntimeError("Shape stabilization is single-thread scoped")

    def _record(self, kind, rows, layer=None):
        self.counts[kind] += 1
        if self.record_events:
            self.events.append(
                {
                    "kind": kind,
                    "layer": layer,
                    "rows": int(rows),
                    "native_rows": self.frame,
                }
            )

    def _enabled(self):
        return self.frame is not None and self.model._config.ced_prefill

    def _call(self, instance, input_ids, *args, **kwargs):
        if instance is not self.model:
            return self.original_call(instance, input_ids, *args, **kwargs)
        self._owner()
        if self.frame is not None:
            raise RuntimeError("Reentrant model call during shape stabilization")
        eligible = (
            kwargs.get("_ced_prefill", False)
            and input_ids.ndim == 2
            and input_ids.shape[0] == 1
        )
        self.frame = int(input_ids.shape[1]) if eligible else None
        try:
            return self.original_call(instance, input_ids, *args, **kwargs)
        finally:
            self.frame = None

    def _projection(
        self, instance, x, indices=None, sorted_indices=False, block_plan=None
    ):
        layer = self.projections.get(id(instance))
        kind = "index-pad"
        if layer is None and id(instance) in self.local_projections:
            layer, kind = self.local_projections[id(instance)], "local-wkv-pad"
        if layer is not None and self._enabled():
            self._owner()
            if indices is not None or sorted_indices or block_plan is not None:
                raise RuntimeError("Unexpected indexed weights projection")
            if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] > self.frame:
                raise RuntimeError("Unexpected index projection input shape")
            if x.shape[1] < self.frame:
                changed_partition = kind == "index-pad" or (
                    quantized_split_partitions(
                        x.shape[1],
                        instance.weight.shape[-2],
                        instance.input_dims,
                        instance.group_size,
                    )
                    != quantized_split_partitions(
                        self.frame,
                        instance.weight.shape[-2],
                        instance.input_dims,
                        instance.group_size,
                    )
                )
                if changed_partition:
                    self._record(kind, x.shape[1], layer)
                    return padded_projection(
                        instance, x, self.frame, original=self.original_projection
                    )
        return self.original_projection(
            instance, x, indices, sorted_indices, block_plan
        )

    def _gate(self, instance, x, image_mask):
        layer = self.gates.get(id(instance))
        if layer is not None and self._enabled() and x.shape[1] < self.frame:
            self._owner()
            if x.ndim != 3 or x.shape[0] != 1 or image_mask is not None:
                raise RuntimeError("Gate shape probe supports singleton text only")
            missing = self.frame - int(x.shape[1])
            padded = mx.concatenate(
                [mx.zeros((1, missing, x.shape[-1]), x.dtype), x], 1
            )
            self._record("gate-pad", x.shape[1], layer)
            ids, weights = self.original_gate(instance, padded, None)
            return ids[:, missing:], weights[:, missing:]
        return self.original_gate(instance, x, image_mask)

    def _moe(self, instance, x, image_mask):
        if instance is self.final_moe and self._enabled() and x.shape[1] == 255:
            self._owner()
            config = instance.gate._config
            if (
                x.ndim != 3
                or x.shape[0] != 1
                or image_mask is not None
                or config.n_routed_experts != 384
                or config.n_activated_experts != 6
            ):
                raise RuntimeError(
                    "MoE256 probe supports only the pinned singleton1017 final layer"
                )
            # One dummy row changes1530 routes to1536, crossing stock MLX's
            # routes/expert>=4 grouped-NAX gate. Rows have no mathematical
            # interaction in MoE. No attention/cache/query bounds are changed.
            padded = mx.concatenate([mx.zeros((1, 1, x.shape[-1]), x.dtype), x], 1)
            self._record("final-moe-256", x.shape[1], self.model._config.n_layers - 1)
            return self.original_moe(instance, padded, None)[:, 1:]
        return self.original_moe(instance, x, image_mask)

    def _hc(self, x, fn, scale, base, config):
        if self.hc and config is self.model._config and self._enabled():
            self._owner()
            # Match the large native chunk's warp32 row-local accumulation,
            # including the final255-row call which normally uses FP32 GEMM.
            if (
                self.frame >= 1024
                and x.ndim == 4
                and x.shape[1] < 1024
                and x.shape[-2] == config.hc_mult == 4
                and x.dtype == mx.bfloat16
                and fn.dtype == mx.float32
                and fn.shape == (24, 4 * x.shape[-1])
                and mx.default_device() == mx.gpu
            ):
                width = int(x.shape[-1]) * 4
                rows = x.size // width
                mixes = hyper_connection._projection_kernel(warp=True)(
                    inputs=[x, fn, mx.array([config.norm_eps], mx.float32)],
                    template=[("D", width)],
                    grid=(rows * 32, 2, 1),
                    threadgroup=(32, 1, 1),
                    output_shapes=[(*x.shape[:-2], 24)],
                    output_dtypes=[mx.float32],
                )[0]
                self._record("hc-warp32", x.shape[1])
                return language._hc_mix_weights(
                    mixes,
                    scale,
                    base,
                    config.hc_mult,
                    config.hc_eps,
                    config.hc_sinkhorn_iters,
                )
        return self.original_hc(x, fn, scale, base, config)
