# SPDX-License-Identifier: Apache-2.0
"""Frozen, causal DSpark prefix admission for the Flash1017 profile.

The rule follows arXiv:2607.05147 section 3.2.2: confidence for position k
uses the prefix through k-1. Stop at the first non-improving prefix; a global
argmax could condition admission on a later sampled candidate. This is an
integration of the published technique, not a new decoding algorithm.

These costs and position temperatures were fitted before qualification on
the exact oQ2.5e-mtp checkpoint, M5 Ultra 256 GiB, width 5, head rows 2..6,
and reference attention rounding. Calibration: cost6-01.jsonl, SHA256
d08e5e6915a3186817f09405bb867d14553aefb710b60df10cf813430e0cd9c8;
the saved confidence-policy6-01.json records sample counts and fit diagnostics.
No live request timing feeds a selection decision.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

DRAFT_WIDTH = 5
CYCLE_MS = (
    40.86379148066044,
    51.855687983334064,
    57.73166654398665,
    66.96656247368082,
    71.52610452612862,
    77.69158302107826,
)
TEMPERATURES = (
    1.0,
    0.9394130628134758,
    1.0644944589178593,
    0.8824969025845955,
    1.3668379411737963,
)


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return e / (1.0 + e)


@dataclass(frozen=True)
class Decision:
    depth: int
    expected_tokens: float
    predicted_tokens_per_ms: float


@dataclass(frozen=True)
class PrefixPolicy:
    # Every cost includes the same fixed-width draft pass and one target
    # token. Index zero admits no draft, but still pays for that draft pass.
    cycle_ms: tuple[float, ...]
    temperatures: tuple[float, ...] = ()

    def __post_init__(self):
        if len(self.cycle_ms) < 2 or any(
            not math.isfinite(x) or x <= 0 for x in self.cycle_ms
        ):
            raise ValueError("Need finite, positive cycle costs for depths 0..K")
        if self.temperatures and (
            len(self.temperatures) != len(self.cycle_ms) - 1
            or any(not math.isfinite(t) or t <= 0 for t in self.temperatures)
        ):
            raise ValueError("Need one finite, positive temperature per position")

    def choose(self, confidence_logits: Iterable[float]) -> Decision:
        survival, expected = 1.0, 1.0
        chosen = Decision(0, expected, expected / self.cycle_ms[0])
        for k, logit in enumerate(confidence_logits, 1):
            if k >= len(self.cycle_ms):
                break
            if not math.isfinite(logit):
                break
            temperature = self.temperatures[k - 1] if self.temperatures else 1.0
            survival *= sigmoid(logit / temperature)
            expected += survival
            rate = expected / self.cycle_ms[k]
            if rate <= chosen.predicted_tokens_per_ms:
                break
            chosen = Decision(k, expected, rate)
        return chosen


FROZEN_POLICY = PrefixPolicy(CYCLE_MS, TEMPERATURES)


@dataclass
class ConfidenceController:
    """Per-model admission with bounded counters, never retained event tensors.

    Only one active request is supported. Context-copy cycles keep their existing
    deterministic proposal/acceptance path and are counted separately.
    """

    draft_width: int = field(default=DRAFT_WIDTH, init=False)
    policy: PrefixPolicy = field(default=FROZEN_POLICY, init=False)
    cycles: int = field(default=0, init=False)
    copied_cycles: int = field(default=0, init=False)
    verified: int = field(default=0, init=False)
    accepted: int = field(default=0, init=False)
    selected_depths: list[int] = field(
        default_factory=lambda: [0] * (DRAFT_WIDTH + 1), init=False
    )

    def select(self, logits, current_depth):
        depth = min(current_depth, self.policy.choose(iter(logits)).depth)
        self.selected_depths[depth] += 1
        return depth

    def record(self, *, verified, accepted, copied, **_unused):
        self.cycles += 1
        self.copied_cycles += int(copied)
        self.verified += verified
        self.accepted += accepted


def configure(language_model):
    """Install only on the explicitly opted-in, fully loaded model."""
    if not language_model._config.flash1017_enabled:
        raise ValueError("Flash1017 confidence policy requires explicit model opt-in")
    if language_model._config.dspark_block_size < DRAFT_WIDTH:
        raise ValueError("Flash1017 requires at least five DSpark draft positions")
    language_model.configure_mtp(True, DRAFT_WIDTH)
    language_model._omlx_mtp_depth_fixed = True
    language_model._flash1017_experiment = ConfidenceController()
    return language_model._flash1017_experiment
