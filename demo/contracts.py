from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Allocation = Literal["uniform_nf4", "uniform_int8", "uniform_fp16", "importance", "ppo"]


@dataclass
class RunRequest:
    prompt: str
    allocation: Allocation
    budget_gib: float
    battery_percent: int = 100
    max_new_tokens: int = 64
    mode: Literal["live", "replay"] = "live"


@dataclass
class RunEvent:
    kind: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    request: dict[str, Any]
    status: str
    output: str = ""
    model_id: str = "allenai/OLMoE-1B-7B-0924"
    model_revision: str | None = None
    checkpoint_sha256: str | None = None
    software: dict[str, str] = field(default_factory=dict)
    proposed_assignment: list[int] = field(default_factory=list)
    executed_assignment: list[int] = field(default_factory=list)
    adjustments: list[dict[str, Any]] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    correctness: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
