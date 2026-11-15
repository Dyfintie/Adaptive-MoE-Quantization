from __future__ import annotations

import importlib.metadata
import time
from collections.abc import Iterator
from typing import Any

from .contracts import RunEvent, RunRecord, RunRequest

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
TIERS = {"NF4": 2, "INT8": 1, "FP16": 0}


class LiveBackendUnavailable(RuntimeError):
    pass


def startup_status() -> dict[str, Any]:
    status: dict[str, Any] = {"cuda": False, "gpu_name": None, "live_ready": False, "issues": []}
    for package in ("torch", "transformers", "bitsandbytes", "gradio", "stable-baselines3"):
        try:
            status[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            status[package] = None
    try:
        import torch
        status["cuda"] = torch.cuda.is_available()
        if status["cuda"]:
            status["gpu_name"] = torch.cuda.get_device_name()
    except ImportError:
        pass
    missing = [p for p in ("torch", "transformers", "bitsandbytes", "stable-baselines3") if status[p] is None]
    if missing:
        status["issues"].append("Missing packages: " + ", ".join(missing))
    if not status["cuda"]:
        status["issues"].append("CUDA GPU unavailable")
    status["issues"].append("Live quantized OLMoE backend is gated until the real-expert compatibility check and memory validation are recorded.")
    return status


def run_live(request: RunRequest, stop_event: Any = None) -> Iterator[RunEvent]:
    """Fail closed: do not report fabricated allocations or unvalidated output."""
    status = startup_status()
    yield RunEvent("error", time.time(), {"message": "Live inference unavailable", "details": status["issues"]})
    raise LiveBackendUnavailable("Live backend gate has not passed. See demo/README.md for the required GPU validation.")


def make_failed_record(request: RunRequest, message: str) -> RunRecord:
    return RunRecord(request=request.__dict__, status="error", measurements={"error": message},
                     software={"gate": "live backend not enabled"})
