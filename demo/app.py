from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from .backend import LiveBackendUnavailable, make_failed_record, run_live, startup_status
from .contracts import RunRequest
from .replay import export_records, load_record

ROOT = Path(__file__).resolve().parent
records: list[dict] = []
active_stop = threading.Event()


def task(prompt: str, allocation: str, budget: float, battery: int, max_tokens: int):
    active_stop.clear()
    method = {"Uniform NF4": "uniform_nf4", "Uniform INT8": "uniform_int8", "Uniform FP16": "uniform_fp16",
              "Importance heuristic": "importance", "PPO + budget": "ppo"}[allocation]
    req = RunRequest(prompt=prompt, allocation=method, budget_gib=budget, battery_percent=battery,
                     max_new_tokens=max_tokens)
    status = startup_status()
    yield "Preparing live backend…", json.dumps(status, indent=2), "No GPU measurements yet", ""
    try:
        for ev in run_live(req, active_stop):
            yield "", json.dumps(ev.data, indent=2), "No GPU measurements yet", ""
    except LiveBackendUnavailable as exc:
        record = make_failed_record(req, str(exc)).to_dict()
        record["events"] = [{"kind": "error", "timestamp": time.time(), "data": {"message": str(exc)}}]
        records.append(record)
        yield "", str(exc), "No GPU measurements recorded", json.dumps(record, indent=2)


def replay(path):
    if not path:
        return "Choose a recording JSON file.", "", "", ""
    try:
        rec = load_record(path)
        banner = f"RECORDED RUN · {rec.get('measurements', {}).get('gpu_name', 'GPU not recorded')} · {rec.get('measurements', {}).get('recorded_at', 'date not recorded')} · {rec['status']}"
        events = rec.get("events", [])
        trace = "\n".join(f"{e.get('kind')}: {json.dumps(e.get('data', {}), ensure_ascii=False)}" for e in events)
        return banner, rec.get("output", ""), trace, json.dumps(rec, indent=2)
    except Exception as exc:
        return f"Invalid recording: {exc}", "", "", ""


try:
    import gradio as gr
except ImportError as exc:
    raise SystemExit("Install demo/requirements-live.txt (or requirements-replay.txt) before launching.") from exc

css = """
body, .gradio-container { background: #f6f8fb !important; color: #172033 !important; }
.block { border-color: #d9e0ea !important; }
"""

with gr.Blocks(title="Adaptive MoE · Quantization Lab", theme=gr.themes.Soft(primary_hue="indigo"), css=css) as app:
    gr.Markdown("# Adaptive MoE · Quantization Lab\nExplore per-expert precision, routing, and measured GPU resources.")
    with gr.Tab("Live demo"):
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Completion prompt", lines=5, value="Solve step by step: If a train travels 120 km in 2 hours, what is its average speed?")
                gr.Examples(["Complete the sentence: Water freezes at", "Return JSON with keys name and status. Use name Ada and status ready.", "Solve step by step: What is 17 × 19?"], prompt)
                allocation = gr.Dropdown(["Uniform NF4", "Uniform INT8", "Uniform FP16", "Importance heuristic", "PPO + budget"], value="PPO + budget", label="Allocation")
                with gr.Row():
                    budget = gr.Number(value=8.0, minimum=0.1, maximum=80, label="PyTorch live allocation cap (GiB)")
                    battery = gr.Radio([100, 50, 10], value=100, label="Policy scenario · battery %")
                max_tokens = gr.Slider(1, 256, value=64, step=1, label="Maximum new tokens")
                with gr.Row():
                    run = gr.Button("Run", variant="primary")
                    stop = gr.Button("Stop")
            with gr.Column(scale=4):
                answer = gr.Textbox(label="Streamed answer", lines=9)
                measurements = gr.Code(label="Status and measurements", language="json")
                resources = gr.Markdown("No GPU measurements recorded")
                record = gr.Code(label="Latest run record", language="json")
        gr.Markdown("**Gate status:** live execution is enabled only after real GPU quantization and memory checks. PPO hardware values are a policy scenario; actual free VRAM must independently constrain deployment. Routing features are intended to come from an NF4 profiling pass.")
        run.click(task, [prompt, allocation, budget, battery, max_tokens], [answer, measurements, resources, record])
        stop.click(lambda: active_stop.set(), None, None)
    with gr.Tab("Experts and resources"):
        gr.Markdown("## Expert routing view\nThe grid and memory chart populate from recorded routing events. No synthetic activity or memory figures are shown when a GPU recording is unavailable.")
        gr.Markdown("Tier colors: **NF4** blue · **INT8** amber · **FP16** violet. Expert-index precision is shared across layers in this policy.")
        gr.Markdown("Live backend compatibility gate is currently closed.")
    with gr.Tab("Compare and evidence"):
        gr.Markdown("Comparisons run sequentially on the live backend and include profiling/switching time. A method that exceeds the chosen allocation cap must be reported as does not fit; no deliberate OOM probes.")
        upload = gr.File(label="Load recorded RunRecord JSON", file_types=[".json"], type="filepath")
        banner = gr.Markdown()
        replay_output = gr.Textbox(label="Recorded output", lines=5)
        replay_events = gr.Code(label="Recorded event trace", language="json")
        replay_record = gr.Code(label="Full RunRecord", language="json")
        upload.change(replay, upload, [banner, replay_output, replay_events, replay_record])
        export = gr.Button("Export current session JSON and CSV")
        files = gr.File(label="Exports")
        def do_export():
            paths = export_records(records, ROOT / "exports")
            return list(map(str, paths))
        export.click(do_export, None, files)

if __name__ == "__main__":
    app.queue().launch(server_name=os.getenv("DEMO_HOST", "127.0.0.1"), server_port=int(os.getenv("DEMO_PORT", "7860")))
