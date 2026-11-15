from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_record(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError("Choose a single RunRecord JSON object; exported session files may contain multiple records")
        data = data[0]
    if not isinstance(data, dict) or "events" not in data or "status" not in data:
        raise ValueError("Recording must be a RunRecord JSON object with events and status")
    return data


def export_records(records: list[dict[str, Any]], directory: str | Path) -> tuple[Path, Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = out / "runs.json", out / "runs.csv"
    json_path.write_text(json.dumps(records, indent=2))
    fields = ["status", "allocation", "budget_gib", "battery_percent", "max_new_tokens", "output", "timings_s", "measurements"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            req = record.get("request", {})
            writer.writerow({"status": record.get("status"), "allocation": req.get("allocation"),
                             "budget_gib": req.get("budget_gib"), "battery_percent": req.get("battery_percent"),
                             "max_new_tokens": req.get("max_new_tokens"), "output": record.get("output"),
                             "timings_s": json.dumps(record.get("timings_s", {})),
                             "measurements": json.dumps(record.get("measurements", {}))})
    return json_path, csv_path
