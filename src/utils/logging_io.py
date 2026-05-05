import csv
import json
from pathlib import Path
from typing import Dict, Optional


class MetricLogger:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "metrics.csv"
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_fp = self.csv_path.open("w", newline="", encoding="utf-8")

    def log(self, row: Dict) -> None:
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_fp.flush()
        with self.jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row) + "\n")

    def close(self) -> None:
        self._csv_fp.close()


def write_json(path: str, payload: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
