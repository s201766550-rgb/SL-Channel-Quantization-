import csv
import json
from pathlib import Path
from typing import Dict, List


class MetricLogger:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "metrics.csv"
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self._rows: List[Dict] = []
        self._fieldnames: List[str] = []

    def _rewrite_csv(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: row.get(k, "") for k in self._fieldnames})

    def log(self, row: Dict) -> None:
        for key in row.keys():
            if key not in self._fieldnames:
                self._fieldnames.append(key)
        self._rows.append(dict(row))
        self._rewrite_csv()
        with self.jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row) + "\n")

    def close(self) -> None:
        return None


def write_json(path: str, payload: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
