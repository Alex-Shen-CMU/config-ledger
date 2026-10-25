"""Stage 3: compile a sweep's results from its manifest. No Hydra recomposition.

    python aggregate.py --sweep hidden_sizes [--axis model.hidden ...] [--pin KEY=VALUE ...]

Axes default to whatever columns actually vary in the manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config_ledger import Ledger
from config_ledger.hydra import sweeps
from toy_mlp import pipeline

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--axis", action="append", default=None)
    parser.add_argument("--pin", action="append", default=[])
    parser.add_argument("--outputs", default=str(HERE / "outputs"))
    args = parser.parse_args()

    outputs = Path(args.outputs)
    ledger = Ledger(outputs / "ledger")
    table = sweeps.load_sweep(ledger, "evaluate", args.sweep)
    if args.pin:
        table = table.pin(dict(p.split("=", 1) for p in args.pin))
    axes = args.axis or table.axes()
    rows = []
    for point, folder_id in table.grid(axes).items():
        metrics_path = outputs / "evaluate" / folder_id / pipeline.METRICS
        record = dict(zip(axes, point))
        record["folder_id"] = folder_id
        if metrics_path.exists():
            record.update(json.loads(metrics_path.read_text()))
        else:
            record["test_mse"] = None  # not done yet, judged from the artifact
        rows.append(record)
    df = pd.DataFrame(rows)
    missing = int(df["test_mse"].isna().sum())
    print(df.to_string(index=False))
    if missing:
        print(f"\nWARNING: {missing}/{len(df)} grid points have no metrics yet")


if __name__ == "__main__":
    main()
