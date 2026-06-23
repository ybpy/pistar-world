#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get(
    "ADV_DATA_DIR",
    str(ROOT / "outputs/lerobot_policy_data/task7_demo_plus_policy1_2500_rollout100_progress_adv_vlm10k"),
))

stats: dict[str, dict[str, int]] = {
    "demo": {"positive": 0, "negative": 0, "none": 0, "other": 0, "frames": 0, "episodes": 0},
    "rollout": {"positive": 0, "negative": 0, "none": 0, "other": 0, "frames": 0, "episodes": 0},
}

paths = sorted((DATA_DIR / "data").rglob("*.parquet"))
if not paths:
    raise SystemExit(f"no parquet episodes under {DATA_DIR / 'data'}")

for path in paths:
    df = pd.read_parquet(path)
    if "intervention" not in df or "adv_ind" not in df:
        raise SystemExit(f"missing intervention/adv_ind in {path}")
    is_demo = bool((df["intervention"].astype(int) == 1).all())
    split = "demo" if is_demo else "rollout"
    stats[split]["episodes"] += 1
    stats[split]["frames"] += len(df)
    values = df["adv_ind"].fillna("none").astype(str)
    for value, count in values.value_counts().items():
        if value in ("positive", "negative", "none"):
            stats[split][value] += int(count)
        else:
            stats[split]["other"] += int(count)

print(json.dumps(stats, indent=2, sort_keys=True))
if stats["demo"]["negative"] or stats["demo"]["none"] or stats["demo"]["other"]:
    raise SystemExit("demo labels are not all positive")
