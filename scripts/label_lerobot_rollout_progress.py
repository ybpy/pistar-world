#!/usr/bin/env python3
"""Add progress-based reward/value labels to a real-policy LeRobot rollout dataset."""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from progress.clustering import cluster_full_success_trajectories
from progress.encoder import VJEPAEncoder
from progress.scoring import compute_all_distances, compute_progress, compute_statistics
from progress.value_labels import compute_rewards_and_values, validate_value_labels


@dataclass
class Trajectory:
    trajectory_id: str
    task_name: str
    source: str
    success: bool
    frames: list[np.ndarray] | None
    episode_length: int
    prefix_timesteps: list[int] = field(default_factory=list)
    prefix_embeddings: np.ndarray | None = None
    full_embedding: np.ndarray | None = None
    distances: np.ndarray | None = None
    p_values: np.ndarray | None = None
    rewards: np.ndarray | None = None
    value_labels: np.ndarray | None = None


def _scalar(value: Any) -> Any:
    if isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        for dtype in (np.float32, np.int64, np.int32):
            parsed = np.frombuffer(raw, dtype=dtype)
            if parsed.size == 1:
                return parsed[0].item()
        return raw
    if isinstance(value, (np.ndarray, list, tuple)):
        array = np.asarray(value).reshape(-1)
        return array[0].item() if array.size else None
    return value.item() if hasattr(value, "item") else value


def _decode_image(value: Any, dataset_root: Path) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path"):
            value = dataset_root / value["path"]
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return np.asarray(image.convert("RGB"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(value))) as image:
            return np.asarray(image.convert("RGB"))
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"Unsupported image value with shape {array.shape}")
    return array.astype(np.uint8, copy=False)


def _load_tasks(data_dir: Path) -> dict[int, str]:
    tasks = {}
    with open(data_dir / "meta" / "tasks.jsonl") as file:
        for line in file:
            record = json.loads(line)
            tasks[int(record["task_index"])] = str(record["task"])
    return tasks


def _episode_success(df: pd.DataFrame) -> bool:
    for column in ("success", "done", "is_success"):
        if column in df:
            return bool(_scalar(df[column].iloc[-1]))
    if "reward" in df and float(_scalar(df["reward"].iloc[-1])) > 0:
        return True
    if "value_label" in df:
        return bool(np.isclose(float(_scalar(df["value_label"].iloc[-1])), 0.0, atol=1e-6))
    raise ValueError("Cannot infer episode success; add a success/done/reward/value_label column")


def _prefix_timesteps(length: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    if stride <= 0:
        raise ValueError("prefix stride must be positive")
    # A stride of 5 means prefixes containing frames [0:5], [0:10], ...
    timesteps = list(range(stride - 1, length, stride))
    if not timesteps or timesteps[-1] != length - 1:
        timesteps.append(length - 1)
    return timesteps


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:80] or "task"


def _encode_prefixes_cached(encoder: VJEPAEncoder, traj: Trajectory) -> np.ndarray:
    task_slug = _slugify(traj.task_name)
    embeddings = []
    for timestep in traj.prefix_timesteps:
        key = encoder._make_cache_key(
            task_slug,
            "prefix",
            trajectory_id=traj.trajectory_id,
            source=traj.source,
            timestep=timestep,
        )
        embedding = encoder.load_cache(task_slug, key)
        if embedding is None:
            embedding = encoder.encode_full(traj.frames[: timestep + 1])
            encoder.save_cache(task_slug, key, embedding)
        embeddings.append(np.asarray(embedding, dtype=np.float32))
    return np.asarray(embeddings, dtype=np.float32)


def _dense_progress(traj: Trajectory) -> np.ndarray:
    timesteps = np.asarray(traj.prefix_timesteps, dtype=np.int64)
    progress = np.asarray(traj.p_values, dtype=np.float32)
    dense = np.interp(np.arange(traj.episode_length), timesteps, progress).astype(np.float32)
    if traj.success:
        dense[-1] = 1.0
    return np.clip(dense, 1e-6, 1.0)


def _write_labels(path: Path, traj: Trajectory) -> dict[str, float]:
    df = pd.read_parquet(path)
    progress = _dense_progress(traj)
    value = progress - 1.0
    reward = np.zeros_like(progress)
    if len(progress) > 1:
        reward[:-1] = progress[:-1] - progress[1:]
    reward[-1] = 0.0 if traj.success else progress[-1] - 1.0

    df["success"] = bool(traj.success)
    df["value_label"] = value.astype(np.float32)
    df["reward_label"] = reward.astype(np.float32)
    if "intervention" not in df:
        df["intervention"] = np.int64(0)
    if "adv_ind" not in df:
        df["adv_ind"] = "none"
    df.to_parquet(path, index=False)
    return {
        "value_min": float(value.min()),
        "value_max": float(value.max()),
        "reward_min": float(reward.min()),
        "reward_max": float(reward.max()),
    }


def _update_info(data_dir: Path) -> None:
    path = data_dir / "meta" / "info.json"
    with open(path) as file:
        info = json.load(file)
    features = info.setdefault("features", {})
    features["success"] = {"dtype": "bool", "shape": [1], "names": ["success"]}
    features["intervention"] = {"dtype": "int64", "shape": [1], "names": ["intervention_flag"]}
    features["value_label"] = {
        "dtype": "float32",
        "shape": [1],
        "names": ["value_label"],
        "description": "Progress-based value label p_t - 1.",
    }
    features["reward_label"] = {
        "dtype": "float32",
        "shape": [1],
        "names": ["reward_label"],
        "description": "Progress-based reward p_t - p_(t+1), with terminal rule.",
    }
    features["adv_ind"] = {"dtype": "string", "shape": [1], "names": ["adv_ind"]}
    with open(path, "w") as file:
        json.dump(info, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix_stride", type=int, default=5)
    parser.add_argument("--image_col", default="image")
    parser.add_argument("--model_path", default="/public/home/chenyuyao1/model/vjepa2/vitg-384.pt")
    parser.add_argument("--cache_dir", default=str(REPO_ROOT / "cache" / "rollout_progress"))
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--p_fail_cap", type=float, default=0.9)
    parser.add_argument("--dbscan_eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)

    tasks = _load_tasks(output_dir)
    grouped: dict[int, list[tuple[Path, Trajectory]]] = defaultdict(list)
    for path in sorted((output_dir / "data").rglob("*.parquet")):
        df = pd.read_parquet(path)
        task_index = int(_scalar(df["task_index"].iloc[0]))
        episode_index = int(_scalar(df["episode_index"].iloc[0]))
        frames = [_decode_image(value, output_dir) for value in df[args.image_col].tolist()]
        success = _episode_success(df)
        grouped[task_index].append(
            (
                path,
                Trajectory(
                    trajectory_id=f"episode_{episode_index:06d}",
                    task_name=tasks[task_index],
                    source="policy_rollout",
                    success=success,
                    frames=frames,
                    episode_length=len(frames),
                ),
            )
        )

    encoder = VJEPAEncoder(
        model_path=args.model_path,
        device_id=args.device_id,
        enable_fp16=True,
        cache_dir=args.cache_dir,
    )
    report = {"input_dir": str(input_dir), "output_dir": str(output_dir), "tasks": {}}
    for task_index, entries in sorted(grouped.items()):
        trajectories = [traj for _, traj in entries]
        successes = [traj for traj in trajectories if traj.success]
        failures = [traj for traj in trajectories if not traj.success]
        if not successes or not failures:
            raise ValueError(
                f"Task {task_index} ({tasks[task_index]}) needs both success and failure rollouts; "
                f"found success={len(successes)}, failure={len(failures)}"
            )

        for traj in trajectories:
            traj.prefix_timesteps = _prefix_timesteps(traj.episode_length, args.prefix_stride)
            traj.prefix_embeddings = _encode_prefixes_cached(encoder, traj)
            traj.full_embedding = traj.prefix_embeddings[-1]
            traj.frames = None

        full_embeddings = np.asarray([traj.full_embedding for traj in successes])
        clustering = cluster_full_success_trajectories(
            full_embeddings,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
        )

        failed_distances, success_distances = compute_all_distances(trajectories, clustering.centers)
        d_bar, sigma_d, stat_source = compute_statistics(failed_distances, success_distances)
        compute_progress(
            trajectories,
            d_bar=d_bar,
            sigma_d=sigma_d,
            alpha=args.alpha,
            p_fail_cap=args.p_fail_cap,
            stat_source=stat_source,
        )
        compute_rewards_and_values(trajectories)
        validation = validate_value_labels(trajectories)
        write_stats = [_write_labels(path, traj) for path, traj in entries]
        report["tasks"][str(task_index)] = {
            "task": tasks[task_index],
            "episodes": len(trajectories),
            "successes": len(successes),
            "failures": len(failures),
            "success_centers": int(clustering.num_centers),
            "prefix_stride": args.prefix_stride,
            "prefix_definition": "prefix lengths stride, 2*stride, ... plus full trajectory",
            "total_prefixes": sum(len(traj.prefix_timesteps) for traj in trajectories),
            "d_bar": d_bar,
            "sigma_d": sigma_d,
            "stat_source": stat_source,
            "sparse_validation": validation,
            "dense_value_min": min(item["value_min"] for item in write_stats),
            "dense_value_max": max(item["value_max"] for item in write_stats),
            "dense_reward_min": min(item["reward_min"] for item in write_stats),
            "dense_reward_max": max(item["reward_max"] for item in write_stats),
        }

    _update_info(output_dir)
    with open(output_dir / "progress_label_report.json", "w") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
