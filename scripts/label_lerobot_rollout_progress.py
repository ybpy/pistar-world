#!/usr/bin/env python3
"""Add progress-based reward/value labels to a real-policy LeRobot rollout dataset."""

from __future__ import annotations

import argparse
import io
import json
import os
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


def _compute_normalized_invquad_endpoint_progress(
    trajectories: list[Trajectory],
    *,
    invquad_lambda: float,
    p_fail_cap: float,
) -> None:
    raw_min = 1.0 / (1.0 + float(invquad_lambda))
    raw_span = max(1.0 - raw_min, 1e-6)
    for traj in trajectories:
        if traj.distances is None or len(traj.distances) == 0:
            traj.p_values = np.asarray([], dtype=np.float32)
            continue
        d = np.clip(np.asarray(traj.distances, dtype=np.float32), 0.0, 1.0)
        raw = 1.0 / (1.0 + float(invquad_lambda) * d * d)
        p_global = np.clip((raw - raw_min) / raw_span, 0.0, 1.0)
        if traj.success:
            denom = float(p_global[-1] - p_global[0])
            if abs(denom) > 1e-6:
                p = np.clip((p_global - p_global[0]) / denom, 0.0, 1.0)
            else:
                p = p_global
        else:
            p = np.minimum(p_global, float(p_fail_cap))
        traj.p_values = np.clip(p, 1e-6, 1.0).astype(np.float32)


def _compute_srpo_sigmoid_progress(
    trajectories: list[Trajectory],
    *,
    steepness: float,
    offset: float,
    failure_scale: float,
    success_scale: float,
    norm_source: str,
    q_low: float,
    q_high: float,
) -> dict[str, Any]:
    if norm_source == "failed_prefix_minmax":
        norm_distances = np.concatenate([
            np.asarray(traj.distances, dtype=np.float32)
            for traj in trajectories
            if not traj.success and traj.distances is not None and len(traj.distances)
        ])
        mode = "failed-prefix-minmax"
        if norm_distances.size == 0:
            raise ValueError("SRPO sigmoid failed-prefix normalization needs at least one failed trajectory")
        d_min = float(np.min(norm_distances))
        d_max = float(np.max(norm_distances))
    elif norm_source == "all_prefix_percentile_minmax":
        norm_distances = np.concatenate([
            np.asarray(traj.distances, dtype=np.float32)
            for traj in trajectories
            if traj.distances is not None and len(traj.distances)
        ])
        mode = "all-prefix-percentile-minmax"
        if norm_distances.size == 0:
            raise ValueError("SRPO sigmoid normalization needs at least one trajectory with prefix distances")
        d_min = float(np.quantile(norm_distances, q_low))
        d_max = float(np.quantile(norm_distances, q_high))
    else:
        raise ValueError(f"Unknown SRPO sigmoid normalization source: {norm_source}")

    denom = max(d_max - d_min, 1e-6)

    for traj in trajectories:
        if traj.distances is None or len(traj.distances) == 0:
            traj.p_values = np.asarray([], dtype=np.float32)
            continue
        d = np.asarray(traj.distances, dtype=np.float32)
        d_norm = np.clip((d - d_min) / denom, 0.0, 1.0)
        logits = np.clip(float(steepness) * (float(offset) - d_norm), -80.0, 80.0)
        scale = float(success_scale) if traj.success else float(failure_scale)
        p = scale / (1.0 + np.exp(-logits))
        traj.p_values = np.clip(p, 1e-6, 1.0).astype(np.float32)

    return {
        "mode": mode,
        "raw_min": d_min,
        "raw_max": d_max,
        "q_low": float(q_low) if norm_source == "all_prefix_percentile_minmax" else None,
        "q_high": float(q_high) if norm_source == "all_prefix_percentile_minmax" else None,
        "steepness": float(steepness),
        "offset": float(offset),
        "failure_scale": float(failure_scale),
        "success_scale": float(success_scale),
    }


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
    for column in ("success", "is_success", "episode_success", "terminal_success"):
        if column in df:
            return bool(_scalar(df[column].iloc[-1]))
    if "reward" in df:
        return float(_scalar(df["reward"].iloc[-1])) > 0.5
    if "value_label" in df:
        terminal_value = float(_scalar(df["value_label"].iloc[-1]))
        if np.isclose(terminal_value, 0.0, atol=1e-6):
            return True
        if np.isclose(terminal_value, -1.0, atol=1e-6):
            return False
    raise ValueError(
        "Cannot infer episode success from labels or rewards; add an explicit "
        "success/is_success/episode_success/terminal_success column or terminal reward/value_label"
    )


def _episode_source(df: pd.DataFrame) -> str:
    if "intervention" not in df:
        return "policy_rollout"
    values = np.asarray([_scalar(value) for value in df["intervention"].tolist()], dtype=np.int64)
    return "demo_success" if values.size and np.all(values == 1) else "policy_rollout"


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


def _encode_prefixes_cached(encoder: VJEPAEncoder, traj: Trajectory, batch_size: int = 8) -> np.ndarray:
    task_slug = _slugify(traj.task_name)
    embeddings: list[np.ndarray | None] = []
    missing: list[tuple[int, str, list[np.ndarray]]] = []
    for index, timestep in enumerate(traj.prefix_timesteps):
        key = encoder._make_cache_key(
            task_slug,
            "prefix",
            trajectory_id=traj.trajectory_id,
            source=traj.source,
            timestep=timestep,
        )
        embedding = encoder.load_cache(task_slug, key)
        if embedding is None:
            embeddings.append(None)
            missing.append((index, key, traj.frames[: timestep + 1]))
        else:
            embeddings.append(np.asarray(embedding, dtype=np.float32))

    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        encoded = encoder.encode_full_batch([item[2] for item in chunk], batch_size=batch_size)
        for (index, key, _), embedding in zip(chunk, encoded, strict=True):
            embedding = np.asarray(embedding, dtype=np.float32)
            encoder.save_cache(task_slug, key, embedding)
            embeddings[index] = embedding

    return np.asarray(embeddings, dtype=np.float32)


def _dense_progress(traj: Trajectory) -> np.ndarray:
    timesteps = np.asarray(traj.prefix_timesteps, dtype=np.int64)
    progress = np.asarray(traj.p_values, dtype=np.float32)
    dense = np.interp(np.arange(traj.episode_length), timesteps, progress).astype(np.float32)
    return np.clip(dense, 1e-6, 1.0)


def _dense_labels(traj: Trajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    progress = _dense_progress(traj)
    value = progress - 1.0
    reward = np.zeros_like(progress)
    if len(progress) > 1:
        reward[:-1] = progress[:-1] - progress[1:]
    if len(progress):
        reward[-1] = progress[-1] - 1.0
    return progress, value.astype(np.float32), reward.astype(np.float32)


def _resample_curve(values: np.ndarray, num_bins: int = 11) -> list[float]:
    if len(values) == 0:
        return []
    if len(values) == 1:
        return [float(values[0])] * num_bins
    x_old = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num_bins, dtype=np.float32)
    return [float(v) for v in np.interp(x_new, x_old, values)]


def _value_curve_report(trajectories: list[Trajectory]) -> dict[str, Any]:
    by_success = {True: [], False: []}
    all_records = []
    for traj in trajectories:
        progress, value, reward = _dense_labels(traj)
        finite = bool(np.all(np.isfinite(value)) and np.all(np.isfinite(reward)))
        diffs = np.diff(value) if len(value) > 1 else np.asarray([], dtype=np.float32)
        max_abs_step = float(np.max(np.abs(diffs))) if diffs.size else 0.0
        reverse_steps = int(np.sum(diffs < -1e-4)) if traj.success else int(np.sum(diffs > 1e-4))
        midpoint = int(round((len(value) - 1) * 0.5)) if len(value) else 0
        record = {
            "trajectory_id": traj.trajectory_id,
            "success": bool(traj.success),
            "length": int(len(value)),
            "terminal_progress": float(progress[-1]) if len(progress) else None,
            "terminal_value": float(value[-1]) if len(value) else None,
            "terminal_distance": float(traj.distances[-1]) if traj.distances is not None and len(traj.distances) else None,
            "start_value": float(value[0]) if len(value) else None,
            "mid_value": float(value[midpoint]) if len(value) else None,
            "end_value": float(value[-1]) if len(value) else None,
            "value_min": float(np.min(value)) if len(value) else None,
            "value_max": float(np.max(value)) if len(value) else None,
            "max_abs_step": max_abs_step,
            "reverse_steps": reverse_steps,
            "has_nan_or_inf": not finite,
            "curve_11": _resample_curve(value, 11),
        }
        by_success[traj.success].append(record)
        all_records.append(record)

    success_terms = np.asarray([r["terminal_value"] for r in by_success[True]], dtype=np.float32)
    failed_terms = np.asarray([r["terminal_value"] for r in by_success[False]], dtype=np.float32)

    def endpoint_stats(values: np.ndarray) -> dict[str, float | int | None]:
        if values.size == 0:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {
            "count": int(values.size),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }

    pair_ratio = None
    if success_terms.size and failed_terms.size:
        pair_ratio = float(np.mean(success_terms[:, None] > failed_terms[None, :]))

    bins = [float(x) for x in np.linspace(0.0, 1.0, 11, dtype=np.float32)]

    def mean_curve(records: list[dict[str, Any]]) -> list[float] | None:
        if not records:
            return None
        curves = np.asarray([r["curve_11"] for r in records], dtype=np.float32)
        return [float(v) for v in np.mean(curves, axis=0)]

    anomalies = [
        r
        for r in all_records
        if r["has_nan_or_inf"]
        or r["value_min"] is not None
        and (r["value_min"] < -1.00001 or r["value_max"] > 1.00001 or r["max_abs_step"] > 0.5)
    ]
    failed_not_below_success = []
    if success_terms.size:
        success_floor = float(np.min(success_terms))
        failed_not_below_success = [r for r in by_success[False] if r["terminal_value"] >= success_floor]

    return {
        "endpoint_value_stats": {
            "success": endpoint_stats(success_terms),
            "failed": endpoint_stats(failed_terms),
            "success_gt_failed_pair_ratio": pair_ratio,
        },
        "normalized_time_bins": bins,
        "mean_value_curve": {
            "success": mean_curve(by_success[True]),
            "failed": mean_curve(by_success[False]),
        },
        "sample_value_curves": {
            "success": by_success[True][:5],
            "failed": by_success[False][:5],
        },
        "anomalies": anomalies[:50],
        "failed_terminal_not_below_success_terminal": failed_not_below_success[:50],
    }


def _write_labels(path: Path, traj: Trajectory) -> dict[str, float]:
    df = pd.read_parquet(path)
    _, value, reward = _dense_labels(traj)

    df["success"] = bool(traj.success)
    df["value_label"] = value.astype(np.float32)
    df["value"] = value.astype(np.float32)
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


def _normalize_trajectory_distances(
    trajectories: list[Trajectory],
    *,
    mode: str,
    q_low: float,
    q_high: float,
) -> dict[str, Any]:
    if mode == "none":
        return {"mode": "none"}
    all_distances = np.concatenate([
        np.asarray(traj.distances, dtype=np.float32)
        for traj in trajectories
        if traj.distances is not None and len(traj.distances)
    ])
    if all_distances.size == 0:
        raise ValueError("Cannot normalize distances: no distances found")
    if mode == "minmax":
        d_min = float(np.min(all_distances))
        d_max = float(np.max(all_distances))
    elif mode == "percentile_minmax":
        d_min = float(np.quantile(all_distances, q_low))
        d_max = float(np.quantile(all_distances, q_high))
    else:
        raise ValueError(f"Unknown distance normalization mode: {mode}")
    denom = max(d_max - d_min, 1e-6)
    for traj in trajectories:
        d = np.asarray(traj.distances, dtype=np.float32)
        traj.distances = np.clip((d - d_min) / denom, 0.0, 1.0).astype(np.float32)
    return {
        "mode": mode,
        "q_low": q_low,
        "q_high": q_high,
        "raw_min": float(np.min(all_distances)),
        "raw_max": float(np.max(all_distances)),
        "raw_q_low": float(np.quantile(all_distances, q_low)),
        "raw_q_high": float(np.quantile(all_distances, q_high)),
        "d_min_used": d_min,
        "d_max_used": d_max,
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
        "description": "Progress-based value label using one rule for all trajectories: p_t - 1.",
    }
    features["value"] = {
        "dtype": "float32",
        "shape": [1],
        "names": ["value"],
        "description": "Alias of value_label for ValueDataLoader.",
    }
    features["reward_label"] = {
        "dtype": "float32",
        "shape": [1],
        "names": ["reward_label"],
        "description": "Progress-based reward: p_t - p_(t+1), terminal p_T - 1, so cumulative return equals p_t - 1.",
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
    parser.add_argument("--model_path", default=os.getenv("VJEPA_MODEL_PATH"))
    parser.add_argument("--cache_dir", default=str(REPO_ROOT / "cache" / "rollout_progress"))
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--p_fail_cap", type=float, default=0.9)
    parser.add_argument("--dbscan_eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=2)
    parser.add_argument("--prefix_batch_size", type=int, default=8)
    parser.add_argument("--distance_normalization", choices=["none", "minmax", "percentile_minmax"], default="none")
    parser.add_argument("--distance_q_low", type=float, default=0.01)
    parser.add_argument("--distance_q_high", type=float, default=0.99)
    parser.add_argument("--sigmoid_beta", type=float, default=None)
    parser.add_argument(
        "--progress_activation",
        choices=["sigmoid", "normalized_invquad_endpoint", "srpo_sigmoid"],
        default="sigmoid",
    )
    parser.add_argument("--invquad_lambda", type=float, default=4.0)
    parser.add_argument("--srpo_sigmoid_steepness", type=float, default=10.0)
    parser.add_argument("--srpo_sigmoid_offset", type=float, default=0.5)
    parser.add_argument("--srpo_failure_scale", type=float, default=0.6)
    parser.add_argument("--srpo_success_scale", type=float, default=1.0)
    parser.add_argument(
        "--srpo_sigmoid_norm_source",
        choices=["failed_prefix_minmax", "all_prefix_percentile_minmax"],
        default="all_prefix_percentile_minmax",
    )
    parser.add_argument(
        "--center_source",
        choices=["all_success", "demo_success"],
        default="all_success",
        help="Which successful trajectories to use as DBSCAN center candidates.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.model_path:
        raise ValueError("Set --model_path or VJEPA_MODEL_PATH to the V-JEPA2 vitg-384.pt weights")

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
        source = _episode_source(df)
        grouped[task_index].append(
            (
                path,
                Trajectory(
                    trajectory_id=f"episode_{episode_index:06d}",
                    task_name=tasks[task_index],
                    source=source,
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
        center_candidates = successes
        if args.center_source == "demo_success":
            center_candidates = [traj for traj in successes if traj.source == "demo_success"]
        if not successes or not failures:
            raise ValueError(
                f"Task {task_index} ({tasks[task_index]}) needs both success and failure rollouts; "
                f"found success={len(successes)}, failure={len(failures)}"
            )
        if not center_candidates:
            raise ValueError(
                f"Task {task_index} ({tasks[task_index]}) has no center candidates for "
                f"center_source={args.center_source}"
            )

        for traj in trajectories:
            traj.prefix_timesteps = _prefix_timesteps(traj.episode_length, args.prefix_stride)
            traj.prefix_embeddings = _encode_prefixes_cached(encoder, traj, batch_size=args.prefix_batch_size)
            traj.full_embedding = traj.prefix_embeddings[-1]
            traj.frames = None

        full_embeddings = np.asarray([traj.full_embedding for traj in center_candidates])
        clustering = cluster_full_success_trajectories(
            full_embeddings,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
        )

        failed_distances, success_distances = compute_all_distances(trajectories, clustering.centers)
        distance_normalization = _normalize_trajectory_distances(
            trajectories,
            mode=args.distance_normalization,
            q_low=args.distance_q_low,
            q_high=args.distance_q_high,
        )
        if args.distance_normalization == "none":
            d_bar, sigma_d, stat_source = compute_statistics(failed_distances, success_distances)
        else:
            d_bar = 0.5 if args.sigmoid_beta is None else float(args.sigmoid_beta)
            sigma_d = 1.0
            stat_source = f"{args.distance_normalization}; sigmoid_beta={d_bar}"
        if args.progress_activation == "sigmoid":
            compute_progress(
                trajectories,
                d_bar=d_bar,
                sigma_d=sigma_d,
                alpha=args.alpha,
                p_fail_cap=args.p_fail_cap,
                stat_source=stat_source,
            )
        elif args.progress_activation == "normalized_invquad_endpoint":
            _compute_normalized_invquad_endpoint_progress(
                trajectories,
                invquad_lambda=args.invquad_lambda,
                p_fail_cap=args.p_fail_cap,
            )
        elif args.progress_activation == "srpo_sigmoid":
            distance_normalization = _compute_srpo_sigmoid_progress(
                trajectories,
                steepness=args.srpo_sigmoid_steepness,
                offset=args.srpo_sigmoid_offset,
                failure_scale=args.srpo_failure_scale,
                success_scale=args.srpo_success_scale,
                norm_source=args.srpo_sigmoid_norm_source,
                q_low=args.distance_q_low,
                q_high=args.distance_q_high,
            )
            d_bar = args.srpo_sigmoid_offset
            sigma_d = 1.0
            stat_source = (
                f"srpo-style sigmoid; norm_source={args.srpo_sigmoid_norm_source}; "
                "success/failure use separate scales; no fail cap"
            )
        else:
            raise ValueError(f"Unknown progress activation: {args.progress_activation}")
        compute_rewards_and_values(trajectories)
        validation = validate_value_labels(trajectories)
        value_curves = _value_curve_report(trajectories)
        write_stats = [_write_labels(path, traj) for path, traj in entries]
        report["tasks"][str(task_index)] = {
            "task": tasks[task_index],
            "episodes": len(trajectories),
            "successes": len(successes),
            "failures": len(failures),
            "center_source": args.center_source,
            "center_candidate_episodes": len(center_candidates),
            "success_centers": int(clustering.num_centers),
            "prefix_stride": args.prefix_stride,
            "prefix_definition": "prefix lengths stride, 2*stride, ... plus full trajectory",
            "total_prefixes": sum(len(traj.prefix_timesteps) for traj in trajectories),
            "d_bar": d_bar,
            "sigma_d": sigma_d,
            "stat_source": stat_source,
            "distance_normalization": distance_normalization,
            "sigmoid_alpha": args.alpha,
            "sigmoid_beta": d_bar,
            "progress_activation": args.progress_activation,
            "invquad_lambda": args.invquad_lambda if args.progress_activation == "normalized_invquad_endpoint" else None,
            "srpo_sigmoid_steepness": args.srpo_sigmoid_steepness if args.progress_activation == "srpo_sigmoid" else None,
            "srpo_sigmoid_offset": args.srpo_sigmoid_offset if args.progress_activation == "srpo_sigmoid" else None,
            "srpo_failure_scale": args.srpo_failure_scale if args.progress_activation == "srpo_sigmoid" else None,
            "srpo_success_scale": args.srpo_success_scale if args.progress_activation == "srpo_sigmoid" else None,
            "srpo_sigmoid_norm_source": args.srpo_sigmoid_norm_source if args.progress_activation == "srpo_sigmoid" else None,
            "p_fail_cap": None if args.progress_activation == "srpo_sigmoid" else args.p_fail_cap,
            "sparse_validation": validation,
            "value_curve_report": value_curves,
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
