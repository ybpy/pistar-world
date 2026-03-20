#!/usr/bin/env python3
"""Merge multiple LeRobot datasets into one dataset in a deterministic order.

Example:
    python scripts/merge_lerobot_datasets.py \
        --sources \
          /home/chaihoa/project_wang/dataset/rollout/piper_plug_task_rl \
          /home/chaihoa/project_wang/dataset/datasets/lerobot_datasets/piper_plug_libero \
        --output /home/chaihoa/project_wang/dataset/datasets/lerobot_datasets/piper_plug_merged \
        --overwrite
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Array2D, Array3D, Array4D, Array5D, Dataset, Features, Image, Sequence, Value

PARQUET_RE = re.compile(r"episode_(\d+)\.parquet$")
VIDEO_RE = re.compile(r"episode_(\d+)\.mp4$")
COERCE_FIXED_SIZE_COLUMNS = ("state", "actions")


@dataclass
class SourceDataset:
    root: Path
    info: dict[str, Any]
    tasks_by_index: dict[int, str]
    episodes_by_index: dict[int, dict[str, Any]]
    stats_by_index: dict[int, dict[str, Any]]
    episode_files: dict[int, Path]
    video_files: dict[int, list[tuple[str, Path]]]
    selected_episode_indices: list[int]


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def discover_episode_files(root: Path) -> dict[int, Path]:
    episode_files: dict[int, Path] = {}
    for fpath in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        if not fpath.is_file():
            continue
        match = PARQUET_RE.match(fpath.name)
        if not match:
            continue
        episode_idx = int(match.group(1))
        episode_files[episode_idx] = fpath
    return episode_files


def discover_video_files(root: Path) -> dict[int, list[tuple[str, Path]]]:
    video_files: dict[int, list[tuple[str, Path]]] = {}
    videos_root = root / "videos"
    if not videos_root.exists():
        return video_files

    for fpath in sorted(videos_root.glob("chunk-*/**/episode_*.mp4")):
        if not fpath.is_file():
            continue
        match = VIDEO_RE.match(fpath.name)
        if not match:
            continue

        ep_idx = int(match.group(1))
        rel = fpath.relative_to(videos_root)
        # rel format: chunk-XYZ/<video_key...>/episode_XXXXXX.mp4
        parts = rel.parts
        if len(parts) < 3:
            continue
        video_key = str(Path(*parts[1:-1]))
        video_files.setdefault(ep_idx, []).append((video_key, fpath))

    return video_files


def select_episode_indices(
    root: Path,
    meta_indices: set[int],
    file_indices: set[int],
    selector: str,
    strict: bool,
) -> list[int]:
    if selector == "intersection":
        selected = sorted(meta_indices & file_indices)
    elif selector == "files":
        selected = sorted(file_indices)
    elif selector == "meta":
        selected = sorted(meta_indices)
    else:
        raise ValueError(f"Unknown selector: {selector}")

    missing_files = [ep for ep in selected if ep not in file_indices]
    if missing_files:
        msg = (
            f"{root}: {len(missing_files)} selected episodes are missing parquet files. "
            f"Example missing episode index: {missing_files[0]}"
        )
        if strict:
            raise ValueError(msg)
        warn(msg + " (skipped)")
        selected = [ep for ep in selected if ep in file_indices]

    return selected


def load_source_dataset(
    root: Path,
    selector: str,
    strict: bool,
    max_episodes_per_source: int | None,
) -> SourceDataset:
    meta_root = root / "meta"
    info_path = meta_root / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json in dataset: {root}")

    info = load_json(info_path)

    tasks_rows = load_jsonl(meta_root / "tasks.jsonl")
    tasks_by_index = {
        int(row["task_index"]): str(row["task"])
        for row in sorted(tasks_rows, key=lambda x: int(x["task_index"]))
    }

    episode_rows = load_jsonl(meta_root / "episodes.jsonl")
    episodes_by_index = {
        int(row["episode_index"]): row
        for row in sorted(episode_rows, key=lambda x: int(x["episode_index"]))
    }

    stats_rows = load_jsonl(meta_root / "episodes_stats.jsonl")
    stats_by_index = {
        int(row["episode_index"]): row
        for row in stats_rows
        if isinstance(row, dict) and "episode_index" in row
    }

    episode_files = discover_episode_files(root)
    video_files = discover_video_files(root)

    selected = select_episode_indices(
        root=root,
        meta_indices=set(episodes_by_index),
        file_indices=set(episode_files),
        selector=selector,
        strict=strict,
    )

    if max_episodes_per_source is not None:
        selected = selected[:max_episodes_per_source]

    if not selected:
        warn(f"{root}: no episodes selected after filtering.")

    return SourceDataset(
        root=root,
        info=info,
        tasks_by_index=tasks_by_index,
        episodes_by_index=episodes_by_index,
        stats_by_index=stats_by_index,
        episode_files=episode_files,
        video_files=video_files,
        selected_episode_indices=selected,
    )


def normalize_feature(ft: dict[str, Any]) -> dict[str, Any]:
    shape = ft.get("shape")
    if shape is None:
        shape = []
    elif isinstance(shape, tuple):
        shape = list(shape)

    return {
        "dtype": ft.get("dtype"),
        "shape": shape,
        "names": ft.get("names"),
    }


def merge_features(sources: list[SourceDataset], strict: bool) -> OrderedDict[str, dict[str, Any]]:
    merged: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for src in sources:
        src_features = src.info.get("features", {})
        for key, ft in src_features.items():
            cur = normalize_feature(ft)
            if key not in merged:
                merged[key] = cur
                continue

            prev = merged[key]
            if cur == prev:
                continue

            if prev["dtype"] != cur["dtype"]:
                raise ValueError(
                    f"Feature '{key}' dtype mismatch: '{prev['dtype']}' vs '{cur['dtype']}'"
                )

            if prev["dtype"] in ("image", "video"):
                if prev.get("names") != cur.get("names"):
                    msg = (
                        f"Feature '{key}' names mismatch in image/video feature: "
                        f"{prev.get('names')} vs {cur.get('names')}"
                    )
                    if strict:
                        raise ValueError(msg)
                    warn(msg + " (keeping the first one)")

                if prev.get("shape") != cur.get("shape"):
                    warn(
                        f"Feature '{key}' shape mismatch for image/video: "
                        f"{prev.get('shape')} vs {cur.get('shape')} (keeping the first one)"
                    )
                continue

            msg = (
                f"Feature '{key}' mismatch: first={prev}, current={cur}. "
                "Only image/video shape mismatch is tolerated by default."
            )
            if strict:
                raise ValueError(msg)
            warn(msg + " (keeping the first one)")

    return merged


def infer_scalar_dtype_from_arrow(arrow_dtype: pa.DataType) -> str:
    if pa.types.is_int8(arrow_dtype):
        return "int8"
    if pa.types.is_int16(arrow_dtype):
        return "int16"
    if pa.types.is_int32(arrow_dtype):
        return "int32"
    if pa.types.is_int64(arrow_dtype):
        return "int64"
    if pa.types.is_uint8(arrow_dtype):
        return "uint8"
    if pa.types.is_uint16(arrow_dtype):
        return "uint16"
    if pa.types.is_uint32(arrow_dtype):
        return "uint32"
    if pa.types.is_uint64(arrow_dtype):
        return "uint64"
    if pa.types.is_float16(arrow_dtype):
        return "float16"
    if pa.types.is_float32(arrow_dtype):
        return "float32"
    if pa.types.is_float64(arrow_dtype):
        return "float64"
    if pa.types.is_boolean(arrow_dtype):
        return "bool"
    raise ValueError(f"Unsupported scalar arrow dtype: {arrow_dtype}")


def infer_feature_from_arrow(name: str, arrow_dtype: pa.DataType) -> dict[str, Any]:
    if pa.types.is_struct(arrow_dtype):
        has_bytes = arrow_dtype.get_field_index("bytes") >= 0
        has_path = arrow_dtype.get_field_index("path") >= 0
        if has_bytes and has_path:
            # Resolution is unknown from schema only; we use a placeholder that keeps channel-first convention.
            return {
                "dtype": "image",
                "shape": [3, 1, 1],
                "names": ["channels", "height", "width"],
            }
        raise ValueError(f"Cannot infer feature for struct field '{name}' with dtype: {arrow_dtype}")

    if pa.types.is_fixed_size_list(arrow_dtype):
        return {
            "dtype": infer_scalar_dtype_from_arrow(arrow_dtype.value_type),
            "shape": [arrow_dtype.list_size],
            "names": None,
        }

    if pa.types.is_integer(arrow_dtype) or pa.types.is_floating(arrow_dtype) or pa.types.is_boolean(arrow_dtype):
        return {
            "dtype": infer_scalar_dtype_from_arrow(arrow_dtype),
            "shape": [1],
            "names": None,
        }

    raise ValueError(f"Cannot infer feature for field '{name}' with dtype: {arrow_dtype}")


def _is_list_like_dtype(dtype: pa.DataType) -> bool:
    return pa.types.is_fixed_size_list(dtype) or pa.types.is_list(dtype) or pa.types.is_large_list(dtype)


def _list_value_dtype(dtype: pa.DataType) -> pa.DataType:
    if not _is_list_like_dtype(dtype):
        raise ValueError(f"Expected list-like dtype, got: {dtype}")
    return dtype.value_type


def _max_observed_list_length(sources: list[SourceDataset], column: str) -> int:
    max_len = 0
    for src in sources:
        for old_ep_idx in src.selected_episode_indices:
            ep_path = src.episode_files[old_ep_idx]
            schema = pq.read_schema(ep_path)
            idx = schema.get_field_index(column)
            if idx < 0:
                continue

            col_type = schema.field(idx).type
            if pa.types.is_fixed_size_list(col_type):
                max_len = max(max_len, int(col_type.list_size))
                continue

            if not (pa.types.is_list(col_type) or pa.types.is_large_list(col_type)):
                continue

            arr = pq.read_table(ep_path, columns=[column]).column(0).combine_chunks()
            if len(arr) == 0:
                continue

            offsets = np.asarray(arr.offsets)
            lengths = offsets[1:] - offsets[:-1]
            if arr.null_count > 0:
                valid_mask = np.asarray(arr.is_valid())
                lengths = lengths[valid_mask]
            if lengths.size > 0:
                max_len = max(max_len, int(np.max(lengths)))
    return max_len


def build_global_arrow_types(
    sources: list[SourceDataset],
) -> tuple[OrderedDict[str, pa.DataType], dict[str, int]]:
    global_types: OrderedDict[str, pa.DataType] = OrderedDict()
    coerce_dims: dict[str, int] = {}

    for src in sources:
        for old_ep_idx in src.selected_episode_indices:
            schema = pq.read_schema(src.episode_files[old_ep_idx])
            for field in schema:
                if field.name not in global_types:
                    global_types[field.name] = field.type
                    continue

                if global_types[field.name].equals(field.type):
                    continue

                prev_type = global_types[field.name]
                cur_type = field.type
                if (
                    field.name in COERCE_FIXED_SIZE_COLUMNS
                    and _is_list_like_dtype(prev_type)
                    and _is_list_like_dtype(cur_type)
                ):
                    prev_value = _list_value_dtype(prev_type)
                    cur_value = _list_value_dtype(cur_type)
                    if not prev_value.equals(cur_value):
                        raise ValueError(
                            f"Column '{field.name}' list value type mismatch across datasets: "
                            f"{prev_value} vs {cur_value}"
                        )
                    continue

                if not prev_type.equals(cur_type):
                    raise ValueError(
                        f"Column '{field.name}' has inconsistent arrow type across datasets: "
                        f"{prev_type} vs {cur_type}"
                    )

    for col in COERCE_FIXED_SIZE_COLUMNS:
        if col not in global_types or not _is_list_like_dtype(global_types[col]):
            continue
        max_len = _max_observed_list_length(sources, col)
        if max_len <= 0:
            col_type = global_types[col]
            if pa.types.is_fixed_size_list(col_type):
                max_len = int(col_type.list_size)
            else:
                raise ValueError(
                    f"Cannot determine target fixed size for '{col}'. "
                    "No non-empty rows observed and no fixed-size schema available."
                )
        value_type = _list_value_dtype(global_types[col])
        global_types[col] = pa.list_(value_type, max_len)
        coerce_dims[col] = max_len

    return global_types, coerce_dims


def build_hf_feature(ft: dict[str, Any]):
    dtype = ft["dtype"]
    shape = tuple(ft.get("shape", []))

    if dtype == "image":
        return Image()
    if dtype == "video":
        return None

    if shape == (1,):
        return Value(dtype=dtype)
    if len(shape) == 1:
        return Sequence(feature=Value(dtype=dtype), length=shape[0])
    if len(shape) == 2:
        return Array2D(shape=shape, dtype=dtype)
    if len(shape) == 3:
        return Array3D(shape=shape, dtype=dtype)
    if len(shape) == 4:
        return Array4D(shape=shape, dtype=dtype)
    if len(shape) == 5:
        return Array5D(shape=shape, dtype=dtype)

    raise ValueError(f"Unsupported feature spec: dtype={dtype}, shape={shape}")


def build_target_schema(
    target_columns: list[str],
    merged_features: OrderedDict[str, dict[str, Any]],
) -> tuple[pa.Schema, dict[bytes, bytes] | None]:
    hf_feature_map: OrderedDict[str, Any] = OrderedDict()

    for col in target_columns:
        hf_ft = build_hf_feature(merged_features[col])
        if hf_ft is None:
            raise ValueError(
                f"Column '{col}' maps to video feature, but video features should not be in parquet columns."
            )
        hf_feature_map[col] = hf_ft

    hf_features = Features(hf_feature_map)
    empty_ds = Dataset.from_dict({k: [] for k in hf_feature_map}, features=hf_features)
    schema_with_meta = empty_ds.data.table.schema
    schema_no_meta = schema_with_meta.remove_metadata()
    return schema_no_meta, schema_with_meta.metadata


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def link_or_copy_file(src: Path, dst: Path, mode: str) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError as e:
            warn(f"Hardlink failed for {src} -> {dst}: {e}. Falling back to copy.")
            shutil.copy2(src, dst)
            return

    if mode == "symlink":
        os.symlink(src, dst)
        return

    raise ValueError(f"Unknown link mode: {mode}")


def set_or_add_column(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx >= 0:
        return table.set_column(idx, name, array)
    return table.append_column(name, array)


def default_array_for_field(field: pa.Field, length: int) -> pa.Array:
    dtype = field.type

    if pa.types.is_integer(dtype):
        return pa.array(np.zeros(length, dtype=np.int64), type=dtype)

    if pa.types.is_floating(dtype):
        np_dtype = np.float32 if pa.types.is_float32(dtype) else np.float64
        return pa.array(np.zeros(length, dtype=np_dtype), type=dtype)

    if pa.types.is_boolean(dtype):
        return pa.array(np.zeros(length, dtype=np.bool_), type=dtype)

    if pa.types.is_fixed_size_list(dtype):
        list_size = dtype.list_size
        value_dtype = dtype.value_type
        if pa.types.is_integer(value_dtype):
            values = np.zeros(length * list_size, dtype=np.int64)
        elif pa.types.is_floating(value_dtype):
            values = np.zeros(length * list_size, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported fixed-size-list value type: {value_dtype}")
        return pa.FixedSizeListArray.from_arrays(pa.array(values, type=value_dtype), list_size)

    if pa.types.is_struct(dtype):
        has_bytes = dtype.get_field_index("bytes") >= 0
        has_path = dtype.get_field_index("path") >= 0
        if has_bytes and has_path:
            raise ValueError(
                f"Cannot auto-fill missing image-like field '{field.name}'. "
                "Please ensure all source datasets have this image column."
            )

    raise ValueError(f"Unsupported field type for default fill: {field}")


def _arrow_scalar_to_numpy_dtype(dtype: pa.DataType) -> np.dtype:
    if pa.types.is_int8(dtype):
        return np.dtype(np.int8)
    if pa.types.is_int16(dtype):
        return np.dtype(np.int16)
    if pa.types.is_int32(dtype):
        return np.dtype(np.int32)
    if pa.types.is_int64(dtype):
        return np.dtype(np.int64)
    if pa.types.is_uint8(dtype):
        return np.dtype(np.uint8)
    if pa.types.is_uint16(dtype):
        return np.dtype(np.uint16)
    if pa.types.is_uint32(dtype):
        return np.dtype(np.uint32)
    if pa.types.is_uint64(dtype):
        return np.dtype(np.uint64)
    if pa.types.is_float16(dtype):
        return np.dtype(np.float16)
    if pa.types.is_float32(dtype):
        return np.dtype(np.float32)
    if pa.types.is_float64(dtype):
        return np.dtype(np.float64)
    if pa.types.is_boolean(dtype):
        return np.dtype(np.bool_)
    raise ValueError(f"Unsupported scalar value type: {dtype}")


def coerce_chunked_list_to_fixed_size(
    chunked: pa.ChunkedArray,
    target_dtype: pa.FixedSizeListType,
) -> tuple[pa.Array, int, int]:
    src_type = chunked.type
    if src_type.equals(target_dtype):
        return chunked.combine_chunks(), 0, 0

    if not _is_list_like_dtype(src_type):
        raise ValueError(f"Cannot coerce non-list column from {src_type} to {target_dtype}")

    src_value_type = _list_value_dtype(src_type)
    target_value_type = target_dtype.value_type
    if not src_value_type.equals(target_value_type):
        raise ValueError(
            f"Cannot coerce list value type from {src_value_type} to {target_value_type}. "
            "Please normalize input value dtype first."
        )

    arr = chunked.combine_chunks()
    n_rows = len(arr)
    target_dim = int(target_dtype.list_size)
    np_dtype = _arrow_scalar_to_numpy_dtype(target_value_type)
    out = np.zeros((n_rows, target_dim), dtype=np_dtype)
    valid_mask = np.asarray(arr.is_valid()) if arr.null_count > 0 else None

    padded_rows = 0
    truncated_rows = 0

    if pa.types.is_fixed_size_list(arr.type):
        src_dim = int(arr.type.list_size)
        src_vals = np.asarray(arr.values).reshape(n_rows, src_dim)
        copy_dim = min(src_dim, target_dim)
        if valid_mask is None:
            out[:, :copy_dim] = src_vals[:, :copy_dim]
            row_count = n_rows
        else:
            out[valid_mask, :copy_dim] = src_vals[valid_mask, :copy_dim]
            row_count = int(np.sum(valid_mask))
        if src_dim < target_dim:
            padded_rows = row_count
        elif src_dim > target_dim:
            truncated_rows = row_count
    else:
        offsets = np.asarray(arr.offsets)
        flat_vals = np.asarray(arr.values)
        for i in range(n_rows):
            if valid_mask is not None and not bool(valid_mask[i]):
                continue
            start = int(offsets[i])
            end = int(offsets[i + 1])
            src_len = max(0, end - start)
            if src_len < target_dim:
                padded_rows += 1
            elif src_len > target_dim:
                truncated_rows += 1
            copy_len = min(src_len, target_dim)
            if copy_len > 0:
                out[i, :copy_len] = flat_vals[start : start + copy_len]

    flat = pa.array(out.reshape(-1), type=target_value_type)
    coerced = pa.FixedSizeListArray.from_arrays(flat, target_dim)
    return coerced, padded_rows, truncated_rows


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def stats_from_np(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.shape[0] == 0:
        return {
            "min": [0],
            "max": [0],
            "mean": [0.0],
            "std": [0.0],
            "count": [0],
        }

    keepdims = arr.ndim == 1
    return {
        "min": np.min(arr, axis=0, keepdims=keepdims).tolist(),
        "max": np.max(arr, axis=0, keepdims=keepdims).tolist(),
        "mean": np.mean(arr, axis=0, keepdims=keepdims).tolist(),
        "std": np.std(arr, axis=0, keepdims=keepdims).tolist(),
        "count": [int(arr.shape[0])],
    }


def default_stats_for_field(field: pa.Field, length: int) -> dict[str, Any] | None:
    dtype = field.type

    if pa.types.is_integer(dtype):
        arr = np.zeros(length, dtype=np.int64)
        return stats_from_np(arr)

    if pa.types.is_floating(dtype):
        arr = np.zeros(length, dtype=np.float32)
        return stats_from_np(arr)

    if pa.types.is_boolean(dtype):
        arr = np.zeros(length, dtype=np.bool_)
        return stats_from_np(arr)

    if pa.types.is_fixed_size_list(dtype):
        list_size = dtype.list_size
        value_dtype = dtype.value_type
        if pa.types.is_integer(value_dtype):
            arr = np.zeros((length, list_size), dtype=np.int64)
            return stats_from_np(arr)
        if pa.types.is_floating(value_dtype):
            arr = np.zeros((length, list_size), dtype=np.float32)
            return stats_from_np(arr)

    return None


def build_episode_stats_row(
    src_stats_row: dict[str, Any] | None,
    new_ep_idx: int,
    new_indices: np.ndarray,
    new_task_indices: np.ndarray,
    missing_columns: list[str],
    target_field_map: dict[str, pa.Field],
) -> dict[str, Any]:
    if src_stats_row is None:
        stats: dict[str, Any] = {}
    else:
        stats = copy.deepcopy(src_stats_row.get("stats", {}))

    ep_count = len(new_indices)
    stats["episode_index"] = stats_from_np(np.full(ep_count, new_ep_idx, dtype=np.int64))
    stats["index"] = stats_from_np(new_indices)
    stats["task_index"] = stats_from_np(new_task_indices)

    for col in missing_columns:
        if col in {"episode_index", "index", "task_index"}:
            continue
        default_stats = default_stats_for_field(target_field_map[col], ep_count)
        if default_stats is not None:
            stats[col] = default_stats

    return {
        "episode_index": new_ep_idx,
        "stats": stats,
    }


def remap_task_indices(
    old_task_indices: np.ndarray,
    episode_tasks: list[str],
    source_tasks_by_index: dict[int, str],
    global_task_to_index: OrderedDict[str, int],
    strict: bool,
    src_root: Path,
) -> np.ndarray:
    new_task_indices = np.empty_like(old_task_indices, dtype=np.int64)
    unique_old = np.unique(old_task_indices)

    for old_val in unique_old:
        old_val_int = int(old_val)
        task_text = source_tasks_by_index.get(old_val_int)

        if task_text is None:
            if len(episode_tasks) == 1:
                task_text = episode_tasks[0]
                warn(
                    f"{src_root}: task_index={old_val_int} not found in tasks.jsonl. "
                    f"Using episode task '{task_text}'."
                )
            else:
                msg = (
                    f"{src_root}: task_index={old_val_int} not found in tasks.jsonl and "
                    f"episode tasks are not uniquely defined: {episode_tasks}"
                )
                if strict:
                    raise ValueError(msg)
                task_text = f"task_{old_val_int}"
                warn(msg + f". Falling back to '{task_text}'.")

        if task_text not in global_task_to_index:
            global_task_to_index[task_text] = len(global_task_to_index)

        new_val = global_task_to_index[task_text]
        new_task_indices[old_task_indices == old_val] = new_val

    return new_task_indices


def choose_common_numeric(
    values: list[int],
    name: str,
    strict: bool,
    override: int | None,
) -> int:
    if override is not None:
        return override

    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]

    msg = f"Input datasets have different '{name}': {unique}. Using the first value: {values[0]}"
    if strict:
        raise ValueError(msg)
    warn(msg)
    return values[0]


def choose_common_str(
    values: list[str | None],
    name: str,
    strict: bool,
) -> str | None:
    normalized = ["" if v is None else str(v) for v in values]
    unique = sorted(set(normalized))
    if len(unique) <= 1:
        return None if unique[0] == "" else unique[0]

    msg = f"Input datasets have different '{name}': {unique}. Using the first value: {values[0]}"
    if strict:
        raise ValueError(msg)
    warn(msg)
    return values[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets into one")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Input LeRobot dataset roots in merge order",
    )
    parser.add_argument("--output", required=True, help="Output merged dataset root")
    parser.add_argument(
        "--selector",
        choices=["intersection", "files", "meta"],
        default="intersection",
        help=(
            "How to choose episodes inside each source dataset: "
            "intersection=meta&parquet (recommended), files=all parquet files, meta=all episodes.jsonl"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    parser.add_argument("--strict", action="store_true", help="Fail on mismatches instead of warning")
    parser.add_argument(
        "--link-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to copy video files (parquet is always rewritten)",
    )
    parser.add_argument("--chunks-size", type=int, default=None, help="Override output chunks_size")
    parser.add_argument("--fps", type=int, default=None, help="Override output fps")
    parser.add_argument(
        "--max-episodes-per-source",
        type=int,
        default=None,
        help="For debug: limit episodes taken from each source",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print summary, do not write output")
    parser.add_argument(
        "--copy-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy/merge videos alongside parquet and metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_roots = [Path(p).expanduser().resolve() for p in args.sources]
    output_root = Path(args.output).expanduser().resolve()

    if len(source_roots) < 1:
        raise ValueError("At least one source dataset is required.")

    log("Loading source datasets...")
    sources: list[SourceDataset] = []
    for root in source_roots:
        src = load_source_dataset(
            root=root,
            selector=args.selector,
            strict=args.strict,
            max_episodes_per_source=args.max_episodes_per_source,
        )
        log(f"  - {src.root}: {len(src.selected_episode_indices)} selected episodes")
        sources.append(src)

    total_selected = sum(len(src.selected_episode_indices) for src in sources)
    if total_selected == 0:
        raise ValueError("No episodes selected from any source dataset.")

    merged_features = merge_features(sources, strict=args.strict)
    global_arrow_types, coerce_dims = build_global_arrow_types(sources)

    if not global_arrow_types:
        raise ValueError("No parquet columns found in selected episodes.")

    for col, arrow_dtype in global_arrow_types.items():
        if col not in merged_features:
            inferred = infer_feature_from_arrow(col, arrow_dtype)
            merged_features[col] = inferred
            warn(f"Feature '{col}' missing in info.json, inferred from parquet schema: {inferred}")

    for col, dim in coerce_dims.items():
        arrow_dtype = global_arrow_types[col]
        if not pa.types.is_fixed_size_list(arrow_dtype):
            continue
        target_value_dtype = infer_scalar_dtype_from_arrow(arrow_dtype.value_type)
        if col in merged_features:
            old_shape = merged_features[col].get("shape")
            merged_features[col]["dtype"] = target_value_dtype
            merged_features[col]["shape"] = [dim]
            if old_shape != [dim]:
                warn(f"Feature '{col}' shape adjusted from {old_shape} to {[dim]} to match merged data.")
        else:
            merged_features[col] = {
                "dtype": target_value_dtype,
                "shape": [dim],
                "names": None,
            }

    target_columns: list[str] = [
        key for key, ft in merged_features.items() if ft.get("dtype") != "video" and key in global_arrow_types
    ]
    for col in global_arrow_types:
        if col not in target_columns:
            target_columns.append(col)

    target_schema_no_meta, target_metadata = build_target_schema(target_columns, merged_features)
    target_field_map = {field.name: field for field in target_schema_no_meta}

    source_chunks = [int(src.info.get("chunks_size", 1000)) for src in sources]
    source_fps = [int(src.info.get("fps", 0)) for src in sources]
    source_versions = [str(src.info.get("codebase_version", "v2.1")) for src in sources]
    source_robot_types = [src.info.get("robot_type") for src in sources]

    chunks_size = choose_common_numeric(source_chunks, "chunks_size", args.strict, args.chunks_size)
    fps = choose_common_numeric(source_fps, "fps", args.strict, args.fps)
    codebase_version = choose_common_str(source_versions, "codebase_version", args.strict) or source_versions[0]
    robot_type = choose_common_str(source_robot_types, "robot_type", args.strict)

    data_path = sources[0].info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    video_path = sources[0].info.get(
        "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")
        if args.dry_run:
            log(f"[dry-run] would remove existing output: {output_root}")
        else:
            shutil.rmtree(output_root)

    if not args.dry_run:
        (output_root / "data").mkdir(parents=True, exist_ok=True)
        (output_root / "meta").mkdir(parents=True, exist_ok=True)

    global_task_to_index: OrderedDict[str, int] = OrderedDict()
    out_episodes: list[dict[str, Any]] = []
    out_episode_stats: list[dict[str, Any]] = []

    new_episode_idx = 0
    global_frame_index_cursor = 0
    total_video_files_copied = 0
    coerce_row_stats: dict[str, dict[str, int]] = {
        col: {"padded": 0, "truncated": 0} for col in coerce_dims
    }

    log("Merging episodes...")
    for src in sources:
        log(f"  Source: {src.root}")
        for old_ep_idx in src.selected_episode_indices:
            src_parquet = src.episode_files[old_ep_idx]
            table = pq.read_table(src_parquet)
            num_rows = table.num_rows

            if num_rows <= 0:
                warn(f"{src.root}: episode {old_ep_idx} has 0 rows, keeping it as-is.")

            old_cols = set(table.column_names)

            if "task_index" in table.column_names:
                old_task_indices = np.asarray(table.column("task_index").combine_chunks(), dtype=np.int64)
            else:
                old_task_indices = np.zeros(num_rows, dtype=np.int64)

            episode_meta = src.episodes_by_index.get(old_ep_idx, {})
            episode_tasks = [str(t) for t in episode_meta.get("tasks", []) if t is not None]

            if not episode_tasks:
                for old_task in np.unique(old_task_indices):
                    task_text = src.tasks_by_index.get(int(old_task))
                    if task_text is not None:
                        episode_tasks.append(task_text)

            if not episode_tasks:
                if src.tasks_by_index:
                    first_task_idx = sorted(src.tasks_by_index)[0]
                    episode_tasks = [src.tasks_by_index[first_task_idx]]
                else:
                    episode_tasks = ["task_0"]

            episode_tasks = dedupe_keep_order(episode_tasks)

            for task_text in episode_tasks:
                if task_text not in global_task_to_index:
                    global_task_to_index[task_text] = len(global_task_to_index)

            new_task_indices = remap_task_indices(
                old_task_indices=old_task_indices,
                episode_tasks=episode_tasks,
                source_tasks_by_index=src.tasks_by_index,
                global_task_to_index=global_task_to_index,
                strict=args.strict,
                src_root=src.root,
            )

            new_indices = np.arange(
                global_frame_index_cursor,
                global_frame_index_cursor + num_rows,
                dtype=np.int64,
            )
            global_frame_index_cursor += num_rows

            ep_idx_arr = pa.array(np.full(num_rows, new_episode_idx, dtype=np.int64), type=pa.int64())
            idx_arr = pa.array(new_indices, type=pa.int64())
            task_arr = pa.array(new_task_indices, type=pa.int64())

            table = set_or_add_column(table, "episode_index", ep_idx_arr)
            table = set_or_add_column(table, "index", idx_arr)
            table = set_or_add_column(table, "task_index", task_arr)

            missing_columns: list[str] = []
            for col in target_columns:
                if col in table.column_names:
                    continue
                fill_arr = default_array_for_field(target_field_map[col], num_rows)
                table = table.append_column(col, fill_arr)
                missing_columns.append(col)

            for col in COERCE_FIXED_SIZE_COLUMNS:
                if col not in table.column_names or col not in target_field_map:
                    continue
                target_type = target_field_map[col].type
                if not pa.types.is_fixed_size_list(target_type):
                    continue
                coerced_col, padded_rows, truncated_rows = coerce_chunked_list_to_fixed_size(
                    table.column(col), target_type
                )
                table = set_or_add_column(table, col, coerced_col)
                if col in coerce_row_stats:
                    coerce_row_stats[col]["padded"] += padded_rows
                    coerce_row_stats[col]["truncated"] += truncated_rows

            table = table.select(target_columns)
            table = table.cast(target_schema_no_meta, safe=False)
            table = table.replace_schema_metadata(target_metadata)

            out_chunk_idx = new_episode_idx // chunks_size
            out_parquet = (
                output_root
                / "data"
                / f"chunk-{out_chunk_idx:03d}"
                / f"episode_{new_episode_idx:06d}.parquet"
            )

            if not args.dry_run:
                ensure_parent(out_parquet)
                pq.write_table(table, out_parquet)

            if args.copy_videos:
                for video_key, src_video in src.video_files.get(old_ep_idx, []):
                    out_video = (
                        output_root
                        / "videos"
                        / f"chunk-{out_chunk_idx:03d}"
                        / video_key
                        / f"episode_{new_episode_idx:06d}.mp4"
                    )
                    if not args.dry_run:
                        link_or_copy_file(src_video, out_video, args.link_mode)
                    total_video_files_copied += 1

            if num_rows > 0:
                task_index_to_text = {idx: task for task, idx in global_task_to_index.items()}
                episode_tasks_for_meta = [
                    task_index_to_text[int(idx)] for idx in sorted(np.unique(new_task_indices).tolist())
                ]
            else:
                episode_tasks_for_meta = episode_tasks

            out_episodes.append(
                {
                    "episode_index": new_episode_idx,
                    "tasks": episode_tasks_for_meta,
                    "length": int(num_rows),
                }
            )

            src_stats_row = src.stats_by_index.get(old_ep_idx)
            out_episode_stats.append(
                build_episode_stats_row(
                    src_stats_row=src_stats_row,
                    new_ep_idx=new_episode_idx,
                    new_indices=new_indices,
                    new_task_indices=new_task_indices,
                    missing_columns=missing_columns,
                    target_field_map=target_field_map,
                )
            )

            new_episode_idx += 1
            if new_episode_idx % 20 == 0:
                log(f"    merged episodes: {new_episode_idx}")

    out_tasks = [
        {"task_index": idx, "task": task}
        for task, idx in sorted(global_task_to_index.items(), key=lambda x: x[1])
    ]

    total_episodes = len(out_episodes)
    total_frames = int(sum(ep["length"] for ep in out_episodes))
    total_chunks = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0

    out_info = {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(out_tasks),
        "total_videos": int(total_video_files_copied),
        "total_chunks": total_chunks,
        "chunks_size": int(chunks_size),
        "fps": int(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": data_path,
        "video_path": video_path,
        "features": merged_features,
    }

    if args.dry_run:
        log("\nDry-run summary:")
        log(f"  output: {output_root}")
        log(f"  total episodes: {total_episodes}")
        log(f"  total frames: {total_frames}")
        log(f"  total tasks: {len(out_tasks)}")
        log(f"  total videos: {total_video_files_copied}")
        return

    write_json(output_root / "meta" / "info.json", out_info)
    write_jsonl(output_root / "meta" / "tasks.jsonl", out_tasks)
    write_jsonl(output_root / "meta" / "episodes.jsonl", out_episodes)
    write_jsonl(output_root / "meta" / "episodes_stats.jsonl", out_episode_stats)

    for col, stats in coerce_row_stats.items():
        if stats["padded"] > 0 or stats["truncated"] > 0:
            warn(
                f"Column '{col}' auto-coercion summary: "
                f"padded_rows={stats['padded']}, truncated_rows={stats['truncated']}."
            )

    log("\nDone.")
    log(f"Output dataset: {output_root}")
    log(f"Episodes: {total_episodes}, frames: {total_frames}, tasks: {len(out_tasks)}")


if __name__ == "__main__":
    main()
