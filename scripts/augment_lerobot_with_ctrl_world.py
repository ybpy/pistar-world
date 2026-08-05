#!/usr/bin/env python3
"""Augment LeRobot rollouts by truncating trajectories and extending them with Ctrl-World.

This is an offline utility for augmenting LIBERO rollout data. For task6, the
default launcher samples 10 cut points from each of 50 rollout episodes to
produce 500 augmented episodes. For each augmented episode it:

1. Samples a source LIBERO rollout episode of length T.
2. Samples a cut index from [min_prefix_steps - 1, T - target_suffix_steps].
3. Copies the real prefix [0, cut] into the new episode.
4. Seeds Ctrl-World from the cut frame and generates target_suffix_steps future
   frames/actions with the policy + world-model interaction loop.
5. Saves prefix + synthetic suffix as a new LeRobot episode.

By default target_suffix_steps = ctrl_interactions * steps_per_ctrl_interaction
= 5 * 12 = 60. The current Ctrl-World rollout
implementation emits pred_step - 1 saved sparse transitions per WM chunk, so this
script may run more internal WM chunks than ctrl_interactions to reach the requested
steps. That preserves the existing Ctrl-World timing semantics.
"""

from __future__ import annotations

import io
import json
import logging
import math
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

# Ctrl-World/OpenPI imports are intentionally lazy. Importing them at module load
# time can initialize TensorFlow/torch stacks and make --help hang for a long time.
torch = None
wm_args = None
Agent = None
RolloutExportArgs = None
LiberoRolloutLeRobotWriter = None
build_history_indices_from_buffer = None
merge_args = None
preprocess_for_writer = None
preprocess_for_wm = None


def _load_ctrl_world_runtime() -> None:
    global torch
    global wm_args
    global Agent
    global RolloutExportArgs
    global LiberoRolloutLeRobotWriter
    global build_history_indices_from_buffer
    global merge_args
    global preprocess_for_writer
    global preprocess_for_wm

    import torch as _torch
    from openpi.training.config_wm import wm_args as _wm_args
    from rollout_wm_libero import (
        Agent as _Agent,
        RolloutExportArgs as _RolloutExportArgs,
        LiberoRolloutLeRobotWriter as _LiberoRolloutLeRobotWriter,
        build_history_indices_from_buffer as _build_history_indices_from_buffer,
        merge_args as _merge_args,
        preprocess_for_writer as _preprocess_for_writer,
        preprocess_for_wm as _preprocess_for_wm,
    )

    torch = _torch
    wm_args = _wm_args
    Agent = _Agent
    RolloutExportArgs = _RolloutExportArgs
    LiberoRolloutLeRobotWriter = _LiberoRolloutLeRobotWriter
    build_history_indices_from_buffer = _build_history_indices_from_buffer
    merge_args = _merge_args
    preprocess_for_writer = _preprocess_for_writer
    preprocess_for_wm = _preprocess_for_wm


@dataclass(frozen=True)
class EpisodeRef:
    path: Path
    episode_index: int
    task_index: int
    task: str
    length: int


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {value}")


def _scalar(value: Any) -> Any:
    if isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        for dtype in (np.float32, np.int64, np.int32, np.bool_):
            parsed = np.frombuffer(raw, dtype=dtype)
            if parsed.size == 1:
                return parsed[0].item()
        return raw
    if isinstance(value, (np.ndarray, list, tuple)):
        array = np.asarray(value).reshape(-1)
        return array[0].item() if array.size else None
    return value.item() if hasattr(value, "item") else value


def _array(value: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(value), dtype=dtype).copy()
    return np.asarray(value, dtype=dtype).reshape(-1)


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
    path = data_dir / "meta" / "tasks.jsonl"
    tasks: dict[int, str] = {}
    if not path.exists():
        return tasks
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            tasks[int(record["task_index"])] = str(record["task"])
    return tasks


def _load_episode_refs(data_dir: Path) -> list[EpisodeRef]:
    tasks = _load_tasks(data_dir)
    refs: list[EpisodeRef] = []
    for path in sorted((data_dir / "data").rglob("*.parquet")):
        df = pd.read_parquet(path, columns=["episode_index", "task_index"])
        episode_index = int(_scalar(df["episode_index"].iloc[0]))
        task_index = int(_scalar(df["task_index"].iloc[0]))
        refs.append(
            EpisodeRef(
                path=path,
                episode_index=episode_index,
                task_index=task_index,
                task=tasks.get(task_index, f"task_{task_index}"),
                length=len(df),
            )
        )
    return refs


def _state8_from_row(row: pd.Series) -> np.ndarray:
    state = _array(row["state"], dtype=np.float32)
    if state.size < 8:
        state = np.pad(state, (0, 8 - state.size))
    return state[:8].astype(np.float32)


def _pose7_from_state8(state8: np.ndarray) -> np.ndarray:
    pose = np.zeros((7,), dtype=np.float32)
    pose[:6] = np.asarray(state8[:6], dtype=np.float32)
    pose[6] = float(state8[6])
    return pose


def _action7_from_row(row: pd.Series) -> np.ndarray:
    if "actions" not in row:
        return np.zeros((7,), dtype=np.float32)
    action = _array(row["actions"], dtype=np.float32)
    if action.size < 7:
        action = np.pad(action, (0, 7 - action.size))
    return action[:7].astype(np.float32)


def _row_to_writer_step(row: pd.Series, dataset_root: Path) -> dict[str, np.ndarray]:
    image = _decode_image(row["image"], dataset_root)
    wrist = _decode_image(row["wrist_image"], dataset_root)
    return {
        "image": preprocess_for_writer(image, image_size=256),
        "wrist_image": preprocess_for_writer(wrist, image_size=256),
        "state": _state8_from_row(row),
        "actions": _action7_from_row(row),
    }


def _encode_history_from_prefix(agent: Agent, df: pd.DataFrame, dataset_root: Path, cut_idx: int, args) -> tuple[list, list, list, list[np.ndarray]]:
    max_cache = int(getattr(args, "wm_history_cache_len", 56))
    start_idx = max(0, cut_idx - max_cache + 1)
    history_rows = df.iloc[start_idx : cut_idx + 1]
    if history_rows.empty:
        raise ValueError("Cannot seed Ctrl-World from an empty prefix")

    his_cond = []
    his_state = []
    his_eef = []
    current_obs = None
    for _, row in history_rows.iterrows():
        base = _decode_image(row["image"], dataset_root)
        wrist = _decode_image(row["wrist_image"], dataset_root)
        base_wm = preprocess_for_wm(base, height=args.height, width=args.width)
        wrist_wm = preprocess_for_wm(wrist, height=args.height, width=args.width)
        zero_wm = np.zeros_like(base_wm)
        _, latent = agent.encode_views([base_wm, wrist_wm, zero_wm])
        state8 = _state8_from_row(row)
        pose7 = _pose7_from_state8(state8)
        his_cond.append(latent)
        his_state.append(state8[None, :])
        his_eef.append(pose7[None, :])
        current_obs = [base_wm, wrist_wm, zero_wm]

    warm_repeats = int(getattr(args, "history_init_repeats", 4))
    min_len = max(int(args.num_history) * warm_repeats, int(args.num_history))
    while len(his_cond) < min_len:
        his_cond.insert(0, his_cond[0])
        his_state.insert(0, his_state[0])
        his_eef.insert(0, his_eef[0])

    his_cond = his_cond[-max_cache:]
    his_state = his_state[-max_cache:]
    his_eef = his_eef[-max_cache:]
    return his_cond, his_state, his_eef, current_obs


def _generate_suffix(agent: Agent, task: str, his_cond, his_state, his_eef, current_obs, target_steps: int, args) -> list[dict]:
    rollout_steps: list[dict] = []
    pred_step = int(args.pred_step)
    steps_per_chunk = max(0, pred_step - 1)
    if steps_per_chunk <= 0:
        raise ValueError(f"pred_step must be > 1, got {pred_step}")
    max_chunks = int(math.ceil(target_steps / steps_per_chunk))

    video_dict_pred = [np.expand_dims(v, axis=0) for v in current_obs]
    for chunk_idx in range(max_chunks):
        if len(rollout_steps) >= target_steps:
            break
        current_state = his_state[-1][0]
        current_pose = his_eef[-1][0]
        current_videos = [v[-1] for v in video_dict_pred]

        policy_in_out, action_sparse, pose_ds, state_ds = agent.forward_policy(
            videos=current_videos,
            state8=current_state,
            pose7=current_pose,
            text=task,
        )
        del policy_in_out

        history_indices = build_history_indices_from_buffer(args, len(his_eef))
        history_pose = np.concatenate([his_eef[idx] for idx in history_indices], axis=0)
        action_cond = np.concatenate([history_pose, pose_ds], axis=0)
        his_latent = torch.cat([his_cond[idx] for idx in history_indices], dim=0).unsqueeze(0)
        current_latent = his_cond[-1]

        _, video_dict_pred, predict_latents = agent.forward_wm(
            action_cond=action_cond,
            current_latent=current_latent,
            his_cond=his_latent,
            text=task if args.text_cond else None,
        )

        for step_j in range(steps_per_chunk):
            if len(rollout_steps) >= target_steps:
                break
            base_raw = np.ascontiguousarray(video_dict_pred[args.policy_base_camera_idx][step_j])
            wrist_raw = np.ascontiguousarray(video_dict_pred[args.policy_wrist_camera_idx][step_j])
            base_out = preprocess_for_writer(base_raw, image_size=256)
            wrist_out = preprocess_for_writer(wrist_raw, image_size=256)
            state_to_save = np.asarray(state_ds[step_j], dtype=np.float32)
            if state_to_save.size < 8:
                state_to_save = np.pad(state_to_save.reshape(-1), (0, 8 - state_to_save.size))
            action_to_save = np.asarray(action_sparse[step_j], dtype=np.float32)
            if action_to_save.size < 7:
                action_to_save = np.pad(action_to_save.reshape(-1), (0, 7 - action_to_save.size))
            rollout_steps.append(
                {
                    "image": np.ascontiguousarray(base_out),
                    "wrist_image": np.ascontiguousarray(wrist_out),
                    "state": state_to_save[:8].astype(np.float32),
                    "actions": action_to_save[:7].astype(np.float32),
                }
            )

        for step_j in range(1, pred_step):
            his_state.append(np.asarray(state_ds[step_j], dtype=np.float32)[None, :])
            his_eef.append(np.asarray(pose_ds[step_j], dtype=np.float32)[None, :])
            his_cond.append(torch.cat([v[step_j] for v in predict_latents], dim=1).unsqueeze(0))

        max_cache = int(getattr(args, "wm_history_cache_len", 56))
        if len(his_state) > max_cache:
            his_state = his_state[-max_cache:]
            his_eef = his_eef[-max_cache:]
            his_cond = his_cond[-max_cache:]

        logging.info("Generated %d/%d suffix steps for current augmented episode", len(rollout_steps), target_steps)

    return rollout_steps[:target_steps]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="outputs/lerobot_policy_data/task6_demo_plus_policy_rollout50",
        help="Source LIBERO LeRobot rollout dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/lerobot_policy_data/task6_ctrlworld_aug500",
        help="Output LeRobot augmented dataset directory",
    )
    parser.add_argument("--num_aug", type=int, default=500)
    parser.add_argument("--sample_mode", choices=["random", "per_episode"], default="per_episode")
    parser.add_argument("--points_per_episode", type=int, default=10)
    parser.add_argument("--episode_shard_index", type=int, default=0)
    parser.add_argument("--episode_shard_count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ctrl_interactions", type=int, default=5)
    parser.add_argument("--steps_per_ctrl_interaction", type=int, default=12)
    parser.add_argument("--target_suffix_steps", type=int, default=None)
    parser.add_argument("--min_prefix_steps", type=int, default=1)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument(
        "--synthetic_success",
        choices=["false", "true", "source"],
        default="false",
        help="Success label for augmented episodes. Default false avoids treating imagined futures as successful demos.",
    )

    # Ctrl-World / policy overrides. Defaults come from config_wm.py.
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--adv_ind_input", type=str, default=None)
    parser.add_argument("--svd_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--pi_ckpt", type=str, default=None)
    parser.add_argument("--data_stat_path", type=str, default=None)
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--pred_step", type=int, default=None)
    parser.add_argument("--policy_downsample_stride", type=int, default=None)
    parser.add_argument("--policy_base_camera_idx", type=int, default=None)
    parser.add_argument("--policy_wrist_camera_idx", type=int, default=None)
    parser.add_argument("--use_dynamics", type=str2bool, default=None)
    parser.add_argument("--dyn_ckpt_path", type=str, default=None)
    parser.add_argument("--dyn_stat_path", type=str, default=None)
    parser.add_argument("--dyn_action_num", type=int, default=None)
    parser.add_argument("--dyn_hidden_size", type=int, default=None)
    parser.add_argument("--dyn_num_layers", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--decode_chunk_size", type=int, default=None)
    parser.add_argument("--text_cond", type=str2bool, default=None)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    cli = build_parser().parse_args()
    _load_ctrl_world_runtime()

    input_dir = Path(cli.input_dir).resolve()
    output_dir = Path(cli.output_dir).resolve()
    if not (input_dir / "data").exists() or not (input_dir / "meta").exists():
        raise FileNotFoundError(f"Input is not a LeRobot dataset: {input_dir}")
    if output_dir.exists():
        if not cli.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite true to replace it")
        shutil.rmtree(output_dir)

    target_suffix_steps = cli.target_suffix_steps
    if target_suffix_steps is None:
        target_suffix_steps = int(cli.ctrl_interactions) * int(cli.steps_per_ctrl_interaction)
    if target_suffix_steps <= 0:
        raise ValueError("target suffix steps must be positive")

    refs = _load_episode_refs(input_dir)
    eligible_all = [ref for ref in refs if ref.length >= int(cli.min_prefix_steps) + target_suffix_steps]
    if not eligible_all:
        longest = max((ref.length for ref in refs), default=0)
        raise ValueError(
            f"No source episodes are long enough for min_prefix_steps={cli.min_prefix_steps} "
            f"and target_suffix_steps={target_suffix_steps}; longest={longest}"
        )
    if int(cli.episode_shard_count) <= 0:
        raise ValueError("episode_shard_count must be positive")
    if not (0 <= int(cli.episode_shard_index) < int(cli.episode_shard_count)):
        raise ValueError("episode_shard_index must be in [0, episode_shard_count)")
    eligible = [
        ref for idx, ref in enumerate(eligible_all)
        if idx % int(cli.episode_shard_count) == int(cli.episode_shard_index)
    ]
    if not eligible:
        raise ValueError(
            f"No eligible episodes assigned to shard {cli.episode_shard_index}/{cli.episode_shard_count}; "
            f"eligible_all={len(eligible_all)}"
        )

    args = wm_args()
    args = merge_args(args, cli)
    args.__post_init__()
    args.seed = int(cli.seed)

    export_args = RolloutExportArgs(
        save_lerobot_rollout=True,
        rollout_repo_id=output_dir.name,
        rollout_output_dir=str(output_dir.parent),
        rollout_overwrite=True,
        rollout_robot_type="panda",
        rollout_fps=10,
        rollout_penalty_value=-1.0,
    )
    writer = LiberoRolloutLeRobotWriter(export_args, image_size=256)
    agent = Agent(args)

    rng = np.random.default_rng(cli.seed)
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "sample_mode": cli.sample_mode,
        "num_aug_requested": int(cli.num_aug),
        "points_per_episode": int(cli.points_per_episode),
        "episode_shard_index": int(cli.episode_shard_index),
        "episode_shard_count": int(cli.episode_shard_count),
        "eligible_episodes_total": len(eligible_all),
        "eligible_episodes_in_shard": len(eligible),
        "target_suffix_steps": int(target_suffix_steps),
        "ctrl_interactions": int(cli.ctrl_interactions),
        "steps_per_ctrl_interaction": int(cli.steps_per_ctrl_interaction),
        "synthetic_success": cli.synthetic_success,
        "episodes": [],
    }

    plan: list[tuple[EpisodeRef, int]] = []
    min_cut = int(cli.min_prefix_steps) - 1
    if cli.sample_mode == "per_episode":
        points_per_episode = int(cli.points_per_episode)
        if points_per_episode <= 0:
            raise ValueError("points_per_episode must be positive")
        for ref in eligible:
            max_cut = ref.length - target_suffix_steps
            legal_cuts = np.arange(min_cut, max_cut + 1, dtype=np.int64)
            replace = legal_cuts.size < points_per_episode
            sampled = rng.choice(legal_cuts, size=points_per_episode, replace=replace)
            for cut in sampled:
                plan.append((ref, int(cut)))
    else:
        for _ in range(int(cli.num_aug)):
            ref = eligible[int(rng.integers(0, len(eligible)))]
            max_cut = ref.length - target_suffix_steps
            cut_idx = int(rng.integers(min_cut, max_cut + 1))
            plan.append((ref, cut_idx))

    manifest["num_aug_planned"] = len(plan)

    for aug_idx, (ref, cut_idx) in enumerate(plan):
        df = pd.read_parquet(ref.path)

        prefix_steps = [_row_to_writer_step(row, input_dir) for _, row in df.iloc[: cut_idx + 1].iterrows()]
        his_cond, his_state, his_eef, current_obs = _encode_history_from_prefix(agent, df, input_dir, cut_idx, args)
        suffix_steps = _generate_suffix(
            agent=agent,
            task=ref.task,
            his_cond=his_cond,
            his_state=his_state,
            his_eef=his_eef,
            current_obs=current_obs,
            target_steps=target_suffix_steps,
            args=args,
        )

        if cli.synthetic_success == "source":
            source_success = False
            if "success" in df.columns:
                source_success = bool(_scalar(df["success"].iloc[-1]))
            success = source_success
        else:
            success = cli.synthetic_success == "true"

        writer.save_episode(steps=prefix_steps + suffix_steps, task=ref.task, success=success)
        record = {
            "aug_index": aug_idx,
            "source_path": str(ref.path.relative_to(input_dir)),
            "source_episode_index": int(ref.episode_index),
            "task_index": int(ref.task_index),
            "task": ref.task,
            "source_length": int(ref.length),
            "cut_idx": int(cut_idx),
            "prefix_steps": int(len(prefix_steps)),
            "synthetic_suffix_steps": int(len(suffix_steps)),
            "new_length": int(len(prefix_steps) + len(suffix_steps)),
            "success": bool(success),
        }
        manifest["episodes"].append(record)
        logging.info("Saved augmented episode %d/%d: %s", aug_idx + 1, len(plan), record)

    manifest_path = output_dir / "ctrl_world_augmentation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Wrote augmentation manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
