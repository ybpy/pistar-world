#!/usr/bin/env python3
"""Task6 SRPO-style pipeline: label data, train VLM, infer advantages, train policy.

The script is intentionally path-portable. All site-specific model/data paths should be
provided through environment variables or CLI flags.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def run(cmd: list[str], *, cwd: Path, log_path: Path, gpus: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpus,
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": env.get("WANDB_MODE", "offline"),
            "PYTHONPATH": f"{cwd / 'src'}:{cwd / 'scripts'}:{env.get('PYTHONPATH', '')}",
            "XLA_PYTHON_CLIENT_PREALLOCATE": env.get("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
            "JAX_PLATFORMS": env.get("JAX_PLATFORMS", "cuda,cpu"),
        }
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] CUDA_VISIBLE_DEVICES={gpus} {' '.join(cmd)}\n")
        f.flush()
        subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def rm_tree(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        shutil.rmtree(path)


def find_vlm_checkpoint(ckpt_dir: Path) -> str:
    for name in ("step_00005000", "step_00004999"):
        if (ckpt_dir / name / "_CHECKPOINT_METADATA").exists():
            return name
    raise FileNotFoundError(f"No 5k/4999 VLM checkpoint found under {ckpt_dir}")


def main() -> None:
    root_default = Path(__file__).resolve().parents[2]
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--root", default=str(root_default))
    pre_parser.add_argument("--env_file", default=None)
    pre_args, _ = pre_parser.parse_known_args()
    root = Path(pre_args.root).resolve()
    load_env_file(Path(pre_args.env_file).expanduser() if pre_args.env_file else root / ".env")

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python"))
    parser.add_argument("--source_data", default=os.environ.get("TASK6_SOURCE_DATA", "outputs/lerobot_policy_data/task6_demo_plus_policy_rollout50"))
    parser.add_argument("--run_name", default="task6_srpo_s1_f01_a12")
    parser.add_argument("--alpha", type=float, default=12.0)
    parser.add_argument("--failure_scale", type=float, default=0.1)
    parser.add_argument("--top_percent", type=float, default=30.0)
    parser.add_argument("--policy_steps", type=int, default=5000)
    parser.add_argument("--vlm_steps", type=int, default=5000)
    parser.add_argument("--policy_batch_size", type=int, default=64)
    parser.add_argument("--vlm_batch_size", type=int, default=32)
    parser.add_argument("--fsdp_devices", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--label_gpu", default="0")
    parser.add_argument("--infer_gpu", default="0")
    parser.add_argument("--vjepe_model_path", dest="vjepa_model_path", default=os.environ.get("VJEPA_MODEL_PATH"))
    parser.add_argument("--tokenizer_path", default=os.environ.get("TOKENIZER_PATH"))
    parser.add_argument("--pi05_base_params", default=os.environ.get("PI05_BASE_PARAMS", "gs://openpi-assets/checkpoints/pi05_base/params"))
    parser.add_argument("--assets_id", default="ybpy/libero_pistar")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    py = args.python
    if not args.vjepa_model_path:
        raise ValueError("Set --vjepa_model_path or VJEPA_MODEL_PATH")
    if not args.tokenizer_path:
        raise ValueError("Set --tokenizer_path or TOKENIZER_PATH")

    source_data = resolve(root, args.source_data)
    labeled = root / "outputs" / "lerobot_policy_data" / f"{args.run_name}_labeled"
    vlm_ckpt = root / "checkpoints" / f"value_{args.run_name}_vlm5k"
    infer_data = root / "outputs" / "lerobot_policy_data" / f"{args.run_name}_vlm_adv_top{int(args.top_percent)}"
    policy_exp = f"policy_{args.run_name}_vlm5k_top{int(args.top_percent)}_{args.policy_steps}_b{args.policy_batch_size}_{args.fsdp_devices}gpu"
    policy_root = root / "checkpoints" / "pi05_star_libero_task6_srpo_baseinit" / policy_exp
    log_dir = root / "outputs" / "training_logs" / args.run_name

    for path in (labeled, vlm_ckpt, infer_data, policy_root):
        rm_tree(path, args.overwrite)

    run(
        [
            py,
            "-u",
            "scripts/label_lerobot_rollout_progress.py",
            "--input_dir",
            str(source_data),
            "--output_dir",
            str(labeled),
            "--overwrite",
            "--prefix_stride",
            "1",
            "--model_path",
            args.vjepa_model_path,
            "--cache_dir",
            str(root / "cache" / args.run_name),
            "--center_source",
            "demo_success",
            "--distance_normalization",
            "percentile_minmax",
            "--distance_q_low",
            "0.01",
            "--distance_q_high",
            "0.99",
            "--progress_activation",
            "srpo_sigmoid",
            "--srpo_sigmoid_norm_source",
            "all_prefix_percentile_minmax",
            "--srpo_sigmoid_steepness",
            str(args.alpha),
            "--srpo_sigmoid_offset",
            "0.5",
            "--srpo_success_scale",
            "1.0",
            "--srpo_failure_scale",
            str(args.failure_scale),
            "--device_id",
            "0",
            "--prefix_batch_size",
            "8",
        ],
        cwd=root,
        gpus=args.label_gpu,
        log_path=log_dir / "01_label.log",
    )

    run(
        [
            py,
            "-u",
            "scripts/train_value.py",
            "--data_dir",
            str(labeled),
            "--checkpoint_dir",
            str(vlm_ckpt),
            "--batch_size",
            str(args.vlm_batch_size),
            "--num_train_steps",
            str(args.vlm_steps),
            "--load_pretrained",
            "--wandb_mode",
            "offline",
            "--wandb_project",
            "pistar",
            "--wandb_run_name",
            f"value_{args.run_name}_vlm5k",
            "--log_interval",
            "100",
            "--save_interval",
            str(args.vlm_steps),
            "--val_interval",
            "0",
            "--peak_lr",
            "2.5e-5",
            "--decay_lr",
            "2.5e-6",
            "--warmup_steps",
            "1000",
            "--freeze_mode",
            "all_backbones",
            "--num_workers",
            "0",
            "--fsdp_devices",
            str(args.fsdp_devices),
            "--tokenizer_path",
            args.tokenizer_path,
        ],
        cwd=root,
        gpus=args.gpus,
        log_path=log_dir / "02_train_vlm.log",
    )

    ckpt_name = find_vlm_checkpoint(vlm_ckpt)
    shutil.copytree(labeled, infer_data)
    run(
        [
            py,
            "-u",
            "scripts/label_advantage_from_vlm.py",
            "--data_dir",
            str(infer_data),
            "--checkpoint_dir",
            str(vlm_ckpt),
            "--checkpoint_name",
            ckpt_name,
            "--lookahead",
            "15",
            "--top_percent",
            str(args.top_percent),
            "--batch_size",
            "8",
            "--num_workers",
            "0",
            "--reward_col",
            "reward_label",
            "--tokenizer_path",
            args.tokenizer_path,
        ],
        cwd=root,
        gpus=args.infer_gpu,
        log_path=log_dir / "03_vlm_infer.log",
    )

    run(
        [
            py,
            "-u",
            "scripts/train.py",
            "pi05_star_libero_task6_srpo_baseinit",
            "--exp_name",
            policy_exp,
            "--data.repo_id",
            str(infer_data),
            "--data.assets.asset_id",
            args.assets_id,
            "--weight_loader.params_path",
            args.pi05_base_params,
            "--batch_size",
            str(args.policy_batch_size),
            "--num_train_steps",
            str(args.policy_steps),
            "--save_interval",
            str(args.policy_steps),
            "--keep_period",
            str(args.policy_steps),
            "--log_interval",
            "10",
            "--num_workers",
            "0",
            "--fsdp_devices",
            str(args.fsdp_devices),
            "--overwrite",
        ],
        cwd=root,
        gpus=args.gpus,
        log_path=log_dir / "04_train_policy.log",
    )

    print(f"done: policy checkpoint root = {policy_root}")


if __name__ == "__main__":
    main()
