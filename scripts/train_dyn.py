import argparse
import json
import logging
import pathlib
from bisect import bisect_right

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import wandb
except Exception:
    wandb = None

from openpi.models.dynamics import Dynamics


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {value}")


def _as_pose_array(values, expected_dim: int = 7) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == object:
        arr = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in values], axis=0)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, expected_dim)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"Expected [N,{expected_dim}] array, got {arr.shape}")
    return arr


class LiberoDynamicsDataset(Dataset):
    """Multi-step dynamics dataset from LeRobot parquet episodes.

    Each sample:
            current_pose6   = abs_pose[t][:6]
            action_seq6     = actions[t:t+H][:, :6]
            next_pose_seq6  = abs_pose[t+1:t+1+H][:, :6]
    """

    def __init__(self, dataset_root: str, horizon: int):
        self.dataset_root = pathlib.Path(dataset_root)
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        self.episode_paths = sorted((self.dataset_root / "data").glob("chunk-*/episode_*.parquet"))
        if not self.episode_paths:
            raise ValueError(f"No parquet episodes found under: {self.dataset_root / 'data'}")

        self.sample_counts = []
        self.cum_counts = []
        running = 0
        for path in self.episode_paths:
            df = pd.read_parquet(path, columns=["abs_pose"])
            n = int(len(df))
            sample_n = max(0, n - self.horizon)
            self.sample_counts.append(sample_n)
            running += sample_n
            self.cum_counts.append(running)

        self.total_samples = running
        if self.total_samples <= 0:
            raise ValueError("No valid one-step samples (all episodes length <= 1)")

        self._cache_idx = -1
        self._cache_pose = None
        self._cache_action = None

    def __len__(self):
        return self.total_samples

    def _load_episode(self, epi_idx: int):
        if self._cache_idx == epi_idx:
            return self._cache_pose, self._cache_action

        path = self.episode_paths[epi_idx]
        df = pd.read_parquet(path, columns=["abs_pose", "actions"])
        pose = _as_pose_array(df["abs_pose"].to_list(), expected_dim=7)
        action = _as_pose_array(df["actions"].to_list(), expected_dim=7)

        if len(action) != len(pose):
            n = min(len(action), len(pose))
            pose = pose[:n]
            action = action[:n]

        self._cache_idx = epi_idx
        self._cache_pose = pose
        self._cache_action = action
        return pose, action

    def __getitem__(self, index: int):
        if index < 0 or index >= self.total_samples:
            raise IndexError(index)

        epi_idx = bisect_right(self.cum_counts, index)
        prev_cum = 0 if epi_idx == 0 else self.cum_counts[epi_idx - 1]
        local_idx = index - prev_cum

        pose, action = self._load_episode(epi_idx)

        current_pose = pose[local_idx, :6]
        act_seq = action[local_idx : local_idx + self.horizon, :6]
        next_pose_seq = pose[local_idx + 1 : local_idx + 1 + self.horizon, :6]

        return {
            "current_pose6": torch.from_numpy(current_pose),
            "action_seq6": torch.from_numpy(act_seq),
            "next_pose_seq6": torch.from_numpy(next_pose_seq),
        }


def _extract_first6(stats: dict, key: str) -> np.ndarray:
    if key not in stats:
        raise KeyError(key)
    arr = np.asarray(stats[key], dtype=np.float32).reshape(-1)
    if arr.shape[0] < 6:
        raise ValueError(f"Stats key {key} must have at least 6 dims, got {arr.shape[0]}")
    return arr[:6]


def load_dynamics_stats(stat_path: str) -> dict:
    """Load dynamics normalization stats from json (canonical keys only)."""
    with open(stat_path, "r") as f:
        stats = json.load(f)

    required = ["dyn_action_01", "dyn_action_99", "dyn_pose_01", "dyn_pose_99"]
    missing = [k for k in required if k not in stats]
    if missing:
        raise ValueError(f"Dynamics stat file missing canonical keys {missing}: {stat_path}")

    return {
        "dyn_action_01": _extract_first6(stats, "dyn_action_01"),
        "dyn_action_99": _extract_first6(stats, "dyn_action_99"),
        "dyn_pose_01": _extract_first6(stats, "dyn_pose_01"),
        "dyn_pose_99": _extract_first6(stats, "dyn_pose_99"),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/public/home/chenyuyao1/.cache/huggingface/lerobot/ybpy/libero_pistar_wm",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument(
        "--dyn_stat_path",
        type=str,
        default="/public/home/chenyuyao1/code/pistar/dataset_meta_info/libero_wm/stat.json",
        help="JSON stats for dynamics normalization (Ctrl-World style min/max).",
    )
    parser.add_argument("--save_dir", type=str, default="checkpoints/dynamics")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)

    # Optional wandb logging.
    parser.add_argument("--wandb_enabled", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="pistar_dyn")
    parser.add_argument("--wandb_run_name", type=str, default="pistar_dyn_train")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_log_steps", type=int, default=100)
    parser.add_argument("--wandb_resume", type=str2bool, default=True)

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.dyn_stat_path:
        raise ValueError("--dyn_stat_path is required for dynamics normalization")
    dyn_stats = load_dynamics_stats(args.dyn_stat_path)

    dataset = LiberoDynamicsDataset(args.dataset_root, horizon=args.horizon)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dynamics(
        pose_dim=6,
        action_dim=6,
        action_num=args.horizon,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    model.set_normalization_stats(
        action_min=dyn_stats["dyn_action_01"],
        action_max=dyn_stats["dyn_action_99"],
        pose_min=dyn_stats["dyn_pose_01"],
        pose_max=dyn_stats["dyn_pose_99"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb_log_steps <= 0:
        raise ValueError("wandb_log_steps must be positive")

    wandb_run = None
    if args.wandb_enabled:
        if wandb is None:
            raise ImportError("wandb_enabled=True but wandb is not installed. Please `pip install wandb`.")

        wandb_id_path = save_dir / "wandb_id.txt"
        init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": vars(args),
            "dir": str(save_dir),
        }
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity

        if args.wandb_resume and wandb_id_path.exists():
            run_id = wandb_id_path.read_text().strip()
            if run_id:
                init_kwargs["id"] = run_id
                init_kwargs["resume"] = "must"

        wandb_run = wandb.init(**init_kwargs)
        if wandb_run is not None and getattr(wandb_run, "id", None):
            wandb_id_path.write_text(str(wandb_run.id))

    config_path = save_dir / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    logging.info("Dataset root: %s", args.dataset_root)
    logging.info("Episodes: %d, multi-step samples: %d", len(dataset.episode_paths), len(dataset))
    logging.info("Device: %s", device)
    logging.info("Dynamics horizon: %d", args.horizon)
    logging.info("Dynamics stat path: %s", args.dyn_stat_path)

    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            n_items = 0

            progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
            for batch in progress:
                current_pose6 = batch["current_pose6"].to(device=device, dtype=torch.float32)
                action_seq6 = batch["action_seq6"].to(device=device, dtype=torch.float32)
                next_pose_seq6 = batch["next_pose_seq6"].to(device=device, dtype=torch.float32)

                pred_next_norm = model(
                    current_pose6,
                    action_seq6,
                    normalize_pose_input=True,
                    normalize_input=True,
                    denormalize_output=False,
                    strict_horizon=True,
                )
                target_next_norm = model.normalize_pose(next_pose_seq6)
                loss = F.mse_loss(pred_next_norm, target_next_norm)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                bsz = action_seq6.shape[0]
                running_loss += float(loss.item()) * bsz
                n_items += bsz
                global_step += 1

                progress.set_postfix(loss=f"{loss.item():.6f}", step=global_step)

                if args.wandb_enabled and (global_step % args.wandb_log_steps == 0):
                    wandb.log(
                        {
                            "train/mse": float(loss.item()),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/step": int(global_step),
                            "train/epoch": int(epoch),
                            "train/dataset_size": int(len(dataset)),
                            "train/batch_size": int(args.batch_size),
                        },
                        step=global_step,
                    )

            epoch_loss = running_loss / max(1, n_items)
            logging.info("Epoch %d/%d - train_mse: %.6f", epoch, args.epochs, epoch_loss)

            if (epoch % args.save_every == 0) or (epoch == args.epochs):
                ckpt_path = save_dir / f"dynamics_epoch_{epoch:03d}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_mse": epoch_loss,
                        "model_kwargs": {
                            "pose_dim": 6,
                            "action_dim": 6,
                            "action_num": args.horizon,
                            "hidden_size": args.hidden_size,
                            "num_layers": args.num_layers,
                        },
                        "dyn_stats": {
                            "dyn_action_01": dyn_stats["dyn_action_01"].tolist(),
                            "dyn_action_99": dyn_stats["dyn_action_99"].tolist(),
                            "dyn_pose_01": dyn_stats["dyn_pose_01"].tolist(),
                            "dyn_pose_99": dyn_stats["dyn_pose_99"].tolist(),
                        },
                    },
                    ckpt_path,
                )
                logging.info("Saved checkpoint: %s", ckpt_path)

                if args.wandb_enabled:
                    wandb.log({"train/checkpoint_epoch": int(epoch)}, step=global_step)
    finally:
        if args.wandb_enabled and wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
