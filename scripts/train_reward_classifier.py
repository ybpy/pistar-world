from __future__ import annotations

import argparse
import io
import json
import logging
import pathlib
import random
from dataclasses import dataclass
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm

from openpi.models.reward_classifier import RewardClassifier

try:
    import wandb
except Exception:
    wandb = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {value}")


def _decode_image_cell(cell: Any, root_dir: pathlib.Path) -> np.ndarray:
    if isinstance(cell, np.ndarray):
        img = cell
    elif isinstance(cell, Image.Image):
        img = np.asarray(cell)
    elif isinstance(cell, (bytes, bytearray)):
        img = iio.imread(io.BytesIO(cell))
    elif isinstance(cell, dict):
        if cell.get("bytes") is not None:
            img = iio.imread(io.BytesIO(cell["bytes"]))
        elif cell.get("path"):
            p = pathlib.Path(cell["path"])
            if not p.is_absolute():
                p = root_dir / p
            img = iio.imread(p)
        else:
            raise ValueError(f"Unsupported image dict keys: {list(cell.keys())}")
    else:
        raise ValueError(f"Unsupported image cell type: {type(cell)}")

    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    return img.astype(np.uint8)


def _as_scalar_int(value: Any) -> int:
    arr = np.asarray(value).reshape(-1)
    if arr.size <= 0:
        raise ValueError(f"Cannot parse int scalar from value: {value}")
    return int(arr[0])


def _as_scalar_float(value: Any) -> float:
    arr = np.asarray(value).reshape(-1)
    if arr.size <= 0:
        raise ValueError(f"Cannot parse float scalar from value: {value}")
    return float(arr[0])


@dataclass
class SampleRecord:
    episode_id: int
    parquet_path: str
    row_index: int
    label: int


class LiberoRewardClassifierDataset(Dataset):
    def __init__(self, records: list[SampleRecord], dataset_root: str, image_size: int):
        self.records = records
        self.dataset_root = pathlib.Path(dataset_root)
        self.image_size = int(image_size)

        self._cache_path: str | None = None
        self._cache_df: pd.DataFrame | None = None

    def __len__(self):
        return len(self.records)

    def _load_parquet(self, parquet_path: str) -> pd.DataFrame:
        if self._cache_path == parquet_path and self._cache_df is not None:
            return self._cache_df

        df = pd.read_parquet(parquet_path, columns=["image", "wrist_image", "reward"])
        self._cache_path = parquet_path
        self._cache_df = df
        return df

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rec = self.records[index]
        df = self._load_parquet(rec.parquet_path)

        row = df.iloc[int(rec.row_index)]
        image = _decode_image_cell(row["image"], self.dataset_root)
        wrist = _decode_image_cell(row["wrist_image"], self.dataset_root)

        image_t = RewardClassifier.preprocess_numpy_image(image, self.image_size)
        wrist_t = RewardClassifier.preprocess_numpy_image(wrist, self.image_size)

        return {
            "image": image_t,
            "wrist_image": wrist_t,
            "label": torch.tensor(float(rec.label), dtype=torch.float32),
        }


def resolve_dataset_root(dataset_repo_or_path: str) -> pathlib.Path:
    path = pathlib.Path(dataset_repo_or_path).expanduser()
    if path.exists():
        return path.resolve()

    # fallback: resolve as LeRobot repo_id under HF cache
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    except ImportError as exc:
        raise FileNotFoundError(
            f"Dataset path does not exist: {dataset_repo_or_path}. "
            "For repo_id resolution please install lerobot."
        ) from exc

    repo_path = pathlib.Path(HF_LEROBOT_HOME) / dataset_repo_or_path
    if repo_path.exists():
        return repo_path.resolve()

    raise FileNotFoundError(
        f"Could not resolve dataset_repo_or_path={dataset_repo_or_path}. "
        f"Tried as path and as HF_LEROBOT_HOME/repo_id ({repo_path})."
    )


def _load_tasks_map(dataset_root: pathlib.Path) -> dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return {}

    tasks_map: dict[int, str] = {}
    with open(tasks_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = int(row["task_index"])
            tasks_map[key] = str(row.get("task", "")).strip()
    return tasks_map


def build_records(
    dataset_root: pathlib.Path,
    task_index: int,
    sample_mode: str,
) -> tuple[list[SampleRecord], dict[int, list[SampleRecord]]]:
    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise ValueError(f"No parquet files found under {(dataset_root / 'data')}")

    records: list[SampleRecord] = []
    records_by_episode: dict[int, list[SampleRecord]] = {}

    for parquet_path in parquet_paths:
        try:
            df = pd.read_parquet(parquet_path, columns=["task_index", "reward"])
        except Exception as exc:
            raise RuntimeError(f"Failed to read {parquet_path}") from exc

        if len(df) <= 0:
            continue

        ep_task_index = _as_scalar_int(df["task_index"].iloc[0])
        if ep_task_index != int(task_index):
            continue

        episode_id = int(parquet_path.stem.split("_")[-1])
        if sample_mode == "last_step_only":
            row_indices = [len(df) - 1]
        elif sample_mode == "all_steps":
            row_indices = list(range(len(df)))
        else:
            raise ValueError(f"Unsupported sample_mode: {sample_mode}")

        for ridx in row_indices:
            reward = _as_scalar_float(df["reward"].iloc[ridx])
            label = int(1 if reward >= 0.5 else 0)
            rec = SampleRecord(
                episode_id=episode_id,
                parquet_path=str(parquet_path),
                row_index=int(ridx),
                label=label,
            )
            records.append(rec)
            records_by_episode.setdefault(episode_id, []).append(rec)

    if not records:
        raise ValueError(
            f"No samples found for task_index={task_index} with sample_mode={sample_mode} in {dataset_root}"
        )

    return records, records_by_episode


def split_train_val_by_episode(
    records_by_episode: dict[int, list[SampleRecord]],
    val_ratio: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    episode_ids = sorted(records_by_episode.keys())
    rng = random.Random(seed)
    rng.shuffle(episode_ids)

    if len(episode_ids) <= 1 or val_ratio <= 0:
        train_ids = episode_ids
        val_ids: list[int] = []
    else:
        val_n = max(1, int(round(len(episode_ids) * val_ratio)))
        val_n = min(val_n, len(episode_ids) - 1)
        val_ids = episode_ids[:val_n]
        train_ids = episode_ids[val_n:]

    train_records = [r for eid in train_ids for r in records_by_episode[eid]]
    val_records = [r for eid in val_ids for r in records_by_episode[eid]]
    return train_records, val_records


def _random_crop_with_pad(batch: torch.Tensor, pad: int = 4) -> torch.Tensor:
    if pad <= 0:
        return batch
    bsz, channels, height, width = batch.shape
    padded = F.pad(batch, (pad, pad, pad, pad), mode="replicate")
    out = torch.empty_like(batch)
    max_offset = 2 * pad
    for i in range(bsz):
        y0 = int(torch.randint(0, max_offset + 1, (1,), device=batch.device).item())
        x0 = int(torch.randint(0, max_offset + 1, (1,), device=batch.device).item())
        out[i] = padded[i, :, y0 : y0 + height, x0 : x0 + width]
    return out


def _evaluate(
    model: RewardClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.75,
) -> dict[str, float]:
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    n_total = 0
    total_loss = 0.0
    total_acc_threshold = 0.0
    total_acc_05 = 0.0

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            wrist = batch["wrist_image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.float32)

            logits = model(image=image, wrist_image=wrist).squeeze(-1)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds_threshold = (probs >= float(threshold)).to(labels.dtype)
            preds_05 = (probs >= 0.5).to(labels.dtype)
            acc_threshold = (preds_threshold == labels).to(torch.float32).mean()
            acc_05 = (preds_05 == labels).to(torch.float32).mean()

            bsz = labels.shape[0]
            n_total += bsz
            total_loss += float(loss.item()) * bsz
            total_acc_threshold += float(acc_threshold.item()) * bsz
            total_acc_05 += float(acc_05.item()) * bsz

    if n_total <= 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / n_total,
        "accuracy_threshold": total_acc_threshold / n_total,
        "accuracy_05": total_acc_05 / n_total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_repo_or_path",
        type=str,
        default="/public/home/chenyuyao1/.cache/huggingface/lerobot/ybpy/libero_pistar_rc",
    )
    parser.add_argument("--task_index", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--save_dir", type=str, default="checkpoints/reward_classifier")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_mode", type=str, default="all_steps", choices=["last_step_only", "all_steps"])

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--augment", type=str2bool, default=True)
    parser.add_argument("--balanced_sampling", type=str2bool, default=True)
    parser.add_argument("--encoder_type", type=str, default="resnet18_imagenet_frozen")
    parser.add_argument("--backbone_ckpt_path", type=str, default=None)
    parser.add_argument("--wandb_enabled", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="pistar_rc")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_resume", type=str2bool, default=False)
    parser.add_argument("--wandb_resume_mode", type=str, default="allow")
    return parser.parse_args()


def _wandb_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(vars(args))
    for k, v in list(cfg.items()):
        try:
            json.dumps(v)
        except TypeError:
            cfg[k] = str(v)
    return cfg


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if int(args.image_size) != 256:
        raise ValueError(
            "This reward-classifier pipeline assumes fixed 256x256 inputs for both training and rollout. "
            f"Got --image_size={args.image_size}."
        )

    dataset_root = resolve_dataset_root(args.dataset_repo_or_path)
    tasks_map = _load_tasks_map(dataset_root)
    task_name = tasks_map.get(int(args.task_index), "")
    out_dir = pathlib.Path(args.save_dir) / f"task_{int(args.task_index):03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        log_with="wandb" if bool(args.wandb_enabled) else None,
        project_dir=str(out_dir / "logs"),
    )
    if accelerator.is_main_process and bool(args.wandb_enabled):
        if wandb is None:
            raise ImportError("wandb_enabled=True but wandb is not installed. Please install wandb.")
        wandb_id_path = out_dir / "wandb_id.txt"
        resume_wandb_run = bool(args.wandb_resume)
        run_id = None
        if resume_wandb_run and wandb_id_path.exists():
            run_id = wandb_id_path.read_text().strip()
        if not run_id:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
        wandb_init_kwargs = dict(
            name=str(args.wandb_run_name) if args.wandb_run_name else f"reward_cls_task_{int(args.task_index):03d}",
            id=run_id,
            resume=resume_wandb_run,
            entity=str(args.wandb_entity) if args.wandb_entity else None,
            config_exclude_keys=[],
        )
        accelerator.init_trackers(
            str(args.wandb_project),
            config=_wandb_config_from_args(args),
            init_kwargs={"wandb": wandb_init_kwargs},
        )
        wandb_id_path.write_text(str(run_id))

    all_records, records_by_episode = build_records(
        dataset_root=dataset_root,
        task_index=int(args.task_index),
        sample_mode=str(args.sample_mode),
    )
    train_records, val_records = split_train_val_by_episode(
        records_by_episode=records_by_episode,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    if not train_records:
        raise ValueError("Train split is empty")

    train_labels = np.asarray([r.label for r in train_records], dtype=np.int64)
    unique_labels = sorted(set(train_labels.tolist()))
    if len(unique_labels) < 2:
        raise ValueError(
            "Training labels contain only one class. "
            "Need both positive and negative samples for binary classifier."
        )

    train_dataset = LiberoRewardClassifierDataset(
        records=train_records,
        dataset_root=str(dataset_root),
        image_size=int(args.image_size),
    )
    val_dataset = LiberoRewardClassifierDataset(
        records=val_records,
        dataset_root=str(dataset_root),
        image_size=int(args.image_size),
    )

    sampler = None
    shuffle = True
    if args.balanced_sampling:
        class_counts = np.bincount(train_labels, minlength=2).astype(np.float64)
        if np.any(class_counts <= 0):
            raise ValueError(f"Cannot use balanced sampling with missing class: counts={class_counts.tolist()}")
        sample_weights = np.asarray([1.0 / class_counts[label] for label in train_labels], dtype=np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardClassifier(
        image_size=int(args.image_size),
        hidden_dim=256,
        num_spatial_blocks=8,
        encoder_bottleneck_dim=256,
        dropout_p=0.1,
        backbone_type=str(args.encoder_type),
        backbone_ckpt_path=args.backbone_ckpt_path,
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    logging.info("dataset_root=%s", dataset_root)
    logging.info("task_index=%d task_name=%s", args.task_index, task_name)
    logging.info("sample_mode=%s train_samples=%d val_samples=%d", args.sample_mode, len(train_dataset), len(val_dataset))
    logging.info("train label counts: neg=%d pos=%d", int((train_labels == 0).sum()), int((train_labels == 1).sum()))
    if accelerator.is_main_process and bool(args.wandb_enabled):
        accelerator.log(
            {
                "dataset/train_samples": int(len(train_dataset)),
                "dataset/val_samples": int(len(val_dataset)),
                "dataset/train_positive": int(np.sum(train_labels == 1)),
                "dataset/train_negative": int(np.sum(train_labels == 0)),
                "dataset/task_index": int(args.task_index),
            },
            step=0,
        )

    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        running_acc_threshold = 0.0
        running_acc_05 = 0.0
        n_items = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            image = batch["image"].to(device=device, dtype=torch.float32)
            wrist = batch["wrist_image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.float32)

            if args.augment:
                image = _random_crop_with_pad(image, pad=4)
                wrist = _random_crop_with_pad(wrist, pad=4)

            logits = model(image=image, wrist_image=wrist).squeeze(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds_threshold = (probs >= float(args.threshold)).to(labels.dtype)
                preds_05 = (probs >= 0.5).to(labels.dtype)
                acc_threshold = (preds_threshold == labels).to(torch.float32).mean()
                acc_05 = (preds_05 == labels).to(torch.float32).mean()

            bsz = labels.shape[0]
            n_items += bsz
            running_loss += float(loss.item()) * bsz
            running_acc_threshold += float(acc_threshold.item()) * bsz
            running_acc_05 += float(acc_05.item()) * bsz

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc_th=f"{acc_threshold.item():.4f}",
            )

        train_loss = running_loss / max(1, n_items)
        train_acc_threshold = running_acc_threshold / max(1, n_items)
        train_acc_05 = running_acc_05 / max(1, n_items)

        if len(val_dataset) > 0:
            val_metrics = _evaluate(model, val_loader, device=device, threshold=float(args.threshold))
            val_loss = float(val_metrics["loss"])
            val_acc_threshold = float(val_metrics["accuracy_threshold"])
            val_acc_05 = float(val_metrics["accuracy_05"])
        else:
            val_loss = 0.0
            val_acc_threshold = 0.0
            val_acc_05 = 0.0

        logging.info(
            "epoch=%d train_loss=%.6f train_acc@th=%.4f train_acc@0.5=%.4f val_loss=%.6f val_acc@th=%.4f val_acc@0.5=%.4f",
            epoch,
            train_loss,
            train_acc_threshold,
            train_acc_05,
            val_loss,
            val_acc_threshold,
            val_acc_05,
        )

        if len(val_dataset) > 0:
            score = val_acc_threshold
        else:
            score = train_acc_threshold

        if score > best_val_acc:
            best_val_acc = score
            best_epoch = epoch

            model_path = out_dir / "model.pt"
            meta_path = out_dir / "meta.json"

            meta = {
                "task_index": int(args.task_index),
                "task_name": task_name,
                "threshold": float(args.threshold),
                "image_size": int(args.image_size),
                "backbone": str(args.encoder_type),
                "backbone_ckpt_path": str(args.backbone_ckpt_path) if args.backbone_ckpt_path else None,
                "encoder_type": "frozen_resnet18 + spatial_learned_embeddings + bottleneck256",
                "sample_mode": str(args.sample_mode),
                "dataset_repo_or_path": str(args.dataset_repo_or_path),
                "resolved_dataset_path": str(dataset_root),
                "train_samples": int(len(train_dataset)),
                "val_samples": int(len(val_dataset)),
                "train_positive": int(np.sum(train_labels == 1)),
                "train_negative": int(np.sum(train_labels == 0)),
                "best_epoch": int(best_epoch),
                "best_score": float(best_val_acc),
                "train_loss": float(train_loss),
                "train_accuracy_threshold": float(train_acc_threshold),
                "train_accuracy_05": float(train_acc_05),
                "val_loss": float(val_loss),
                "val_accuracy_threshold": float(val_acc_threshold),
                "val_accuracy_05": float(val_acc_05),
                "label_definition": {
                    "positive": "reward == 1",
                    "negative": "reward == 0",
                },
                "best_model_selection_metric": "val_accuracy_threshold" if len(val_dataset) > 0 else "train_accuracy_threshold",
            }

            model.save_checkpoint(model_path, threshold=float(args.threshold), meta=meta)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            logging.info("Saved best checkpoint to %s", model_path)

        if accelerator.is_main_process and bool(args.wandb_enabled):
            payload = {
                "epoch": int(epoch),
                "train/loss": float(train_loss),
                "train/accuracy_threshold": float(train_acc_threshold),
                "train/accuracy_05": float(train_acc_05),
                "val/loss": float(val_loss),
                "val/accuracy_threshold": float(val_acc_threshold),
                "val/accuracy_05": float(val_acc_05),
                "best/score": float(best_val_acc),
                "best/epoch": int(best_epoch),
                "config/threshold": float(args.threshold),
            }
            accelerator.log(payload, step=epoch)

    logging.info("Training done. best_epoch=%d best_score=%.4f", best_epoch, best_val_acc)
    accelerator.end_training()

if __name__ == "__main__":
    main()
