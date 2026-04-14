from __future__ import annotations

import dataclasses
import json
import math
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import einops
import mediapy
import numpy as np
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

try:
    import wandb
except Exception:
    wandb = None

from openpi.models.ctrl_world import CtrlWorld
from openpi.models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from openpi.training.config_wm import wm_args


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {value}")


class LiberoWMDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode: str = "train"):
        super().__init__()
        self.args = args
        self.mode = mode

        dataset_names = [x for x in str(args.dataset_names).split("+") if x]
        dataset_cfgs = [x for x in str(args.dataset_cfgs).split("+") if x]
        if len(dataset_cfgs) == 0:
            dataset_cfgs = dataset_names
        if len(dataset_names) != len(dataset_cfgs):
            raise ValueError(
                f"dataset_names ({dataset_names}) and dataset_cfgs ({dataset_cfgs}) must have same length"
            )

        self.items: list[dict] = []
        self.norm_by_dataset: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for dataset_name, dataset_cfg in zip(dataset_names, dataset_cfgs):
            dataset_root = Path(args.dataset_root_path) / dataset_name
            sample_path = Path(args.dataset_meta_info_path) / dataset_cfg / f"{mode}_sample.json"
            stat_path = Path(args.dataset_meta_info_path) / dataset_cfg / "stat.json"

            if not sample_path.exists():
                raise FileNotFoundError(f"Sample file not found: {sample_path}")
            if not stat_path.exists():
                raise FileNotFoundError(f"Stat file not found: {stat_path}")

            with open(sample_path, "r") as f:
                samples = json.load(f)
            with open(stat_path, "r") as f:
                stat = json.load(f)

            if "wm_state_01" not in stat or "wm_state_99" not in stat:
                raise ValueError(
                    f"Stat file must contain canonical keys wm_state_01/wm_state_99: {stat_path}"
                )
            state_p01 = np.asarray(stat["wm_state_01"], dtype=np.float32)[None, :]
            state_p99 = np.asarray(stat["wm_state_99"], dtype=np.float32)[None, :]
            if state_p01.shape[-1] != 7 or state_p99.shape[-1] != 7:
                raise ValueError(
                    f"Expected 7D WM pose normalization bounds, got "
                    f"wm_state_01={state_p01.shape}, wm_state_99={state_p99.shape} at {stat_path}"
                )
            self.norm_by_dataset[dataset_name] = (state_p01, state_p99)

            for sample in samples:
                self.items.append(
                    {
                        "dataset_name": dataset_name,
                        "dataset_root": dataset_root,
                        "sample": sample,
                    }
                )

        if len(self.items) == 0:
            raise RuntimeError(f"No samples loaded for mode={mode}")

        print(f"[INFO] {mode}: loaded {len(self.items)} samples from {dataset_names}")

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def normalize_bound(
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    @staticmethod
    def _load_latent_video(path: Path, frame_ids: np.ndarray) -> torch.Tensor:
        lat = torch.load(path, map_location="cpu")
        if isinstance(lat, np.ndarray):
            lat = torch.from_numpy(lat)
        lat = lat.float()
        return lat[frame_ids]

    def _build_rgb_indices(self, frame_now: int, video_length: int) -> np.ndarray:
        """Build deterministic WM indices on downsampled 5Hz timeline.

        Sample `frame_ids[0]` is the current window start `t` (5Hz index).
        We always construct:
          history = [oldest, t-12, t-9, t-6, t-4, t-2, t-1]
          future  = [t, t+1, t+2, t+3, t+4]

                Rules:
                - history insufficiency is allowed and clipped to dynamic oldest
                    oldest = max(0, t - wm_history_cache_len + 1)
        - future insufficiency is NOT allowed; raise error
          (meta generation should already filter such samples)
        """
        t = int(frame_now)
        n = int(video_length)

        if n <= 0:
            raise ValueError(f"video_length must be positive, got {video_length}")

        if t < 0 or t >= n:
            raise ValueError(f"frame_now out of range: t={t}, video_length={n}")

        future = [t + i for i in range(int(self.args.num_frames))]
        if future[-1] >= n:
            raise ValueError(
                f"Future window out of range for sample t={t}: "
                f"need [t..t+{self.args.num_frames - 1}] within video_length={n}."
            )

        if not bool(getattr(self.args, "history_use_oldest_anchor", True)):
            raise ValueError("history_use_oldest_anchor must be True")

        rel = list(getattr(self.args, "history_relative_offsets", [-12, -9, -6, -4, -2, -1]))
        if len(rel) != int(self.args.num_history) - 1:
            raise ValueError(
                f"history_relative_offsets length must be num_history-1={self.args.num_history - 1}, got {len(rel)}"
            )

        cache_len = int(getattr(self.args, "wm_history_cache_len", 56))
        oldest = max(0, t - cache_len + 1)
        history = [oldest] + [max(oldest, t + int(o)) for o in rel]
        rgb_id = np.asarray(history + future, dtype=np.int64)

        if rgb_id.shape[0] != int(self.args.num_history + self.args.num_frames):
            raise ValueError(
                f"Unexpected rgb_id length {rgb_id.shape[0]}, "
                f"expect {self.args.num_history + self.args.num_frames}"
            )
        return rgb_id

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        dataset_name = item["dataset_name"]
        dataset_root: Path = item["dataset_root"]
        sample = item["sample"]

        episode_id = int(sample["episode_id"])
        frame_now = int(sample["frame_ids"][0])

        ann_path = dataset_root / self.args.annotation_name / self.mode / f"{episode_id}.json"
        with open(ann_path, "r") as f:
            ann = json.load(f)

        video_length = int(ann["video_length"])

        rgb_id = self._build_rgb_indices(frame_now=frame_now, video_length=video_length)

        lat0 = self._load_latent_video(
            dataset_root / self.args.latent_video_name / self.mode / str(episode_id) / "0.pt",
            rgb_id,
        )
        lat1 = self._load_latent_video(
            dataset_root / self.args.latent_video_name / self.mode / str(episode_id) / "1.pt",
            rgb_id,
        )
        lat2 = self._load_latent_video(
            dataset_root / self.args.latent_video_name / self.mode / str(episode_id) / "2.pt",
            rgb_id,
        )

        latent = torch.zeros((self.args.num_history + self.args.num_frames, 4, 72, 40), dtype=torch.float32)
        latent[:, :, 0:24] = lat0
        latent[:, :, 24:48] = lat1
        latent[:, :, 48:72] = lat2

        # WM conditioning uses abs_pose only (7D), never observation.state.
        # Shape: [num_history + num_frames, 7]
        pose7 = np.asarray(ann["abs_pose"], dtype=np.float32)
        if pose7.ndim != 2 or pose7.shape[1] != 7:
            raise ValueError(f"Expected abs_pose [N,7], got {pose7.shape} in {ann_path}")
        action = pose7[rgb_id]

        state_p01, state_p99 = self.norm_by_dataset[dataset_name]

        # wm_state_* only normalizes the first 6 continuous pose dimensions.
        # Gripper (last dim) is passed through without percentile normalization.
        pose6 = action[:, :6]
        gripper = action[:, 6:7]
        state_p01_6 = state_p01[:, :6]
        state_p99_6 = state_p99[:, :6]

        sat_mode = int(getattr(self.args, "sat_mode", 1))
        if sat_mode == 1 or sat_mode == 2:
            # Saturation diagnostics are computed on pose6 only, to avoid binary
            # gripper dominating saturation ratio.
            raw_norm_6 = 2 * (pose6 - state_p01_6) / (state_p99_6 - state_p01_6 + 1e-8) - 1
            sat_ratio = float(((raw_norm_6 < -1.0) | (raw_norm_6 > 1.0)).mean())

            emit_level = None
            if sat_ratio > float(getattr(self.args, "sat_ratio_fail", 0.30)):
                emit_level = "ERROR"
                print(
                    f"[ERROR] severe training wm_cond pose saturation ({sat_ratio:.2%}) "
                    f"episode={episode_id}, mode={self.mode}"
                )
            elif sat_ratio > float(getattr(self.args, "sat_ratio_warn", 0.15)):
                emit_level = "WARN"
                print(
                    f"[WARN] high training wm_cond pose saturation ({sat_ratio:.2%}) "
                    f"episode={episode_id}, mode={self.mode}"
                )

            if sat_mode == 2 and emit_level is not None:
                dim_sat_ratio = ((raw_norm_6 < -1.0) | (raw_norm_6 > 1.0)).mean(axis=0)
                dim_low_ratio = (raw_norm_6 < -1.0).mean(axis=0)
                dim_high_ratio = (raw_norm_6 > 1.0).mean(axis=0)
                dim_min = raw_norm_6.min(axis=0)
                dim_max = raw_norm_6.max(axis=0)

                parts = []
                for j in range(raw_norm_6.shape[1]):
                    parts.append(
                        f"dim{j}:sat={float(dim_sat_ratio[j]):.2f},"
                        f"low={float(dim_low_ratio[j]):.2f},"
                        f"high={float(dim_high_ratio[j]):.2f},"
                        f"min={float(dim_min[j]):.2f},"
                        f"max={float(dim_max[j]):.2f}"
                    )
                print(f"[{emit_level}] wm_cond pose detail episode={episode_id} mode={self.mode} {' | '.join(parts)}")

        pose6_norm = self.normalize_bound(pose6, state_p01_6, state_p99_6)
        action = np.concatenate([pose6_norm, gripper], axis=-1)

        text = str((ann.get("texts") or [""])[0])

        return {
            "latent": latent,
            "action": torch.tensor(action, dtype=torch.float32),
            "text": text,
        }


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def validate_video_generation(model, val_dataset, args, train_steps: int, videos_dir: str, accelerator: Accelerator):
    if len(val_dataset) == 0:
        return None

    base_model = _unwrap_model(model)
    pipeline = base_model.pipeline
    device = accelerator.device

    videos_row = max(1, int(args.video_num))
    videos_col = 2
    stride = max(1, int(len(val_dataset) / max(1, videos_row * videos_col)))
    batch_id = list(range(0, len(val_dataset), stride))[: videos_row * videos_col]
    if len(batch_id) == 0:
        batch_id = [0]

    batch_list = [val_dataset.__getitem__(idx) for idx in batch_id]
    video_gt = torch.stack([b["latent"] for b in batch_list], dim=0).to(device, non_blocking=True)
    text = [b["text"] for b in batch_list]
    actions = torch.stack([b["action"] for b in batch_list], dim=0).to(device, non_blocking=True)

    his_latent_gt = video_gt[:, : args.num_history]
    future_latent_gt = video_gt[:, args.num_history :]
    current_latent = future_latent_gt[:, 0]

    with torch.no_grad():
        action_latent = base_model.action_encoder(
            actions,
            text,
            base_model.tokenizer,
            base_model.text_encoder,
            args.frame_level_cond,
        )

        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(3 * args.height),
            num_frames=args.num_frames,
            history=his_latent_gt,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type="latent",
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )

    pred_latents = einops.rearrange(pred_latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1)
    video_gt_full = torch.cat([his_latent_gt, future_latent_gt], dim=1)
    video_gt_full = einops.rearrange(video_gt_full, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1)

    def _decode(v_latent: torch.Tensor) -> torch.Tensor:
        decoded = []
        bsz, frame_num = v_latent.shape[:2]
        x = v_latent.flatten(0, 1)
        decode_kwargs = {}
        for i in range(0, x.shape[0], args.decode_chunk_size):
            chunk = x[i : i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        x = torch.cat(decoded, dim=0)
        return x.reshape(bsz, frame_num, *x.shape[1:])

    video_gt_img = _decode(video_gt_full)
    pred_video_img = _decode(pred_latents)

    video_gt_img = ((video_gt_img / 2.0 + 0.5).clamp(0, 1) * 255)
    video_gt_img = video_gt_img.detach().cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

    pred_video_img = ((pred_video_img / 2.0 + 0.5).clamp(0, 1) * 255)
    pred_video_img = pred_video_img.detach().cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

    pred_video_img = np.concatenate([video_gt_img[:, : args.num_history], pred_video_img], axis=1)
    video_vis = np.concatenate([video_gt_img, pred_video_img], axis=-3)
    video_vis = np.concatenate([v for v in video_vis], axis=-2).astype(np.uint8)

    os.makedirs(f"{videos_dir}/samples", exist_ok=True)
    filename = f"{videos_dir}/samples/train_steps_{train_steps}.mp4"
    mediapy.write_video(filename, video_vis, fps=2)
    return filename


def _wandb_config_from_args(args) -> dict:
    cfg = dataclasses.asdict(args)
    for k, v in list(cfg.items()):
        try:
            json.dumps(v)
        except TypeError:
            cfg[k] = str(v)
    return cfg


def merge_args(cfg, cli_args):
    for k, v in vars(cli_args).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def build_optimizer(model, args):
    base_lr = float(args.learning_rate)
    unet_lr = float(args.unet_learning_rate) if args.unet_learning_rate is not None else base_lr
    action_encoder_lr = (
        float(args.action_encoder_learning_rate)
        if args.action_encoder_learning_rate is not None
        else base_lr
    )

    unet_params = [p for p in model.unet.parameters() if p.requires_grad]
    action_encoder_params = [p for p in model.action_encoder.parameters() if p.requires_grad]

    if len(unet_params) == 0 and len(action_encoder_params) == 0:
        raise ValueError("No trainable params found in model.unet and model.action_encoder")

    optimizer = torch.optim.AdamW(
        [
            {"params": unet_params, "lr": unet_lr, "name": "unet"},
            {"params": action_encoder_params, "lr": action_encoder_lr, "name": "action_encoder"},
        ],
        lr=base_lr,
    )
    return optimizer


def build_parser() -> ArgumentParser:
    # Parser only provides optional overrides for config_wm.py defaults.
    # default=None means: do not override config default unless CLI explicitly sets it.
    parser = ArgumentParser()

    parser.add_argument("--svd_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    # Training-only resume checkpoint. Inference checkpoint is managed by rollout script via --ckpt_path.
    parser.add_argument("--resume_ckpt_path", type=str, default=None)

    parser.add_argument("--dataset_root_path", type=str, default=None)
    parser.add_argument("--dataset_meta_info_path", type=str, default=None)
    parser.add_argument("--dataset_names", type=str, default=None)
    parser.add_argument("--dataset_cfgs", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sat_mode", type=int, default=None)
    parser.add_argument("--unet_learning_rate", type=float, default=None)
    parser.add_argument("--action_encoder_learning_rate", type=float, default=None)

    return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    wandb_run = None
    if accelerator.is_main_process and bool(getattr(args, "wandb_enabled", False)):
        if wandb is None:
            raise ImportError("wandb_enabled=True but wandb is not installed")

        wandb_id_path = Path(args.output_dir) / "wandb_id.txt"
        init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": _wandb_config_from_args(args),
        }
        if getattr(args, "wandb_entity", None):
            init_kwargs["entity"] = args.wandb_entity

        if bool(getattr(args, "wandb_resume", False)) and wandb_id_path.exists():
            run_id = wandb_id_path.read_text().strip()
            if run_id:
                init_kwargs["id"] = run_id
                init_kwargs["resume"] = str(getattr(args, "wandb_resume_mode", "allow"))

        wandb_run = wandb.init(**init_kwargs)
        if wandb_run is not None and getattr(wandb_run, "id", None):
            wandb_id_path.write_text(str(wandb_run.id))

    model = CtrlWorld(args)
    # Training checkpoint semantics:
    # - default: fresh start from SVD initialization inside CtrlWorld(args)
    # - optional: resume only when resume_ckpt_path is explicitly provided
    resume_path = getattr(args, "resume_ckpt_path", None)
    if resume_path:
        if not Path(resume_path).exists():
            raise FileNotFoundError(
                f"resume_ckpt_path does not exist: {resume_path}. "
                "Provide a valid path or leave resume_ckpt_path unset for fresh training."
            )
        print(f"[INFO] Resuming world model training from {resume_path}")
        state_dict = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    else:
        print("[INFO] Training from SVD initialization (no resume checkpoint)")

    model.to(accelerator.device)
    model.train()

    optimizer = build_optimizer(model, args)

    train_dataset = LiberoWMDataset(args, mode="train")
    val_dataset = LiberoWMDataset(args, mode="val")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=max(0, min(args.num_workers, 2)),
        pin_memory=True,
        drop_last=False,
    )

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    steps_per_epoch = max(1, len(train_dataloader))
    num_train_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    print("***** Running WM training (libero_wm) *****")
    print(f"  Num train samples = {len(train_dataset)}")
    print(f"  Num val samples = {len(val_dataset)}")
    print(f"  Num epochs ~= {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size = {total_batch_size}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    train_loss = 0.0
    train_loss_count = 0
    log_every = int(getattr(args, "wandb_log_steps", 100))
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    try:
        for _epoch in range(num_train_epochs):
            for _step, batch in enumerate(train_dataloader):
                grad_norm_value = None
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        loss_gen, _ = model(batch)

                    avg_loss = accelerator.gather(loss_gen.repeat(args.train_batch_size)).mean()
                    train_loss += float(avg_loss.item())
                    train_loss_count += 1

                    accelerator.backward(loss_gen)
                    if accelerator.sync_gradients and float(args.max_grad_norm) > 0:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        grad_norm_value = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % log_every == 0:
                        smooth_loss = train_loss / max(1, train_loss_count)
                        progress_bar.set_postfix({"loss": smooth_loss})

                        if accelerator.is_main_process and bool(getattr(args, "wandb_enabled", False)) and wandb_run is not None:
                            group_lr = {
                                str(g.get("name", f"group_{i}")): float(g["lr"])
                                for i, g in enumerate(optimizer.param_groups)
                            }
                            payload = {
                                "train/loss": smooth_loss,
                                "train/lr": float(args.learning_rate),
                                "train/lr_unet": group_lr.get("unet"),
                                "train/lr_action_encoder": group_lr.get("action_encoder"),
                                "train/step": int(global_step),
                            }
                            if grad_norm_value is not None:
                                payload["train/grad_norm"] = float(grad_norm_value)
                            wandb.log(payload, step=global_step)

                        train_loss = 0.0
                        train_loss_count = 0

                    if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                        torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
                        print(f"[INFO] Saved checkpoint to {save_path}")

                    if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                        model.eval()
                        with accelerator.autocast():
                            video_path = validate_video_generation(model, val_dataset, args, global_step, args.output_dir, accelerator)
                        model.train()

                        if (
                            bool(getattr(args, "wandb_enabled", False))
                            and bool(getattr(args, "wandb_log_validation_video", True))
                            and wandb_run is not None
                            and video_path is not None
                            and Path(video_path).exists()
                        ):
                            wandb.log(
                                {
                                    "val/sample_video": wandb.Video(str(video_path), fps=2, format="mp4"),
                                    "train/step": int(global_step),
                                },
                                step=global_step,
                            )

                    if global_step >= args.max_train_steps:
                        break

            if global_step >= args.max_train_steps:
                break
    finally:
        if accelerator.is_main_process and bool(getattr(args, "wandb_enabled", False)) and wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()

    args = wm_args()
    args = merge_args(args, cli_args)
    args.__post_init__()

    main(args)
