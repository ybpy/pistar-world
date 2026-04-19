from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLearnedEmbeddings(nn.Module):
    """PyTorch equivalent of hil-serl SpatialLearnedEmbeddings.

    Input:  x in NCHW
    Output: flattened weighted spatial embeddings in [B, C * K]
    """

    def __init__(self, channels: int, height: int, width: int, num_features: int = 8):
        super().__init__()
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.num_features = int(num_features)
        self.kernel = nn.Parameter(torch.empty(self.height, self.width, self.channels, self.num_features))
        nn.init.trunc_normal_(self.kernel, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x in [B,C,H,W], got {tuple(x.shape)}")
        bsz, channels, height, width = x.shape
        if channels != self.channels or height != self.height or width != self.width:
            raise ValueError(
                "SpatialLearnedEmbeddings input shape mismatch: "
                f"expected C/H/W={self.channels}/{self.height}/{self.width}, got {channels}/{height}/{width}"
            )

        # [B, C, H, W] -> [B, H, W, C]
        x_hwc = x.permute(0, 2, 3, 1)
        # same semantics as hil-serl:
        # sum_{h,w} features[h,w,c] * kernel[h,w,c,k]
        weighted = torch.sum(x_hwc.unsqueeze(-1) * self.kernel.unsqueeze(0), dim=(1, 2))
        # [B, C, K] -> [B, C*K]
        return weighted.reshape(bsz, -1)


class ResNetSpatialEncoder(nn.Module):
    """Frozen ImageNet ResNet trunk + spatial learned embeddings + bottleneck.

    This mirrors the hil-serl stack semantically:
    - frozen pretrained conv trunk
    - spatial learned embeddings pooling
    - dropout(0.1)
    - bottleneck linear + layer norm + tanh
    """

    def __init__(
        self,
        image_size: int = 256,
        num_spatial_blocks: int = 8,
        bottleneck_dim: int = 256,
        dropout_p: float = 0.1,
        backbone_ckpt_path: str | None = None,
    ):
        super().__init__()

        try:
            from torchvision.models import ResNet18_Weights
            from torchvision.models import resnet18
        except ImportError as exc:
            raise ImportError("torchvision is required for reward classifier encoder") from exc

        if backbone_ckpt_path is not None:
            resnet = resnet18(weights=None)
            state_dict = torch.load(backbone_ckpt_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            resnet.load_state_dict(state_dict, strict=True)
        else:
            weights = ResNet18_Weights.DEFAULT
            resnet = resnet18(weights=weights)

        # Keep conv trunk only (up to layer4), no avgpool/fc.
        self.trunk = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        for p in self.trunk.parameters():
            p.requires_grad = False
        # Keep pretrained trunk frozen in both gradient and running-stat sense.
        self.trunk.eval()

        self.image_size = int(image_size)
        self.dropout = nn.Dropout(float(dropout_p))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            feat = self.trunk(dummy)
            _, c, h, w = feat.shape

        self.pool = SpatialLearnedEmbeddings(channels=c, height=h, width=w, num_features=num_spatial_blocks)
        self.proj = nn.Linear(c * int(num_spatial_blocks), int(bottleneck_dim))
        self.ln = nn.LayerNorm(int(bottleneck_dim))

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # Allow train/eval toggling for learnable layers in this module,
        # but always keep pretrained trunk in eval mode to prevent BN drift.
        super().train(mode)
        self.trunk.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image in [B,C,H,W], got {tuple(image.shape)}")

        x = image.to(dtype=torch.float32) / 255.0
        x = (x - self.pixel_mean) / self.pixel_std

        # Keep pretrained trunk frozen (same spirit as hil-serl pretrained encoder).
        # Re-assert eval mode in case parent module train() was called.
        self.trunk.eval()
        with torch.no_grad():
            feat = self.trunk(x)

        pooled = self.pool(feat)
        pooled = self.dropout(pooled)
        out = self.proj(pooled)
        out = self.ln(out)
        out = torch.tanh(out)
        return out


class RewardClassifier(nn.Module):
    """Binary reward classifier with two independent image encoders.

    Inputs: `image`, `wrist_image` only.
    Output: single logit in [B, 1].
    """

    def __init__(
        self,
        image_size: int = 256,
        hidden_dim: int = 256,
        num_spatial_blocks: int = 8,
        encoder_bottleneck_dim: int = 256,
        dropout_p: float = 0.1,
        backbone_type: str = "resnet18_imagenet_frozen",
        backbone_ckpt_path: str | None = None,
    ):
        super().__init__()
        if backbone_type != "resnet18_imagenet_frozen":
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        self.image_size = int(image_size)
        self.hidden_dim = int(hidden_dim)
        self.num_spatial_blocks = int(num_spatial_blocks)
        self.encoder_bottleneck_dim = int(encoder_bottleneck_dim)
        self.dropout_p = float(dropout_p)
        self.backbone_type = str(backbone_type)
        self.backbone_ckpt_path = backbone_ckpt_path

        self.image_encoder = ResNetSpatialEncoder(
            image_size=self.image_size,
            num_spatial_blocks=self.num_spatial_blocks,
            bottleneck_dim=self.encoder_bottleneck_dim,
            dropout_p=self.dropout_p,
            backbone_ckpt_path=self.backbone_ckpt_path,
        )
        self.wrist_image_encoder = ResNetSpatialEncoder(
            image_size=self.image_size,
            num_spatial_blocks=self.num_spatial_blocks,
            bottleneck_dim=self.encoder_bottleneck_dim,
            dropout_p=self.dropout_p,
            backbone_ckpt_path=self.backbone_ckpt_path,
        )

        in_dim = self.encoder_bottleneck_dim * 2
        self.head_fc = nn.Linear(in_dim, self.hidden_dim)
        self.head_dropout = nn.Dropout(self.dropout_p)
        self.head_ln = nn.LayerNorm(self.hidden_dim)
        self.head_out = nn.Linear(self.hidden_dim, 1)

    def forward(self, image: torch.Tensor, wrist_image: torch.Tensor) -> torch.Tensor:
        feat_image = self.image_encoder(image)
        feat_wrist = self.wrist_image_encoder(wrist_image)
        x = torch.cat([feat_image, feat_wrist], dim=-1)
        x = self.head_fc(x)
        x = self.head_dropout(x)
        x = self.head_ln(x)
        x = F.relu(x)
        return self.head_out(x)

    @staticmethod
    def _ensure_3ch_uint8(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape {arr.shape}")
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels, got shape {arr.shape}")
        return np.ascontiguousarray(arr.astype(np.uint8, copy=False))

    @classmethod
    def preprocess_numpy_image(cls, image: np.ndarray, image_size: int) -> torch.Tensor:
        arr = cls._ensure_3ch_uint8(image)
        t = torch.from_numpy(arr).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False)
        return t.squeeze(0)

    def infer_numpy(self, image: np.ndarray, wrist_image: np.ndarray, device: torch.device | None = None) -> dict[str, float]:
        dev = device if device is not None else next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            image_t = self.preprocess_numpy_image(image, self.image_size).unsqueeze(0).to(dev)
            wrist_t = self.preprocess_numpy_image(wrist_image, self.image_size).unsqueeze(0).to(dev)
            logit_t = self.forward(image_t, wrist_t).reshape(-1)
            logit = float(logit_t[0].item())
            prob = float(torch.sigmoid(logit_t)[0].item())
        return {"logit": logit, "prob": prob}

    def get_model_kwargs(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "hidden_dim": self.hidden_dim,
            "num_spatial_blocks": self.num_spatial_blocks,
            "encoder_bottleneck_dim": self.encoder_bottleneck_dim,
            "dropout_p": self.dropout_p,
            "backbone_type": self.backbone_type,
            "backbone_ckpt_path": self.backbone_ckpt_path,
        }

    def save_checkpoint(self, checkpoint_path: str | pathlib.Path, *, threshold: float, meta: dict[str, Any] | None = None) -> None:
        path = pathlib.Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "model_kwargs": self.get_model_kwargs(),
            "threshold": float(threshold),
            "meta": meta or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | pathlib.Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["RewardClassifier", dict[str, Any]]:
        payload = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            model_kwargs = dict(payload.get("model_kwargs", {}))
            model = cls(**model_kwargs)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            return model, payload

        # fallback: raw state_dict checkpoint
        model = cls(image_size=256)
        model.load_state_dict(payload, strict=True)
        return model, {"threshold": None, "meta": {}}


class RewardClassifierPredictor:
    def __init__(
        self,
        model: RewardClassifier,
        *,
        threshold: float,
        checkpoint_path: str,
        task_index: int | None,
        image_size: int,
        device: torch.device,
    ):
        self.model = model.eval().to(device)
        self.threshold = float(threshold)
        self.checkpoint_path = str(checkpoint_path)
        self.task_index = None if task_index is None else int(task_index)
        self.image_size = int(image_size)
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        meta_path: str | None = None,
        default_threshold: float | None = None,
        expected_image_size: int = 256,
        device: torch.device | None = None,
    ) -> "RewardClassifierPredictor":
        dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, payload = RewardClassifier.load_from_checkpoint(
            checkpoint_path,
            map_location=dev,
        )

        meta = {}
        if meta_path is not None and pathlib.Path(meta_path).exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)

        expected_image_size = int(expected_image_size)
        ckpt_image_size = None
        if isinstance(payload, dict):
            model_kwargs = payload.get("model_kwargs")
            if isinstance(model_kwargs, dict) and model_kwargs.get("image_size") is not None:
                ckpt_image_size = int(model_kwargs["image_size"])
        meta_image_size = int(meta["image_size"]) if isinstance(meta, dict) and meta.get("image_size") is not None else None

        if ckpt_image_size is not None and ckpt_image_size != expected_image_size:
            raise ValueError(
                "Reward classifier checkpoint image_size mismatch: "
                f"expected={expected_image_size}, checkpoint={ckpt_image_size}, path={checkpoint_path}"
            )
        if meta_image_size is not None and meta_image_size != expected_image_size:
            raise ValueError(
                "Reward classifier meta image_size mismatch: "
                f"expected={expected_image_size}, meta={meta_image_size}, meta_path={meta_path}"
            )
        if int(model.image_size) != expected_image_size:
            raise ValueError(
                "Reward classifier loaded model image_size mismatch: "
                f"expected={expected_image_size}, loaded={int(model.image_size)}, path={checkpoint_path}"
            )

        threshold = None
        if default_threshold is not None:
            threshold = float(default_threshold)
        elif "threshold" in meta:
            threshold = float(meta["threshold"])
        elif isinstance(payload, dict) and payload.get("threshold") is not None:
            threshold = float(payload["threshold"])
        else:
            threshold = 0.5

        task_index = meta.get("task_index") if isinstance(meta, dict) else None
        if task_index is None and isinstance(payload, dict):
            task_index = payload.get("task_index")

        return cls(
            model=model,
            threshold=threshold,
            checkpoint_path=checkpoint_path,
            task_index=task_index,
            image_size=expected_image_size,
            device=dev,
        )

    def predict_numpy(self, image: np.ndarray, wrist_image: np.ndarray) -> dict[str, Any]:
        out = self.model.infer_numpy(image=image, wrist_image=wrist_image, device=self.device)
        success = bool(out["prob"] >= self.threshold)
        return {
            "logit": float(out["logit"]),
            "prob": float(out["prob"]),
            "threshold": float(self.threshold),
            "success": success,
            "task_index": self.task_index,
            "checkpoint_path": self.checkpoint_path,
        }
