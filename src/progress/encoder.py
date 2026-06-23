"""Frozen V-JEPA2 encoder for trajectory prefix and full-trajectory encoding.

Provides a self-contained V-JEPA2 encoder wrapper that:
  - Loads vit_giant_xformers_rope with pretrained weights.
  - Handles frame sampling: linspace-64 for long videos, last-frame padding for short ones.
  - Applies Resize → CenterCrop → ClipToTensor → ImageNet normalization.
  - Returns mean-pooled embedding vector.

Public API::

    encoder = VJEPAEncoder(model_path, img_size=384, enable_fp16=True)
    h_full = encoder.encode_full(frames)
    h_prefixes = encoder.encode_prefixes(frames, prefix_timesteps)

Cache keys embed task / trajectory_id / source / timestep / img_size /
num_frames / encoder_name to avoid cache confusion across configurations.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# vjepa2 path setup — must happen BEFORE any vjepa2 imports
# Defaults to the vendored third_party/vjepa2 copy; override with VJEPA2_PATH if needed.
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_VJEPA2_PATH = Path(__file__).resolve().parents[2] / "third_party" / "vjepa2"
_VJEPA2_PATH = os.environ.get("VJEPA2_PATH", str(_DEFAULT_VJEPA2_PATH))
import sys as _sys
if _VJEPA2_PATH not in _sys.path:
    _sys.path.insert(0, _VJEPA2_PATH)

import vjepa2.datasets.utils.video.volume_transforms as _volume_transforms
import vjepa2.datasets.utils.video.transforms as _video_transforms
from vjepa2.models.vision_transformer import vit_giant_xformers_rope

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
DEFAULT_NUM_FRAMES = 64
DEFAULT_IMG_SIZE = 384
ENCODER_NAME = "vjepa2_vitg_384"


# ──────────────────────────────────────────────────────────────────────────────
# WorldEncoder — thin wrapper around vit_giant_xformers_rope
# ──────────────────────────────────────────────────────────────────────────────


class _WorldEncoder:
    """V-JEPA2 video encoder with 64-frame sampling, resize/crop/normalize,
    and mean-pooled embedding output.

    This is a self-contained implementation — no dependency on openpi.models.
    """

    def __init__(
        self,
        model_path: str,
        img_size: int = DEFAULT_IMG_SIZE,
        device_id: int = 0,
        enable_fp16: bool = False,
        num_frames_for_embedding: int = DEFAULT_NUM_FRAMES,
    ) -> None:
        if num_frames_for_embedding <= 0:
            raise ValueError("num_frames_for_embedding must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("V-JEPA2 encoding requires a CUDA device")
        self.model_path = model_path
        self.img_size = img_size
        self.device = f"cuda:{device_id}"
        self.enable_fp16 = enable_fp16
        self.auto_cast_dtype = torch.float16 if enable_fp16 else torch.float32
        self.num_frames_for_embedding = num_frames_for_embedding

        print(
            f"WorldEncoder initialized with model_path={model_path}, "
            f"img_size={img_size}, device={self.device}, enable_fp16={enable_fp16}. "
            "It will take several minutes, please be patient..."
        )
        self.pt_video_transform, self.model_pt = self._create_model_instance()
        self.embedding_dim = self.model_pt.norm.bias.shape[0]
        print(f"Embedding model loaded successfully on {self.device}")

    # ── Build video transform ───────────────────────────────────────

    def _build_pt_video_transform(self):
        """Resize → CenterCrop → ClipToTensor → ImageNet Normalize."""
        short_side_size = int(256.0 / 224 * self.img_size)
        return _video_transforms.Compose([
            _video_transforms.Resize(short_side_size, interpolation="bilinear"),
            _video_transforms.CenterCrop(size=(self.img_size, self.img_size)),
            _volume_transforms.ClipToTensor(),
            _video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

    # ── Load pretrained weights ──────────────────────────────────────

    def _load_pretrained_vjepa_pt_weights(self, model, pretrained_weights):
        """Load pretrained weights, stripping 'module.' and 'backbone.' prefixes."""
        try:
            pretrained_dict = torch.load(
                pretrained_weights, weights_only=True, map_location=self.device
            )["encoder"]
            pretrained_dict = {
                k.replace("module.", "").replace("backbone.", ""): v
                for k, v in pretrained_dict.items()
            }
            msg = model.load_state_dict(pretrained_dict, strict=False)
            print(f"Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}")
        except Exception as e:
            print(f"Failed to load pretrained weights from {pretrained_weights}: {e}")
            raise

    # ── Create model instance ────────────────────────────────────────

    def _create_model_instance(self):
        """Create vit_giant_xformers_rope, load weights, set eval mode."""
        model_pt = vit_giant_xformers_rope(
            img_size=(self.img_size, self.img_size),
            num_frames=self.num_frames_for_embedding,
        )
        self._load_pretrained_vjepa_pt_weights(model_pt, self.model_path)
        model_pt.eval().to(self.device)
        pt_video_transform = self._build_pt_video_transform()
        return pt_video_transform, model_pt

    # ── Core encoding ────────────────────────────────────────────────

    def _extract_video_embedding(self, video_tensor: torch.Tensor) -> np.ndarray:
        """Run the model on a preprocessed video tensor [1, T, C, H, W] and
        return a mean-pooled float32 numpy embedding [D]."""
        with torch.inference_mode():
            x = self.pt_video_transform(video_tensor.to(self.device)).to(self.device).unsqueeze(0)
            autocast_context = (
                torch.amp.autocast("cuda", dtype=self.auto_cast_dtype)
                if self.enable_fp16
                else nullcontext()
            )
            with autocast_context:
                embedding = self.model_pt(x)
            return embedding.mean(dim=1).to(torch.float32).squeeze(0).cpu().numpy()

    def encode(self, frames: List[np.ndarray]) -> np.ndarray:
        """Encode a list of video frames into a single embedding vector.

        Sampling / padding:
          - frames >= 64:  uniform linspace sampling of 64 frames.
          - frames < 64:   pad with the last frame to reach 64 (NO cyclic padding).

        Args:
            frames: list of np.ndarray [H, W, C] uint8.

        Returns:
            embedding vector [D] float32.
        """
        if not frames:
            embedding = np.zeros((self.embedding_dim,), dtype=np.float32)
            return embedding

        num_frames = len(frames)
        if num_frames >= self.num_frames_for_embedding:
            indices = np.linspace(0, num_frames - 1, num=self.num_frames_for_embedding, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            indices = np.arange(num_frames)
            padded_indices = np.concatenate([
                indices,
                np.full(self.num_frames_for_embedding - len(indices), indices[-1]),
            ])
            sampled_frames = [frames[i] for i in padded_indices]

        # Stack frames: (T, H, W, C) → (T, C, H, W)
        video_tensor = torch.from_numpy(np.stack(sampled_frames)).permute(0, 3, 1, 2)
        return self._extract_video_embedding(video_tensor)


# ──────────────────────────────────────────────────────────────────────────────
# VJEPAEncoder — public API with disk cache
# ──────────────────────────────────────────────────────────────────────────────


class VJEPAEncoder:
    """Frozen V-JEPA2 encoder with optional cache I/O helpers.

    Usage::

        encoder = VJEPAEncoder(
            model_path="/path/to/vitg-384.pt",
            img_size=384,
            enable_fp16=True,
            cache_dir="/path/to/cache",
        )
        h_full = encoder.encode_full(frames)           # W(o_{0:T})
        h_prefixes = encoder.encode_prefixes(          # W(o_{0:t}) for each t
            frames, prefix_timesteps=[0, 5, 10, ...]
        )
    """

    def __init__(
        self,
        model_path: str,
        img_size: int = DEFAULT_IMG_SIZE,
        device_id: int = 0,
        enable_fp16: bool = True,
        num_frames_for_embedding: int = DEFAULT_NUM_FRAMES,
        cache_dir: str | None = None,
    ):
        self.model_path = model_path
        self.img_size = img_size
        self.num_frames = num_frames_for_embedding
        self.cache_dir = cache_dir
        self._encoder = None
        self._device_id = device_id
        self._enable_fp16 = enable_fp16

    @property
    def encoder(self) -> _WorldEncoder:
        """Lazy-load the WorldEncoder (GPU model)."""
        if self._encoder is None:
            print(f"Loading V-JEPA encoder from {self.model_path}...")
            t0 = time.time()
            self._encoder = _WorldEncoder(
                model_path=self.model_path,
                img_size=self.img_size,
                device_id=self._device_id,
                enable_fp16=self._enable_fp16,
                num_frames_for_embedding=self.num_frames,
            )
            print(f"  V-JEPA encoder loaded ({time.time() - t0:.1f}s)")
        return self._encoder

    # ── Public API ──────────────────────────────────────────────────────────

    def encode_full(self, frames: List[np.ndarray]) -> np.ndarray:
        """Encode a FULL trajectory:  W(o_{0:T}).

        Args:
            frames: list of [H, W, C] uint8 frames for the entire trajectory.

        Returns:
            embedding vector [D].
        """
        return self.encoder.encode(frames)

    def encode_prefixes(
        self,
        frames: List[np.ndarray],
        prefix_timesteps: List[int],
    ) -> np.ndarray:
        """Encode trajectory prefixes:  h_t = W(o_{0:t})  for each t in timesteps.

        Args:
            frames:           list of [H, W, C] uint8 frames (full trajectory).
            prefix_timesteps: list of 0-based frame indices to encode up to.

        Returns:
            embeddings [T_sampled, D].
        """
        embs = []
        for t in prefix_timesteps:
            prefix_frames = frames[: t + 1]
            h_t = self.encoder.encode(prefix_frames)
            embs.append(h_t)
        return np.array(embs)

    @property
    def embedding_dim(self) -> int:
        return self.encoder.embedding_dim

    # ── Cache key builder ───────────────────────────────────────────────────

    def _make_cache_key(self, task_slug: str, prefix: str, *,
                        trajectory_id: str = "",
                        source: str = "",
                        timestep: Optional[int] = None,
                        ) -> str:
        """Build a cache key that includes encoder metadata to avoid collisions.

        Format:  {prefix}_{trajectory_id}_{source}_t{timestep}__{encoder_meta}
        The encoder meta suffix encodes img_size, num_frames, and encoder name.
        Key fields: task, trajectory_id, source, timestep, img_size, num_frames, encoder_name.
        """
        meta = f"enc={ENCODER_NAME}_sz={self.img_size}_nf={self.num_frames}"
        parts = [prefix]
        if trajectory_id:
            parts.append(f"tid={trajectory_id}")
        if source:
            parts.append(f"src={source}")
        if timestep is not None:
            parts.append(f"t={timestep:04d}")
        parts.append(meta)
        return "__".join(parts)

    # ── Cache I/O ───────────────────────────────────────────────────────────

    def _cache_path_dir(self, task_slug: str) -> str:
        if self.cache_dir is None:
            return ""
        d = os.path.join(self.cache_dir, task_slug)
        os.makedirs(d, exist_ok=True)
        return d

    def load_cache(self, task_slug: str, key: str):
        """Load cached data; returns None if not found."""
        if self.cache_dir is None:
            return None
        d = self._cache_path_dir(task_slug)
        path = os.path.join(d, key)
        npy_path = path + ".npy"
        if os.path.exists(npy_path):
            return np.load(npy_path)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def save_cache(self, task_slug: str, key: str, data):
        """Save data to cache (np.ndarray → .npy, else → pickle)."""
        if self.cache_dir is None:
            return
        d = self._cache_path_dir(task_slug)
        path = os.path.join(d, key)
        if isinstance(data, np.ndarray):
            np.save(path, data)
        else:
            with open(path, "wb") as f:
                pickle.dump(data, f)
