from __future__ import annotations

import multiprocessing
from collections import OrderedDict
from collections.abc import Iterator
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import torch

from openpi.models import model as _model
from openpi.shared import console

logger = logging.getLogger("openpi")

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp


class ValueDataset(TorchDataset):
    """LeRobot 价值函数数据集。"""

    def __init__(
        self,
        data_dir: str,
        max_token_len: int = 48,
        image_size: tuple[int, int] = (224, 224),
    ):
        """
        Args:
            data_dir: LeRobot 数据集路径
            max_token_len: 最大 token 长度
            image_size: 图像大小
        """
        self.data_dir = Path(data_dir)
        self.max_token_len = max_token_len
        self.image_size = image_size
        self.preload_parquet = _env_flag("OPENPI_VALUE_PRELOAD_PARQUET", default=False)
        self._zero_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.float32)

        # 加载任务文本信息
        self._load_task_texts()

        parquet_dir = self.data_dir / "data"
        self.parquet_files = sorted(parquet_dir.rglob("*.parquet"))

        if not self.parquet_files:
            raise ValueError(f"找不到 parquet 文件: {parquet_dir}")

        self.episode_lengths = []
        self.episode_offsets = [0]
        self._episode_cache: "OrderedDict[int, pd.DataFrame]" = OrderedDict()

        if self.preload_parquet:
            # 预加载所有parquet数据到内存，避免重复磁盘I/O
            logger.info(console.info("预加载所有parquet数据到内存..."))
            self.episode_data = []

            for pf in self.parquet_files:
                df = pd.read_parquet(pf)
                self.episode_data.append(df)  # 缓存到内存
                self.episode_lengths.append(len(df))
                self.episode_offsets.append(self.episode_offsets[-1] + len(df))

            self.total_frames = self.episode_offsets[-1]
            logger.info(console.info(f"加载 {len(self.parquet_files)} 个 episodes, 共 {self.total_frames} 帧"))
        else:
            cache_env = os.getenv("OPENPI_VALUE_PARQUET_CACHE_EPISODES")
            if cache_env:
                try:
                    self.max_cached_episodes = max(0, int(cache_env))
                except ValueError:
                    logger.warning(
                        console.warn(
                            f"环境变量 OPENPI_VALUE_PARQUET_CACHE_EPISODES='{cache_env}' 无效，"
                            "使用默认值 2"
                        )
                    )
                    self.max_cached_episodes = 2
            else:
                self.max_cached_episodes = 2

            logger.info(
                console.info(f"按需加载parquet数据，缓存 {self.max_cached_episodes} 个 episode")
            )
            for pf in self.parquet_files:
                num_rows = _get_parquet_num_rows(pf)
                self.episode_lengths.append(num_rows)
                self.episode_offsets.append(self.episode_offsets[-1] + num_rows)

            self.total_frames = self.episode_offsets[-1]
            logger.info(console.info(f"索引 {len(self.parquet_files)} 个 episodes, 共 {self.total_frames} 帧"))

    def _load_task_texts(self):
        """加载任务文本信息。"""
        import json

        tasks_file = self.data_dir / "meta" / "tasks.jsonl"
        if not tasks_file.exists():
            raise ValueError(f"找不到任务文件: {tasks_file}")

        self.task_texts = {}
        with open(tasks_file, "r") as f:
            for line in f:
                task_data = json.loads(line.strip())
                task_index = task_data["task_index"]
                task_text = task_data["task"]
                self.task_texts[task_index] = task_text

        logger.info(console.ok(f"加载了 {len(self.task_texts)} 个任务文本"))

        # 不在这里初始化 tokenizer，延迟到实际使用时
        self.tokenizer = None
        self.tokenizer_path = "/public/home/chenyuyao1/.cache/openpi/big_vision/paligemma_tokenizer.model"
        self._tokenized_task_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _init_gemma3_tokenizer(self):
        """初始化 Gemma3 tokenizer（使用本地文件）"""
        # 使用 /home/wang/data/Project1/openpi/gemma 中的 Gemma3Tokenizer
        import os
        import sys

        gemma_path = os.path.join(os.path.dirname(__file__), "../../../gemma")
        sys.path.insert(0, gemma_path)

        from gemma.gm.text._tokenizer import Gemma3Tokenizer

        # 使用本地的 tokenizer.model 文件
        local_tokenizer_path = "/public/home/chenyuyao1/.cache/openpi/big_vision/paligemma_tokenizer.model"
        self.tokenizer = Gemma3Tokenizer(path=local_tokenizer_path)
        logger.info(console.ok(f"成功加载本地 Gemma3Tokenizer: {local_tokenizer_path}"))

    def _tokenize_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """使用 Gemma3 tokenizer 进行文本tokenization（按照 Pi0 方式）"""
        # 延迟初始化 tokenizer（在每个 worker 进程中独立初始化）
        if self.tokenizer is None:
            self._init_gemma3_tokenizer()

        # 添加Value:后缀，明确这是价值估计任务
        text_with_suffix = f"{text}\nValue:"

        # 使用 Gemma3Tokenizer（和 Pi0 使用 PaliGemma tokenizer 一样的方式）
        tokens = self.tokenizer.encode(text_with_suffix, add_bos=True, add_eos=False)

        # 截断或填充到固定长度
        if len(tokens) > self.max_token_len:
            tokens = tokens[:self.max_token_len]
        else:
            tokens.extend([0] * (self.max_token_len - len(tokens)))  # pad with 0

        tokens = np.array(tokens, dtype=np.int32)
        mask = (tokens != 0).astype(bool)  # 非0的都是有效token

        return tokens, mask

    def _get_tokenized_prompt(self, task_index: int, task_text: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._tokenized_task_cache.get(task_index)
        if cached is not None:
            return cached
        tokens, mask = self._tokenize_text(task_text)
        self._tokenized_task_cache[task_index] = (tokens, mask)
        return tokens, mask

    def __len__(self) -> int:
        return self.total_frames

    def _find_episode(self, idx: int) -> tuple[int, int]:
        """找到 idx 对应的 episode 和帧索引。"""
        for i, offset in enumerate(self.episode_offsets[1:], 1):
            if idx < offset:
                episode_idx = i - 1
                frame_idx = idx - self.episode_offsets[episode_idx]
                return episode_idx, frame_idx
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int) -> dict:
        episode_idx, frame_idx = self._find_episode(idx)

        if self.preload_parquet:
            row = self.episode_data[episode_idx].iloc[frame_idx]
        else:
            row = self._get_episode_df(episode_idx).iloc[frame_idx]

        image = self._load_image(row["image"])

        wrist_raw = row.get("wrist_image", None)
        has_wrist = wrist_raw is not None
        if has_wrist and isinstance(wrist_raw, float) and np.isnan(wrist_raw):
            has_wrist = False
        if has_wrist:
            wrist_image = self._load_image(wrist_raw)
        else:
            wrist_image = self._zero_image

        # 获取任务文本
        task_index = int(row["task_index"])
        task_text = self.task_texts.get(task_index, "unknown task")
        tokenized_prompt, prompt_mask = self._get_tokenized_prompt(task_index, task_text)

        # Value model 不需要机器人状态，只需要图像
        # state = np.array(row["state"], dtype=np.float32)

        if "value" not in row:
            raise ValueError("数据中没有 value 字段，请先运行 add_value_labels.py")
        value = np.float32(row["value"])

        result = {
            "image": {
                "base_0_rgb": image,
            },
            "image_mask": {
                "base_0_rgb": True,
            },
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": prompt_mask,
            "state": np.zeros(8, dtype=np.float32),  # placeholder required by Observation.from_dict
            "value": value,
            "episode_index": np.int64(row["episode_index"]),
            "frame_index": np.int64(row["frame_index"]),
            "index": np.int64(row["index"]),
        }

        result["image"]["wrist_0_rgb"] = wrist_image
        result["image_mask"]["wrist_0_rgb"] = has_wrist

        return result

    def _load_image(self, image_data) -> np.ndarray:
        """加载图像数据。"""
        import io

        from PIL import Image

        if isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]
        elif isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            raise ValueError(f"未知的图像格式: {type(image_data)}")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)

        image_array = np.array(image, dtype=np.float32)
        return image_array / 255.0 * 2.0 - 1.0

    def _get_episode_df(self, episode_idx: int) -> pd.DataFrame:
        if episode_idx in self._episode_cache:
            df = self._episode_cache.pop(episode_idx)
            self._episode_cache[episode_idx] = df
            return df

        df = pd.read_parquet(self.parquet_files[episode_idx])
        if self.max_cached_episodes > 0:
            self._episode_cache[episode_idx] = df
            while len(self._episode_cache) > self.max_cached_episodes:
                self._episode_cache.popitem(last=False)
        return df


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _get_parquet_num_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        df = pd.read_parquet(path)
        return len(df)


def collate_fn(batch: list[dict]) -> dict:
    """将 batch 合并为 numpy 数组。"""
    result = {}

    result["image"] = {key: np.stack([item["image"][key] for item in batch], axis=0) for key in batch[0]["image"]}
    result["image_mask"] = {
        key: np.array([item["image_mask"][key] for item in batch], dtype=bool) for key in batch[0]["image_mask"]
    }

    # 添加文本数据处理
    result["tokenized_prompt"] = np.stack([item["tokenized_prompt"] for item in batch], axis=0)
    result["tokenized_prompt_mask"] = np.array([item["tokenized_prompt_mask"] for item in batch], dtype=bool)

    # Observation类需要state字段，但Value Model不使用，提供空占位符
    batch_size = len(batch)
    result["state"] = np.zeros((batch_size, 1), dtype=np.float32)  # 最小占位符

    result["value"] = np.array([item["value"] for item in batch], dtype=np.float32)

    return result


def worker_init_fn_impl(worker_id: int, seed: int):
    """Worker进程初始化函数，π0风格。"""
    # 避免worker进程中的JAX预分配GPU内存
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    # 设置每个worker的独立随机种子
    np.random.seed(seed + worker_id)
    import random
    random.seed(seed + worker_id)


class ValueDataLoader:
    """价值函数数据加载器。"""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        num_workers: int = 4,
        max_token_len: int = 48,
        seed: int = 42,
        sharding: "jax.sharding.Sharding | None" = None,
    ):
        self.dataset = ValueDataset(data_dir, max_token_len=max_token_len)
        self.seed = seed

        self._debug_logged = False
        self._default_sharding = None

        # π0风格的多进程优化
        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")  # 更稳定的上下文

        # π0风格的随机数生成器
        generator = torch.Generator()
        generator.manual_seed(seed)

        import functools

        prefetch_factor = None
        if num_workers > 0:
            prefetch_env = os.getenv("OPENPI_VALUE_PREFETCH_FACTOR")
            if prefetch_env:
                try:
                    prefetch_factor = max(1, int(prefetch_env))
                except ValueError:
                    logger.warning(
                        console.warn(f"OPENPI_VALUE_PREFETCH_FACTOR='{prefetch_env}' 无效，使用默认值 4")
                    )
                    prefetch_factor = 4
            else:
                prefetch_factor = 4

        self._torch_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,       # π0优化
            persistent_workers=num_workers > 0,       # π0优化：保持worker存活
            collate_fn=collate_fn,
            worker_init_fn=functools.partial(worker_init_fn_impl, seed=seed), # π0优化：worker初始化
            drop_last=True,
            pin_memory=False,                         # 避免CUDA警告
            generator=generator,                      # π0优化：固定种子
            prefetch_factor=prefetch_factor,  # 修复：只在多进程时使用prefetch_factor
        )

        # 延迟初始化sharding，避免多进程序列化问题
        self._sharding = sharding

    def __len__(self) -> int:
        return len(self._torch_loader)

    def set_sharding(self, sharding: "jax.sharding.Sharding"):
        """设置 sharding（在初始化后）。"""
        self._sharding = sharding

    def __iter__(self) -> Iterator[tuple[_model.Observation, jnp.ndarray]]:
        import jax

        for batch in self._torch_loader:
            if not self._debug_logged:
                from torch.utils.data import get_worker_info
                logger.info("DataLoader batch ready")
                logger.info(f"worker_info: {get_worker_info()}")

                def _summarize(obj):
                    if isinstance(obj, dict):
                        return {k: _summarize(v) for k, v in obj.items()}
                    return {
                        "type": type(obj).__name__,
                        "dtype": getattr(obj, "dtype", None),
                        "shape": getattr(obj, "shape", None),
                    }

                logger.info(f"batch_summary: {_summarize(batch)}")
                logger.info("Before jax.make_array_from_process_local_data")
            # 如果没有设置sharding，使用默认的单设备sharding
            if self._sharding is None:
                # 创建临时的默认sharding（在__iter__中，不会被序列化）
                if self._default_sharding is None:
                    self._default_sharding = jax.sharding.NamedSharding(
                        jax.sharding.Mesh(jax.devices(), ("B",)),
                        jax.sharding.PartitionSpec("B"),
                    )
                sharding_to_use = self._default_sharding
            else:
                sharding_to_use = self._sharding

            batch_jax = jax.tree.map(lambda x: jax.make_array_from_process_local_data(sharding_to_use, x), batch)

            if not self._debug_logged:
                logger.info("After jax.make_array_from_process_local_data")
                self._debug_logged = True

            observation = _model.Observation.from_dict(batch_jax)
            value = batch_jax["value"]

            yield observation, value
