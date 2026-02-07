"""
Pistar LeRobot 数据处理流程脚本（内存优化版本）

优化要点：
1. 使用流式处理，不将所有数据加载到内存
2. 采用两遍处理：
   - Pass 1: 计算 epsilon（只保存 advantages 轻量数据）
   - Pass 2: 流式写入数据集
3. 相比原版大幅节省内存（不在内存中保存完整 episode 数据）

处理流程：
- Pass 1: 计算所有 episodes 的 advantages，统计 epsilon（不保存图像数据）
- Pass 2: 重新加载数据，边读边写，使用 epsilon 计算 adv_ind

注意：
- 输入数据集不需要包含 reward, value, adv, epsilon, adv_ind 等 feature
- Value 取值范围: [-1.0, 0.0]
- 所有 value 使用 --default_value 参数设置（默认 0.0）
- value_model_path 功能待实现
- 可通过 --default_adv_ind 跳过 adv 计算，直接设置所有 adv_ind
- 需要提供 --original_reward_key 指定原始 reward 的 feature 名称

Usage:
# 从 LeRobot 数据集计算 adv 和 adv_ind（内存优化）
python examples/libero/pistar_lerobot_data_processing_optimized.py \
    --input_repo_id ybpy/libero \
    --output_repo_id ybpy/libero_pistar \
    --original_reward_key reward \
    --default_value 0.0 \
    --n_steps 10

# 跳过 adv 计算，直接设置 adv_ind
python examples/libero/pistar_lerobot_data_processing_optimized.py \
    --input_repo_id ybpy/libero \
    --output_repo_id ybpy/libero_pistar \
    --original_reward_key reward \
    --default_value 0.0 \
    --default_adv_ind positive

# 使用 value 模型（待实现）
python examples/libero/pistar_lerobot_data_processing_optimized.py \
    --input_repo_id ybpy/libero \
    --output_repo_id ybpy/libero_pistar \
    --original_reward_key reward \
    --value_model_path /path/to/value_model.pth \
    --n_steps 10

# unbuffered 输出日志（实时查看）
python -u examples/libero/pistar_lerobot_data_processing_optimized.py ... 
"""

import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro


def transform_reward(original_reward: float, is_terminal: bool, is_last: bool, episode_length: int) -> float:
    """
    转换 reward 的规则：
    1. 如果 is_terminal 或 is_last 为 True:
       - 原始 reward = 1.0 → 0.0
       - 原始 reward = 0.0 → -1.0
    2. 如果 is_terminal 和 is_last 都为 False (中间步骤):
       - reward = -1 / episode_length
    """
    if is_terminal or is_last:
        # 至少一个为 True
        if original_reward == 1.0:
            return 0.0
        else:  # original_reward == 0.0
            return -1.0
    else:
        # 都为 False (中间步骤): reward = -1 / episode_length
        return -1.0 / episode_length


def compute_value_placeholder(step_data: dict) -> float:
    """
    占位函数：计算 value
    TODO: 替换为实际的模型推理
    
    Args:
        step_data: 包含 observation, action 等的字典
    
    Returns:
        value: 预测的状态价值 (范围: [-1.0, 0.0])
    """
    # 当前使用 0.0 作为默认值
    # Value 的取值范围预定在 [-1.0, 0.0] 之间
    return 0.0


def compute_advantage(
    rewards: np.ndarray,
    values: np.ndarray,
    n_steps: int,
    gamma: float = 1.0
) -> np.ndarray:
    """
    计算 advantage
    
    adv[t] = sum(rewards[t:t+N]) + value[t+N] - value[t]
    
    注意：对于 episode 的最后 N 个 step：
    - 窗口会截断到 episode 结束
    - 使用 episode 最后一个 step 的 value 作为 bootstrap
    - 例如：t=95, T=100, N=10 时，actual_steps=5
      adv[95] = sum(rewards[95:100]) + gamma^5 * value[99] - value[95]
    
    Args:
        rewards: shape (T,) 的 reward 数组
        values: shape (T,) 的 value 数组
        n_steps: N-step 窗口大小
        gamma: 折扣因子 (默认 1.0)
    
    Returns:
        advantages: shape (T,) 的 advantage 数组
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    
    for t in range(T):
        # 计算 N-step return (窗口会自动截断到 episode 结束)
        n_step_return = 0.0
        actual_steps = min(n_steps, T - t)  # 实际能看到的步数
        
        for i in range(actual_steps):
            n_step_return += (gamma ** i) * rewards[t + i]
        
        # 添加 bootstrap value
        if t + n_steps < T:
            # 正常情况：加上 N 步后的 value
            n_step_return += (gamma ** n_steps) * values[t + n_steps]
        else:
            # 最后 N 个 step：加上 episode 最后一个 step 的 value
            n_step_return += (gamma ** actual_steps) * values[T - 1]
        
        # advantage = n_step_return - value[t]
        advantages[t] = n_step_return - values[t]
    
    return advantages


def main(
    input_repo_id: str,
    output_repo_id: str,
    *,
    original_reward_key: Optional[str] = None,
    n_steps: int = 10,
    value_model_path: str | None = None,
    default_value: float = 0.0,
    default_adv_ind: str | None = None,
    epsilon_percentile: float = 70.0,
    push_to_hub: bool = False,
):
    """
    Pistar LeRobot 数据处理和转换（内存优化版本）
    
    Args:
        input_repo_id: 输入 LeRobot 数据集 ID (例如: ybpy/libero)
        output_repo_id: 输出 LeRobot 数据集 ID (例如: ybpy/libero_pistar)
        original_reward_key: 原始 reward 的 feature 名称 (如果输入数据集包含 reward)
                            如果为 None，将假设所有 episode 成功 (最后一步 reward=1.0)
        n_steps: N-step advantage 计算的窗口大小
        value_model_path: Value 模型路径 (可选，待实现)
        default_value: 默认 value 值 (范围 [-1.0, 0.0]，默认 0.0)
        default_adv_ind: 默认 adv_ind 值 ("positive" 或 "negative")，
                        如果设置，将跳过 adv 和 epsilon 计算
        epsilon_percentile: epsilon 的分位数 (默认 70.0)
        push_to_hub: 是否推送到 HuggingFace Hub
    """
    
    print("=" * 80)
    print("🚀 Pistar LeRobot 数据处理流程（内存优化版本）")
    print("=" * 80)
    print(f"Input repo: {input_repo_id}")
    print(f"Output repo: {output_repo_id}")
    print(f"N-step window: {n_steps}")
    print(f"Default value: {default_value}")
    if default_adv_ind:
        print(f"Default adv_ind: {default_adv_ind} (跳过 adv 计算)")
    else:
        print(f"Epsilon percentile: {epsilon_percentile}%")
    
    # 检查 value 模型
    if value_model_path:
        print(f"⚠️  Value model loading not yet implemented")
        print(f"    Using default value: {default_value}")
    else:
        print(f"📌 Using default value: {default_value}")
    
    # ========================================================================
    # Pass 1: 计算 epsilon（只保存轻量级数据：rewards/values/task）
    # ========================================================================
    task_epsilon = {}
    
    if not default_adv_ind:
        print("\n" + "=" * 80)
        print("📈 Pass 1: 计算 advantages 并统计 epsilon（轻量级扫描）")
        print("=" * 80)
        
        # 加载输入数据集（只用于轻量级扫描）
        print(f"Loading input dataset: {input_repo_id}")
        input_dataset = LeRobotDataset(input_repo_id)
        
        print(f"Dataset info:")
        print(f"  Total frames: {len(input_dataset)}")
        print(f"  Total episodes: {input_dataset.num_episodes}")
        print(f"  Features: {list(input_dataset.features.keys())}")
        
        # 按 task 收集所有 advantages
        task_advantages = defaultdict(list)
        
        # 按 episode 迭代
        for episode_idx in range(input_dataset.num_episodes):
            # 获取该 episode 的所有帧索引
            episode_indices = input_dataset.episode_data_index["from"][episode_idx].item()
            next_episode_start = (
                input_dataset.episode_data_index["from"][episode_idx + 1].item()
                if episode_idx + 1 < input_dataset.num_episodes
                else len(input_dataset)
            )
            episode_length = next_episode_start - episode_indices
            
            # 只提取计算所需的最小数据（不保存图像）
            rewards = []
            values = []
            task = None
            
            for step_idx in range(episode_length):
                frame_idx = episode_indices + step_idx
                frame = input_dataset[frame_idx]
                
                # 获取 task (假设存在 task 字段)
                if step_idx == 0:
                    task = frame.get('task', f'episode_{episode_idx}')
                
                # 获取原始 reward
                if original_reward_key and original_reward_key in frame:
                    original_reward = float(frame[original_reward_key])
                else:
                    # 如果没有原始 reward，假设成功的 episode
                    # 最后一步 reward=1.0，其他步骤 reward=0.0
                    original_reward = 1.0 if step_idx == episode_length - 1 else 0.0
                
                # 判断是否为最后一步
                is_last = (step_idx == episode_length - 1)
                # 判断是否为 terminal (基于 original_reward)
                is_terminal = (original_reward == 1.0) and is_last
                
                # 转换 reward
                transformed_reward = transform_reward(
                    original_reward, is_terminal, is_last, episode_length
                )
                
                # 获取或计算 value
                if value_model_path:
                    # TODO: 使用实际模型计算 value
                    value = compute_value_placeholder(frame)
                else:
                    value = default_value
                
                rewards.append(transformed_reward)
                values.append(value)
            
            # 计算 advantages
            rewards_array = np.array(rewards, dtype=np.float32)
            values_array = np.array(values, dtype=np.float32)
            advantages = compute_advantage(rewards_array, values_array, n_steps)
            
            # 收集到 task_advantages 中
            task_advantages[task].extend(advantages.tolist())
            
            if (episode_idx + 1) % 50 == 0:
                print(f"   Processed {episode_idx + 1}/{input_dataset.num_episodes} episodes")
        
        # 计算每个 task 的 epsilon (基于 advantages)
        print(f"\n📊 Computing epsilon for {len(task_advantages)} unique tasks")
        for task, advantages in task_advantages.items():
            epsilon = np.percentile(advantages, epsilon_percentile)
            task_epsilon[task] = epsilon
            task_str = str(task)[:50] if task else "unknown"
            print(f"Task: {task_str}...")
            print(f"  Advantages count: {len(advantages)}")
            print(f"  Epsilon ({epsilon_percentile}%): {epsilon:.4f}")
        
        print(f"\n✅ Pass 1 complete: {input_dataset.num_episodes} episodes scanned")
    else:
        print("\n" + "=" * 80)
        print(f"⏭️  Pass 1: 跳过 (使用默认 adv_ind: {default_adv_ind})")
        print("=" * 80)
    
    # ========================================================================
    # Pass 2: 流式写入数据集（边读边写，不在内存中积累）
    # ========================================================================
    print("\n" + "=" * 80)
    print("💾 Pass 2: 流式写入数据集（内存友好模式）")
    print("=" * 80)
    
    # 重新加载输入数据集
    print(f"Reloading input dataset: {input_repo_id}")
    input_dataset = LeRobotDataset(input_repo_id)
    
    # 清理已存在的数据集
    output_path = HF_LEROBOT_HOME / output_repo_id
    if output_path.exists():
        print(f"🗑️  Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # 构建新的 features（保留原始 features + 添加新 features）
    new_features = dict(input_dataset.features)
    
    # 添加 PiStar 相关的 features
    new_features.update({
        "reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["reward"],
        },
        "value": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["value"],
        },
        "adv": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["adv"],
        },
        "epsilon": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["epsilon"],
        },
        "adv_ind": {
            "dtype": "string",
            "shape": (1,),
            "names": ["adv_ind"],
        },
    })
    
    # 创建 LeRobot 数据集
    # 从输入数据集的元数据中获取 robot_type 和 fps (从 info.json)
    robot_type = input_dataset.meta.robot_type
    fps = input_dataset.meta.fps
    
    output_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        robot_type=robot_type,
        fps=fps,
        features=new_features,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # 重新扫描数据集，边读边写
    total_steps = 0
    
    for episode_idx in range(input_dataset.num_episodes):
        # 获取该 episode 的所有帧索引
        episode_indices = input_dataset.episode_data_index["from"][episode_idx].item()
        next_episode_start = (
            input_dataset.episode_data_index["from"][episode_idx + 1].item()
            if episode_idx + 1 < input_dataset.num_episodes
            else len(input_dataset)
        )
        episode_length = next_episode_start - episode_indices
        
        # 计算本 episode 的 advantages 和 epsilon（轻量级操作）
        if default_adv_ind:
            # 使用默认 adv_ind，跳过 adv 计算
            advantages = np.zeros(episode_length, dtype=np.float32)
            epsilon = 0.0
            task = None
        else:
            # 重新计算 advantages（轻量级操作）
            rewards = []
            values = []
            task = None
            
            for step_idx in range(episode_length):
                frame_idx = episode_indices + step_idx
                frame = input_dataset[frame_idx]
                
                # 获取 task
                if step_idx == 0:
                    task = frame.get('task', f'episode_{episode_idx}')
                
                # 获取原始 reward
                if original_reward_key and original_reward_key in frame:
                    original_reward = float(frame[original_reward_key])
                else:
                    original_reward = 1.0 if step_idx == episode_length - 1 else 0.0
                
                is_last = (step_idx == episode_length - 1)
                is_terminal = (original_reward == 1.0) and is_last
                
                transformed_reward = transform_reward(
                    original_reward, is_terminal, is_last, episode_length
                )
                
                if value_model_path:
                    value = compute_value_placeholder(frame)
                else:
                    value = default_value
                
                rewards.append(transformed_reward)
                values.append(value)
            
            rewards_array = np.array(rewards, dtype=np.float32)
            values_array = np.array(values, dtype=np.float32)
            advantages = compute_advantage(rewards_array, values_array, n_steps)
            epsilon = task_epsilon[task]
        
        # 写入每个 step
        for step_idx in range(episode_length):
            frame_idx = episode_indices + step_idx
            frame = input_dataset[frame_idx]
            
            # 获取 task（如果之前没有获取）
            if task is None:
                task = frame.get('task', f'episode_{episode_idx}')
            
            # 计算 transformed_reward 和 value（如果使用默认 adv_ind，需要重新计算）
            if default_adv_ind:
                if original_reward_key and original_reward_key in frame:
                    original_reward = float(frame[original_reward_key])
                else:
                    original_reward = 1.0 if step_idx == episode_length - 1 else 0.0
                
                is_last = (step_idx == episode_length - 1)
                is_terminal = (original_reward == 1.0) and is_last
                
                transformed_reward = transform_reward(
                    original_reward, is_terminal, is_last, episode_length
                )
                
                if value_model_path:
                    value = compute_value_placeholder(frame)
                else:
                    value = default_value
            else:
                # 已经在前面计算过了
                transformed_reward = rewards[step_idx]
                value = values[step_idx]
            
            adv = advantages[step_idx]
            
            if default_adv_ind:
                adv_ind = default_adv_ind
            else:
                adv_ind = "positive" if adv > epsilon else "negative"
            
            # 构建新的 frame（保留原始数据 + 添加新字段）
            # 需要移除 LeRobot 自动生成的系统字段
            system_fields = {'index', 'task_index', 'episode_index', 'frame_index', 'timestamp'}
            new_frame = {k: v for k, v in frame.items() if k not in system_fields}
            
            # 添加 task 和新的 PiStar 字段
            new_frame.update({
                "task": task,
                "reward": np.array([transformed_reward], dtype=np.float32),
                "value": np.array([value], dtype=np.float32),
                "adv": np.array([adv], dtype=np.float32),
                "epsilon": np.array([epsilon], dtype=np.float32),
                "adv_ind": adv_ind,
            })
            
            output_dataset.add_frame(new_frame)
            total_steps += 1
        
        output_dataset.save_episode()
        
        if (episode_idx + 1) % 50 == 0:
            print(f"   Written {episode_idx + 1}/{input_dataset.num_episodes} episodes")
    
    print(f"\n✅ Pass 2 complete!")
    print(f"   Total episodes: {input_dataset.num_episodes}")
    print(f"   Total steps: {total_steps}")
    print(f"   Output path: {output_path}")
    
    # 推送到 Hub
    if push_to_hub:
        print(f"\n📤 Pushing to Hugging Face Hub...")
        output_dataset.push_to_hub(
            tags=["pistar", "value", "advantage"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"✅ Successfully pushed to Hub!")
    
    print("\n" + "=" * 80)
    print("🎉 All processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    tyro.cli(main)
