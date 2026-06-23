"""Reward and value label computation from progress p_t.

Given progress values p_t ∈ (0, 1] for each trajectory prefix:

  r_t = p_t - p_{t+1}          (Bellman-consistent temporal-difference reward)
    - Success terminal:  r_T = 0
    - Failed terminal:   r_T = p_T - 1   (negative penalty for failure)

  value_label_t = p_t - 1     (value label for PiStar06)
    - Range: (-1, 0]  (since p_t ∈ (0, 1])
    - Success terminal:  value_label_T = 0  (p_T = 1)
    - Failed terminal:   value_label_T < 0  (p_T < 1)

The value_label format matches PiStar06 expectations: negative for
incomplete states, zero for successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ValueLabelResult:
    """Summary statistics for value labels across a set of trajectories."""

    success_value_label_mean: float = 0.0
    success_value_label_min: float = 0.0
    success_value_label_max: float = 0.0
    failed_value_label_mean: float = 0.0
    failed_value_label_min: float = 0.0
    failed_value_label_max: float = 0.0
    success_reward_mean: float = 0.0
    failed_reward_mean: float = 0.0
    all_value_labels: List[float] = field(default_factory=list)
    all_rewards: List[float] = field(default_factory=list)


def compute_rewards_and_values(
    trajectories: list,      # list of Trajectory with p_values already populated
) -> ValueLabelResult:
    """Compute r_t and value_label_t from p_t for all trajectories.

    Mutates trajectories in-place, setting .rewards and .value_labels fields.

    Args:
        trajectories: list of Trajectory with p_values populated.

    Returns:
        ValueLabelResult with aggregate statistics.
    """
    result = ValueLabelResult()

    success_value_labels = []
    failed_value_labels = []
    success_rewards = []
    failed_rewards = []

    for traj in trajectories:
        p_vals = traj.p_values
        if p_vals is None or len(p_vals) == 0:
            traj.rewards = np.array([], dtype=np.float32)
            traj.value_labels = np.array([], dtype=np.float32)
            continue

        n = len(p_vals)

        # ── value_label = p_t - 1 ──
        v_labels = p_vals - 1.0  # range: (-1, 0]
        traj.value_labels = v_labels.astype(np.float32)

        # ── reward: r_t = p_t - p_{t+1} ──
        r_vals = np.zeros(n, dtype=np.float32)
        if n > 1:
            r_vals[:-1] = p_vals[:-1] - p_vals[1:]
        # Terminal reward
        if traj.success:
            r_vals[-1] = 0.0  # success terminal: no penalty
        else:
            r_vals[-1] = p_vals[-1] - 1.0  # failed terminal: negative penalty

        traj.rewards = r_vals

        # Collect for statistics
        for v in v_labels:
            result.all_value_labels.append(float(v))
        for r in r_vals:
            result.all_rewards.append(float(r))

        if traj.success:
            success_value_labels.extend([float(v) for v in v_labels])
            success_rewards.extend([float(r) for r in r_vals])
        else:
            failed_value_labels.extend([float(v) for v in v_labels])
            failed_rewards.extend([float(r) for r in r_vals])

    # Compute summary statistics
    if success_value_labels:
        result.success_value_label_mean = float(np.mean(success_value_labels))
        result.success_value_label_min = float(np.min(success_value_labels))
        result.success_value_label_max = float(np.max(success_value_labels))
    if failed_value_labels:
        result.failed_value_label_mean = float(np.mean(failed_value_labels))
        result.failed_value_label_min = float(np.min(failed_value_labels))
        result.failed_value_label_max = float(np.max(failed_value_labels))
    if success_rewards:
        result.success_reward_mean = float(np.mean(success_rewards))
    if failed_rewards:
        result.failed_reward_mean = float(np.mean(failed_rewards))

    return result


def validate_value_labels(trajectories: list) -> dict:
    """Validate finite ranges, terminal rules, and reward/value consistency."""
    all_v = []
    terminal_success = []
    terminal_failed = []
    issues = []
    max_identity_error = 0.0

    for traj_index, traj in enumerate(trajectories):
        if traj.value_labels is None or len(traj.value_labels) == 0:
            continue

        values = np.asarray(traj.value_labels, dtype=np.float32)
        rewards = np.asarray(traj.rewards, dtype=np.float32)
        if len(rewards) != len(values):
            issues.append(f"trajectory {traj_index}: reward/value length mismatch")
            continue
        if not np.all(np.isfinite(values)):
            issues.append(f"trajectory {traj_index}: non-finite value labels")
        if not np.all(np.isfinite(rewards)):
            issues.append(f"trajectory {traj_index}: non-finite rewards")

        all_v.extend(values.astype(float).tolist())
        if len(values) > 1:
            identity_error = float(np.max(np.abs(rewards[:-1] - (values[:-1] - values[1:]))))
            max_identity_error = max(max_identity_error, identity_error)
            if identity_error > 1e-5:
                issues.append(f"trajectory {traj_index}: reward/value identity error={identity_error:.3g}")

        terminal_value = float(values[-1])
        terminal_reward = float(rewards[-1])
        if traj.success:
            terminal_success.append(terminal_value)
            if not np.isclose(terminal_value, 0.0, atol=1e-5):
                issues.append(f"trajectory {traj_index}: success terminal value={terminal_value:.4g}")
            if not np.isclose(terminal_reward, 0.0, atol=1e-5):
                issues.append(f"trajectory {traj_index}: success terminal reward={terminal_reward:.4g}")
        else:
            terminal_failed.append(terminal_value)
            if terminal_value >= -1e-6:
                issues.append(f"trajectory {traj_index}: failed terminal value must be negative")
            if not np.isclose(terminal_reward, terminal_value, atol=1e-5):
                issues.append(f"trajectory {traj_index}: failed terminal reward != terminal value")

    if not all_v:
        return {"valid": False, "reason": "no value labels", "issues": ["no value labels"]}

    values = np.asarray(all_v, dtype=np.float32)
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    mean_v = float(np.mean(values))
    if max_v > 1e-5:
        issues.append(f"max={max_v:.4f} > 0")
    if min_v < -1.00001:
        issues.append(f"min={min_v:.4f} < -1")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "num_values": len(all_v),
        "success_terminal_mean": float(np.mean(terminal_success)) if terminal_success else None,
        "failed_terminal_mean": float(np.mean(terminal_failed)) if terminal_failed else None,
        "max_reward_value_identity_error": max_identity_error,
        "range_check": "OK: [-1, 0]" if not issues else f"ISSUES: {issues}",
    }
