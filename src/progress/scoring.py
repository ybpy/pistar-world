"""Distance-to-progress mapping (SRPO-inspired).

For each trajectory prefix h_t:
  1. d_t = min_{c in C} ||h_t - c||
  2. d_bar, sigma_d from FAILED-prefix distances
  3. p_t = sigmoid(-alpha * (d_t - d_bar) / (sigma_d + eps))
  4. Terminal rules: success p_T = 1, failed p_T capped at p_fail_cap.

Key design: all statistics (d_bar, sigma_d) come exclusively from FAILED
prefix distances, following the SRPO convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from progress.clustering import compute_min_distance_to_centers


@dataclass
class ScoringResult:
    """Aggregated scoring results for one task."""

    # Statistics
    d_bar: float = 0.0
    sigma_d: float = 0.0
    stat_source: str = ""  # "failed-prefix" | "all-prefix" | "emergency"

    # Aggregate distance lists
    failed_prefix_distances: List[float] = field(default_factory=list)
    success_prefix_distances: List[float] = field(default_factory=list)

    # Terminal p values
    success_terminal_p_values: List[float] = field(default_factory=list)
    failed_terminal_p_values: List[float] = field(default_factory=list)

    # Alpha used
    alpha: float = 1.0


def compute_all_distances(
    trajectories: list,                         # list of Trajectory
    centers: np.ndarray,                        # [K, D]
    use_cosine: bool = False,
) -> tuple[list[float], list[float]]:
    """Compute d_t for every prefix in every trajectory (mutates trajectories in-place).

    Args:
        trajectories: list of Trajectory with prefix_embeddings already populated.
        centers:      [K, D] success center matrix.
        use_cosine:   whether to use cosine distance.
    """
    all_failed_dists = []
    all_success_dists = []

    for traj in trajectories:
        n = len(traj.prefix_embeddings)
        dists = np.zeros(n, dtype=np.float32)
        for i in range(n):
            d = compute_min_distance_to_centers(
                traj.prefix_embeddings[i], centers, use_cosine=use_cosine
            )
            dists[i] = d
            if traj.success:
                all_success_dists.append(d)
            else:
                all_failed_dists.append(d)
        traj.distances = dists

    return all_failed_dists, all_success_dists


def compute_statistics(
    failed_distances: List[float],
    success_distances: List[float],
    min_prefixes_for_stats: int = 10,
) -> tuple[float, float, str]:
    """Compute d_bar and sigma_d from FAILED-prefix distances.

    Follows SRPO convention: statistics are computed from failed prefixes ONLY.
    Falls back to all-prefix stats if too few failed prefixes.

    Returns:
        (d_bar, sigma_d, stat_source)
    """
    if len(failed_distances) >= min_prefixes_for_stats:
        d_bar = float(np.mean(failed_distances))
        sigma_d = float(np.std(failed_distances))
        stat_source = "failed-prefix"
    else:
        all_dists = failed_distances + success_distances
        if len(all_dists) >= min_prefixes_for_stats:
            d_bar = float(np.mean(all_dists))
            sigma_d = float(np.std(all_dists))
            stat_source = f"all-prefix (fallback: only {len(failed_distances)} failed prefixes)"
        else:
            d_bar = float(np.mean(all_dists)) if all_dists else 0.0
            sigma_d = float(np.std(all_dists)) if all_dists else 1.0
            stat_source = "all-prefix (emergency fallback)"

    return d_bar, sigma_d, stat_source


def compute_progress(
    trajectories: list,      # list of Trajectory with distances already populated
    d_bar: float,
    sigma_d: float,
    alpha: float = 1.0,
    eps: float = 1e-6,
    p_fail_cap: float = 0.9,
    stat_source: str = "",
) -> ScoringResult:
    """Map distances to progress p_t for all trajectories (mutates in-place).

    p_t = sigmoid(-alpha * (d_t - d_bar) / (sigma_d + eps))

    Terminal rules:
      - success: p_T = 1.0
      - failed:  p_T = min(p_T, p_fail_cap)

    Args:
        trajectories: list of Trajectory with distances populated.
        d_bar, sigma_d: statistics from failed prefixes.
        alpha:     steepness of sigmoid.
        eps:       numerical stability.
        p_fail_cap: upper bound for failed terminal p_T.

    Returns:
        ScoringResult with aggregate statistics.
    """
    result = ScoringResult(
        d_bar=d_bar,
        sigma_d=sigma_d,
        stat_source=stat_source,
        alpha=alpha,
    )

    for traj in trajectories:
        n = len(traj.prefix_timesteps)
        p_vals = np.zeros(n, dtype=np.float32)

        for i in range(n):
            z = (traj.distances[i] - d_bar) / max(sigma_d, eps)
            logit = float(np.clip(alpha * z, -80.0, 80.0))
            p = 1.0 / (1.0 + np.exp(logit))  # sigmoid(-alpha * z)
            p_vals[i] = float(p)

        # Terminal rules
        if n > 0:
            T_idx = n - 1
            if traj.success:
                p_vals[T_idx] = 1.0
            else:
                if p_vals[T_idx] > p_fail_cap:
                    p_vals[T_idx] = p_fail_cap

        # Clamp to (eps, 1.0]
        p_vals = np.clip(p_vals, eps, 1.0)
        traj.p_values = p_vals

        # Collect distances for aggregate reporting
        for i in range(n):
            d = float(traj.distances[i])
            if traj.success:
                result.success_prefix_distances.append(d)
            else:
                result.failed_prefix_distances.append(d)

    # Collect terminal p values
    for traj in trajectories:
        if traj.p_values is not None and len(traj.p_values) > 0:
            if traj.success:
                result.success_terminal_p_values.append(float(traj.p_values[-1]))
            else:
                result.failed_terminal_p_values.append(float(traj.p_values[-1]))

    return result
