"""Success center construction via DBSCAN clustering.

Provides:
  - full_success_center:         cluster FULL success trajectory embeddings W(o_{0:T}).
  - success_prefix_center:       (reserved) cluster success PREFIX embeddings W(o_{0:t}).
  - compute_min_distance:        min L2 / cosine distance from one embedding to centers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize as l2_normalize


@dataclass
class ClusteringResult:
    """Result of success center construction."""

    centers: np.ndarray  # [K, D]
    method: str = "full_success_center"  # full_success_center | success_prefix_center
    num_input_points: int = 0
    num_centers: int = 0
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    use_cosine: bool = False
    dbscan_fallback: bool = False
    fallback_reason: str = ""
    num_noise_points: int = 0
    num_clusters: int = 0


def cluster_full_success_trajectories(
    full_trajectory_embeddings: np.ndarray,  # [N, D]
    *,
    eps: float = 0.5,
    min_samples: int = 2,
    use_cosine: bool = False,
) -> ClusteringResult:
    """Cluster FULL success trajectory embeddings with DBSCAN.

    Args:
        full_trajectory_embeddings: [N, D] where N = number of COMPLETE success
                                    trajectories, D = V-JEPA embedding dim.
        eps:        DBSCAN epsilon.
        min_samples: DBSCAN min_samples.
        use_cosine:  if True, L2-normalize embeddings and use cosine distance.

    Returns:
        ClusteringResult with centers and metadata.

    Fallback:
        If DBSCAN labels all points as noise, fall back to using ALL full success
        trajectory embeddings as individual centers.
    """
    n = len(full_trajectory_embeddings)

    if n == 0:
        embedding_dim = full_trajectory_embeddings.shape[1] if full_trajectory_embeddings.ndim == 2 else 0
        return ClusteringResult(
            centers=np.empty((0, embedding_dim), dtype=np.float32),
            num_input_points=0,
            num_centers=0,
            dbscan_eps=eps,
            dbscan_min_samples=min_samples,
            use_cosine=use_cosine,
        )

    if n == 1:
        return ClusteringResult(
            centers=full_trajectory_embeddings.copy(),
            num_input_points=1,
            num_centers=1,
            dbscan_eps=eps,
            dbscan_min_samples=min_samples,
            use_cosine=use_cosine,
        )

    # Cosine DBSCAN must operate on normalized vectors directly. Per-feature
    # standardization changes vector directions and invalidates cosine geometry.
    if use_cosine:
        cluster_inputs = l2_normalize(full_trajectory_embeddings, norm="l2")
    else:
        cluster_inputs = full_trajectory_embeddings.copy()

    metric = "cosine" if use_cosine else "euclidean"
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(cluster_inputs)

    unique_labels = set(clustering.labels_) - {-1}
    n_noise = int((clustering.labels_ == -1).sum())

    centers = []
    for label in sorted(unique_labels):
        mask = clustering.labels_ == label
        center = cluster_inputs[mask].mean(axis=0)
        if use_cosine:
            center = center / (np.linalg.norm(center) + 1e-12)
        centers.append(center)

    fallback = False
    fallback_reason = ""

    if len(centers) == 0:
        fallback = True
        fallback_reason = (
            f"DBSCAN found NO clusters (all {n} points are noise). "
            f"Fallback: using all {n} full success trajectory embeddings as individual centers."
        )
        centers = [full_trajectory_embeddings[i].copy() for i in range(n)]
        print(f"    ⚠️  {fallback_reason}")
    else:
        print(
            f"    ✓ DBSCAN: {len(centers)} clusters, {n_noise} noise points "
            f"(out of {n} total)."
        )

    return ClusteringResult(
        centers=np.array(centers),
        num_input_points=n,
        num_centers=len(centers),
        dbscan_eps=eps,
        dbscan_min_samples=min_samples,
        use_cosine=use_cosine,
        dbscan_fallback=fallback,
        fallback_reason=fallback_reason,
        num_noise_points=n_noise,
        num_clusters=len(unique_labels),
    )


def cluster_success_prefixes(
    prefix_embeddings_list: list[np.ndarray],  # list of [T_i, D]
    *,
    eps: float = 0.5,
    min_samples: int = 2,
    use_cosine: bool = False,
) -> ClusteringResult:
    """(Reserved) Cluster success PREFIX embeddings.

    Currently a stub — not yet implemented.  Falls back to concatenating all
    prefix embeddings and clustering them.
    """
    nonempty = [embeddings for embeddings in prefix_embeddings_list if len(embeddings) > 0]
    if not nonempty:
        return cluster_full_success_trajectories(
            np.empty((0, 0), dtype=np.float32),
            eps=eps,
            min_samples=min_samples,
            use_cosine=use_cosine,
        )
    all_prefixes = np.concatenate(nonempty, axis=0)
    return cluster_full_success_trajectories(
        all_prefixes,
        eps=eps,
        min_samples=min_samples,
        use_cosine=use_cosine,
    )


def compute_min_distance_to_centers(
    embedding: np.ndarray,          # [D]
    cluster_centers: np.ndarray,    # [K, D]
    *,
    use_cosine: bool = False,
) -> float:
    """Compute min L2 (or cosine) distance from one embedding to cluster centers.

    Args:
        embedding:        single embedding vector [D].
        cluster_centers:  [K, D] center matrix.
        use_cosine:       if True, use cosine distance (1 - cos_similarity).

    Returns:
        min distance (float).  If no centers exist, returns inf.
    """
    if len(cluster_centers) == 0:
        return float("inf")

    if use_cosine:
        a = embedding / (np.linalg.norm(embedding) + 1e-12)
        b = cluster_centers / (np.linalg.norm(cluster_centers, axis=1, keepdims=True) + 1e-12)
        similarities = np.dot(b, a)
        distances = 1.0 - similarities
    else:
        distances = np.linalg.norm(cluster_centers - embedding, axis=1)

    return float(np.min(distances))
