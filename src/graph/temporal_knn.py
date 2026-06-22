from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class TemporalKNNConfig:
    k: int = 20
    metric: str = "cosine"
    mutual: bool = False
    min_similarity: float | None = None


def build_temporal_knn_graph(
    features: np.ndarray,
    months: np.ndarray,
    labels: np.ndarray | None = None,
    config: TemporalKNNConfig | None = None,
) -> pd.DataFrame:
    """
    Build directed SIMILAR edges: src -> dst only when months[dst] < months[src].

    features: (n, d) preprocessed feature matrix
    months: (n,) application month per row
    """
    cfg = config or TemporalKNNConfig()
    n = features.shape[0]
    nn = NearestNeighbors(n_neighbors=min(cfg.k + 1, n), metric=cfg.metric)
    nn.fit(features)

    edges: list[dict[str, float | int]] = []
    for src in range(n):
        src_month = int(months[src])
        dists, idxs = nn.kneighbors(features[src : src + 1], return_distance=True)
        for dist, dst in zip(dists[0], idxs[0]):
            if dst == src:
                continue
            if int(months[dst]) >= src_month:
                continue
            if cfg.metric == "cosine":
                sim = 1.0 - float(dist)
            else:
                sim = 1.0 / (1.0 + float(dist))
            if cfg.min_similarity is not None and sim < cfg.min_similarity:
                continue
            edges.append({"src": int(src), "dst": int(dst), "similarity": sim})

    edge_df = pd.DataFrame(edges)
    if edge_df.empty:
        return edge_df

    if cfg.mutual:
        pairs = set(zip(edge_df["src"], edge_df["dst"]))
        reverse = {(d, s) for s, d in pairs}
        mutual_pairs = pairs & reverse
        edge_df = edge_df[
            edge_df.apply(lambda r: (int(r["src"]), int(r["dst"])) in mutual_pairs, axis=1)
        ].reset_index(drop=True)

    return edge_df


def temporal_knn_stats(
    edge_df: pd.DataFrame,
    labels: np.ndarray,
    months: np.ndarray,
) -> dict:
    """Graph topology and label homophily for temporal kNN graph."""
    n_nodes = len(labels)
    if edge_df.empty:
        return {
            "n_nodes": n_nodes,
            "n_edges": 0,
            "density": 0.0,
            "avg_degree": 0.0,
            "homophily": None,
            "temporal_violations": 0,
        }

    out_degree = edge_df.groupby("src").size()
    avg_degree = float(out_degree.mean()) if len(out_degree) else 0.0
    density = float(len(edge_df) / max(n_nodes * (n_nodes - 1), 1))

    same_label = edge_df.apply(
        lambda r: int(labels[int(r["src"])]) == int(labels[int(r["dst"])]), axis=1
    )
    homophily = float(same_label.mean()) if len(same_label) else None

    violations = int(
        edge_df.apply(lambda r: int(months[int(r["dst"])]) >= int(months[int(r["src"])]), axis=1).sum()
    )

    return {
        "n_nodes": n_nodes,
        "n_edges": int(len(edge_df)),
        "density": density,
        "avg_degree": avg_degree,
        "homophily": homophily,
        "temporal_violations": violations,
    }


def mean_neighbor_features(
    features: np.ndarray,
    edge_df: pd.DataFrame,
    n_nodes: int,
) -> np.ndarray:
    """GraphSAGE-lite: 1-hop mean pooling of neighbor features per node."""
    agg = np.zeros_like(features)
    counts = np.zeros(n_nodes, dtype=float)
    if edge_df.empty:
        return agg

    for _, row in edge_df.iterrows():
        src, dst = int(row["src"]), int(row["dst"])
        agg[src] += features[dst]
        counts[src] += 1.0

    for i in range(n_nodes):
        if counts[i] > 0:
            agg[i] /= counts[i]
    return agg
