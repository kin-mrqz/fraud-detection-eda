from __future__ import annotations

import numpy as np
import pandas as pd


PROPERTY_CATEGORICAL_COLS = [
    "source",
    "device_os",
    "payment_type",
    "employment_status",
]

RISK_NUMERIC_COLS = [
    "device_fraud_count",
    "velocity_24h",
]


def _risk_bucket(series: pd.Series, quantiles: tuple[float, float] = (0.33, 0.66)) -> pd.Series:
    q1, q2 = series.quantile(list(quantiles))
    return pd.cut(series, bins=[-np.inf, q1, q2, np.inf], labels=["low", "mid", "high"]).astype(str)


def build_property_graph(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build heterogeneous property graph as edge list + entity node table.

    Application nodes: index 0..n-1
    Entity nodes: encoded as negative ids per (type, value) pair
    """
    n = len(df)
    app_nodes = pd.DataFrame({"node_id": np.arange(n), "node_type": "Application", "label": df.get("fraud_bool", np.nan)})

    entity_map: dict[tuple[str, str], int] = {}
    next_entity_id = -1
    edges: list[dict[str, int | str]] = []

    for col in PROPERTY_CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        for app_idx, val in enumerate(df[col].astype(str)):
            key = (col, val)
            if key not in entity_map:
                entity_map[key] = next_entity_id
                next_entity_id -= 1
            edges.append({"src": app_idx, "dst": entity_map[key], "edge_type": f"HAS_{col.upper()}"})

    for col in RISK_NUMERIC_COLS:
        if col not in df.columns:
            continue
        buckets = _risk_bucket(df[col].replace(-1, np.nan).fillna(df[col].median()))
        for app_idx, val in enumerate(buckets):
            key = (f"{col}_band", str(val))
            if key not in entity_map:
                entity_map[key] = next_entity_id
                next_entity_id -= 1
            edges.append({"src": app_idx, "dst": entity_map[key], "edge_type": f"HAS_{col.upper()}_BAND"})

    entity_nodes = pd.DataFrame(
        [{"node_id": nid, "node_type": key[0], "value": key[1]} for key, nid in entity_map.items()]
    )
    edge_df = pd.DataFrame(edges)
    return app_nodes, edge_df, entity_nodes


def property_graph_stats(app_nodes: pd.DataFrame, edge_df: pd.DataFrame, labels: np.ndarray) -> dict:
    n_apps = len(app_nodes)
    if edge_df.empty:
        return {"n_application_nodes": n_apps, "n_entity_nodes": 0, "n_edges": 0, "avg_app_degree": 0.0}

    app_degree = edge_df.groupby("src").size()
    entity_nodes = edge_df["dst"].nunique()
    return {
        "n_application_nodes": n_apps,
        "n_entity_nodes": int(entity_nodes),
        "n_edges": int(len(edge_df)),
        "avg_app_degree": float(app_degree.mean()) if len(app_degree) else 0.0,
        "n_edge_types": int(edge_df["edge_type"].nunique()) if "edge_type" in edge_df.columns else 0,
    }


def property_graph_embeddings(
    features: np.ndarray,
    edge_df: pd.DataFrame,
    n_nodes: int,
) -> np.ndarray:
    """
    Mean-aggregate entity-linked application neighbors via shared entity nodes.
    Two-hop: app -> entity -> app (excluding self).
    """
    if edge_df.empty:
        return np.zeros_like(features)

    entity_to_apps: dict[int, list[int]] = {}
    for _, row in edge_df.iterrows():
        entity_to_apps.setdefault(int(row["dst"]), []).append(int(row["src"]))

    agg = np.zeros_like(features)
    counts = np.zeros(n_nodes, dtype=float)
    for apps in entity_to_apps.values():
        if len(apps) < 2:
            continue
        for src in apps:
            for dst in apps:
                if src == dst:
                    continue
                agg[src] += features[dst]
                counts[src] += 1.0

    for i in range(n_nodes):
        if counts[i] > 0:
            agg[i] /= counts[i]
    return agg
