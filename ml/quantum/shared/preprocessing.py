"""
ml/quantum/shared/preprocessing.py
────────────────────────────────────────────────────────────────────────────────
Centralised preprocessing layer shared by all quantum modules.

All modules should call `build_quantum_context(df)` once and pass the
returned QuantumContext object around.  This prevents each module from
redundantly normalising data or recomputing similarity graphs.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# ── Numeric features used across all quantum modules ─────────────────────────
NUMERIC_FEATURES  = ["annual_income", "spending_score", "credit_score",
                     "loyalty_years", "age"]
CATEGORICAL_FEATS = ["gender", "preferred_category", "age_group"]


class QuantumContext:
    """
    Immutable container produced by build_quantum_context().
    Shared by every quantum module in the suite.
    """
    def __init__(
        self,
        df_raw:        pd.DataFrame,
        X_norm:        np.ndarray,
        feature_names: List[str],
        cosine_sim:    np.ndarray,
        dist_matrix:   np.ndarray,
        similarity_graph: nx.Graph,
        label_encoders: Dict[str, LabelEncoder],
        scaler:         StandardScaler,
        n_customers:    int,
        build_time_s:   float,
    ):
        self.df_raw          = df_raw
        self.X_norm          = X_norm           # shape (n, k) float64
        self.feature_names   = feature_names    # canonical names of columns used
        self.cosine_sim      = cosine_sim        # (n, n)
        self.dist_matrix     = dist_matrix       # (n, n) Euclidean
        self.similarity_graph = similarity_graph # nx.Graph with edge weight = cosine_sim
        self.label_encoders  = label_encoders
        self.scaler          = scaler
        self.n_customers     = n_customers
        self.build_time_s    = build_time_s

    # ── Derived helpers ───────────────────────────────────────────────────────
    def top_k_adjacency(self, k: int = 5) -> np.ndarray:
        """Sparse adjacency: keep only top-k neighbours per row."""
        n   = self.n_customers
        adj = np.zeros((n, n), dtype=float)
        for i in range(n):
            row  = self.cosine_sim[i].copy()
            row[i] = -1.0
            top_k = np.argsort(row)[-k:]
            for j in top_k:
                w = max(self.cosine_sim[i, j], 0.0)
                adj[i, j] = w
                adj[j, i] = w
        return adj

    def customer_vector(self, idx: int) -> np.ndarray:
        return self.X_norm[idx]


# ── QUBO builder helpers ──────────────────────────────────────────────────────
def qubo_max_cut(adj: np.ndarray) -> Dict[Tuple[int,int], float]:
    """
    Build QUBO for Max-Cut:  H = -Σ_{(i,j)} w_ij (x_i + x_j - 2 x_i x_j)
    Returns a dict {(i,j): coefficient} compatible with dimod.
    """
    n = adj.shape[0]
    Q: Dict[Tuple[int,int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            w = adj[i, j]
            if w == 0:
                continue
            Q[(i, i)] = Q.get((i, i), 0.0) - w
            Q[(j, j)] = Q.get((j, j), 0.0) - w
            Q[(i, j)] = Q.get((i, j), 0.0) + 2.0 * w
    return Q


def qubo_max_weight_matching(
    biadj: np.ndarray, n_left: int
) -> Dict[Tuple[int,int], float]:
    """
    Build QUBO for max-weight bipartite matching.
    biadj: (n_left, n_right) weight matrix.
    Variables 0..n_left-1 correspond to left nodes,
    n_left..n_left+n_right-1 to right nodes.
    Soft-constraint: each node matched at most once.
    """
    n_right = biadj.shape[1]
    N = n_left + n_right
    Q: Dict[Tuple[int,int], float] = {}
    penalty = float(np.max(biadj) + 1.0)

    # Reward edges
    for i in range(n_left):
        for j in range(n_right):
            w = biadj[i, j]
            qi, qj = i, n_left + j
            Q[(qi, qj)] = Q.get((qi, qj), 0.0) - w

    # Penalty: each left node matched ≤1 right node
    for i in range(n_left):
        for j1 in range(n_right):
            for j2 in range(j1 + 1, n_right):
                qi1, qi2 = n_left + j1, n_left + j2
                Q[(qi1, qi2)] = Q.get((qi1, qi2), 0.0) + penalty

    return Q


# ── Main factory ──────────────────────────────────────────────────────────────
def build_quantum_context(
    df: pd.DataFrame,
    max_customers: int = 200,
) -> QuantumContext:
    """
    Build a QuantumContext from a customer DataFrame.

    Parameters
    ----------
    df            : normalised-column DataFrame (output of ingestion map_columns)
    max_customers : cap dataset size for tractable quantum circuits.
    """
    t0 = time.perf_counter()

    # ── 1. Subset ─────────────────────────────────────────────────────────────
    if len(df) > max_customers:
        df = df.sample(n=max_customers, random_state=42).reset_index(drop=True)
    df = df.copy()

    # ── 2. Encode categoricals ────────────────────────────────────────────────
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATS:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str).fillna("Unknown"))
            label_encoders[col] = le

    # ── 3. Select numeric features present in this dataset ───────────────────
    all_numeric = NUMERIC_FEATURES + [c + "_enc" for c in CATEGORICAL_FEATS]
    present = [c for c in all_numeric if c in df.columns]
    if not present:
        raise ValueError("Dataset contains no usable numeric features.")
    X_raw = df[present].fillna(0).astype(float).values

    # ── 4. Normalise ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_raw)

    # ── 5. Similarity matrices ────────────────────────────────────────────────
    cos_sim  = cosine_similarity(X_norm)          # range [-1,1] → clamp to [0,1]
    cos_sim  = np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0)
    dist_mat = euclidean_distances(X_norm)

    # ── 6. Similarity graph (top-3 edges per node for sparsity) ───────────────
    G = nx.Graph()
    n = len(df)
    G.add_nodes_from(range(n))
    for i in range(n):
        row = cos_sim[i].copy(); row[i] = -1.0
        top3 = np.argsort(row)[-3:]
        for j in top3:
            w = float(cos_sim[i, j])
            if w > 0 and not G.has_edge(i, j):
                G.add_edge(i, j, weight=w)

    build_time = time.perf_counter() - t0
    return QuantumContext(
        df_raw=df,
        X_norm=X_norm,
        feature_names=present,
        cosine_sim=cos_sim,
        dist_matrix=dist_mat,
        similarity_graph=G,
        label_encoders=label_encoders,
        scaler=scaler,
        n_customers=n,
        build_time_s=build_time,
    )
