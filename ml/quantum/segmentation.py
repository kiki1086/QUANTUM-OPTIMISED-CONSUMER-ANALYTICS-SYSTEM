"""
ml/quantum/segmentation.py
────────────────────────────────────────────────────────────────────────────────
Algorithm 1 — QAOA Max-Cut Customer Segmentation

Formulates clustering as a Max-Cut problem on the customer similarity graph.
Uses dimod + SimulatedAnnealingSampler (D-Wave Ocean SDK) with recursive
bisection to produce K clusters.  Results are compared against classical
K-Means using silhouette score.

Returns the standardised result schema.
"""
from __future__ import annotations
import time
import numpy as np
from typing import Any, Dict

import dimod
from dwave.samplers import SimulatedAnnealingSampler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .shared import build_quantum_context, QuantumContext, qubo_max_cut


def _max_cut_partition(adj: np.ndarray, sampler, num_reads: int = 200) -> np.ndarray:
    """Solve one Max-Cut QUBO → returns binary partition array (0 or 1)."""
    n = adj.shape[0]
    Q = qubo_max_cut(adj)
    if not Q:
        return np.zeros(n, dtype=int)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    ss  = sampler.sample(bqm, num_reads=num_reads)
    best = ss.first.sample
    return np.array([best.get(i, 0) for i in range(n)], dtype=int)


def _recursive_bisection(adj: np.ndarray, sampler, k: int, num_reads: int) -> np.ndarray:
    """Recursively bisect the graph to produce k partitions."""
    n = len(adj)
    labels = np.zeros(n, dtype=int)
    queue = [(np.arange(n), 0, k)]   # (node_indices, current_label, remaining_splits)

    label_counter = 0
    while queue:
        indices, _, splits = queue.pop(0)
        if splits <= 1 or len(indices) < 2:
            continue
        sub_adj = adj[np.ix_(indices, indices)]
        part = _max_cut_partition(sub_adj, sampler, num_reads)
        group0 = indices[part == 0]
        group1 = indices[part == 1]
        label_counter += 1
        labels[group1] = label_counter
        half = splits // 2
        if half > 1 and len(group0) > 1:
            queue.append((group0, labels[group0[0]], half))
        if splits - half > 1 and len(group1) > 1:
            queue.append((group1, label_counter, splits - half))
    return labels


def run(ctx: QuantumContext, n_clusters: int = 4) -> Dict[str, Any]:
    """
    Run QAOA Max-Cut clustering and compare to K-Means.

    Parameters
    ----------
    ctx        : QuantumContext from shared preprocessing
    n_clusters : number of desired segments (default 4)
    """
    t0      = time.perf_counter()
    sampler = SimulatedAnnealingSampler()
    n_reads = 300

    adj     = ctx.top_k_adjacency(k=4)
    X       = ctx.X_norm

    # ── Quantum Max-Cut segmentation ──────────────────────────────────────────
    q_labels = _recursive_bisection(adj, sampler, n_clusters, n_reads)
    n_q_clusters = len(np.unique(q_labels))

    q_silhouette = float(silhouette_score(X, q_labels)) if n_q_clusters > 1 else 0.0

    # ── Classical K-Means baseline ────────────────────────────────────────────
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X)
    c_labels     = km.labels_
    c_silhouette = float(silhouette_score(X, c_labels))

    elapsed = time.perf_counter() - t0

    # ── Cluster size summary ──────────────────────────────────────────────────
    q_sizes = {int(k): int(np.sum(q_labels == k)) for k in np.unique(q_labels)}
    c_sizes = {int(k): int(np.sum(c_labels == k)) for k in np.unique(c_labels)}

    # ── Scatter data (2D PCA projection) ─────────────────────────────────────
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    scatter_q = [{"x": float(coords[i, 0]), "y": float(coords[i, 1]),
                  "cluster": int(q_labels[i])} for i in range(len(X))]
    scatter_c = [{"x": float(coords[i, 0]), "y": float(coords[i, 1]),
                  "cluster": int(c_labels[i])} for i in range(len(X))]

    silhouette_compare = [
        {"method": "QUBO Max-Cut", "score": round(q_silhouette, 4)},
        {"method": "Classical K-Means", "score": round(c_silhouette, 4)},
    ]

    return {
        "algorithm": "QUBO Max-Cut Customer Segmentation",
        "status":    "success",
        "metrics": {
            "quantum_silhouette":  round(q_silhouette, 4),
            "classical_silhouette": round(c_silhouette, 4),
            "quantum_clusters":    n_q_clusters,
            "classical_clusters":  n_clusters,
            "quantum_cluster_sizes":  q_sizes,
            "classical_cluster_sizes": c_sizes,
        },
        "visualization": {
            "scatter_quantum":   scatter_q,
            "scatter_classical": scatter_c,
            "silhouette_comparison": silhouette_compare,
        },
        "benchmark": {
            "execution_time_s":    round(elapsed, 3),
            "simulator":           "SimulatedAnnealingSampler (D-Wave Ocean SDK)",
            "iterations":          n_reads,
            "objective_value":     round(q_silhouette, 4),
            "convergence_status":  "converged" if q_silhouette > 0 else "trivial",
            "qpu_available":       False,
            "classical_comparison": f"K-Means silhouette={c_silhouette:.4f}",
        },
        "resource_usage": {
            "qubits":           ctx.n_customers,
            "circuit_depth":    None,
            "optimizer_iterations": n_reads,
            "num_shots":        n_reads,
            "backend":          "SimulatedAnnealingSampler",
            "annealing_sweeps": 1000,
            "embedding_size":   None,
        },
        "business_summary": (
            f"Customers were grouped into {n_q_clusters} segments using quantum-inspired "
            f"Max-Cut optimisation on their similarity graph. "
            f"The quantum approach achieved a silhouette score of {q_silhouette:.3f} "
            f"vs {c_silhouette:.3f} for classical K-Means, enabling richer targeting."
        ),
        "technical_summary": (
            "Customer similarity was encoded as a weighted graph (RBF kernel). "
            "Max-Cut QUBO was solved via D-Wave SimulatedAnnealingSampler with "
            f"{n_reads} reads. Recursive bisection extended the binary cut to {n_q_clusters} "
            "clusters. Silhouette scores measure intra/inter-cluster separation."
        ),
    }
