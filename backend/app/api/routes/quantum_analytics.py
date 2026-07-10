"""
backend/app/api/routes/quantum_analytics.py
────────────────────────────────────────────────────────────────────────────────
POST /quantum/analyze  — runs the 7 heavy algorithms once and caches the context.
GET  /quantum/lookalike — runs ONLY the look-alike search (< 0.5 s) using cached ctx.

This split means changing "Query Customer #" never re-runs the heavy algorithms.
"""
from __future__ import annotations
import sys
import os
import time
import traceback
import concurrent.futures
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Session store (shared with ingestion.py) ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ml'))

try:
    from app.api.routes.ingestion import SESSION_STORE
except ImportError:
    SESSION_STORE: Dict[str, Any] = {}

# ── Quantum context cache (session_id → QuantumContext) ───────────────────────
# Avoids rebuilding the 150-customer preprocessing on every lookalike call.
CTX_CACHE: Dict[str, Any] = {}

# ── Quantum modules ───────────────────────────────────────────────────────────
try:
    from quantum.shared import build_quantum_context
    from quantum import (segmentation, value_tiering, feature_interact,
                         lookalike, qsvc, loyalty_markov,
                         category_match, quantum_feature_importance)
    _QUANTUM_AVAILABLE = True
    _QUANTUM_ERROR     = ""
except Exception as _qe:
    _QUANTUM_AVAILABLE = False
    _QUANTUM_ERROR     = str(_qe)

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    session_id: str
    query_customer_idx: int = 0
    n_clusters: int = 4


def _run_algorithm(name: str, fn, *args) -> Dict[str, Any]:
    """Run a single algorithm, returning a structured error result on failure."""
    try:
        return fn(*args)
    except Exception:
        tb = traceback.format_exc()
        return {
            "algorithm": name,
            "status":    "error",
            "error":     tb[-500:],
            "metrics": {}, "visualization": {}, "benchmark": {},
            "resource_usage": {},
            "business_summary": f"{name} encountered an error.",
            "technical_summary": tb[-300:],
        }


def _get_or_build_ctx(session_id: str):
    """Return cached QuantumContext or build a new one from the stored DataFrame."""
    if session_id in CTX_CACHE:
        return CTX_CACHE[session_id]

    df = SESSION_STORE.get(session_id)
    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Upload a CSV first."
        )
    try:
        ctx = build_quantum_context(df, max_customers=150)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    # Cache the context (bounded to 10 sessions)
    if len(CTX_CACHE) >= 10:
        oldest = next(iter(CTX_CACHE))
        del CTX_CACHE[oldest]
    CTX_CACHE[session_id] = ctx
    return ctx


# ── POST /quantum/analyze — heavy 7-algorithm run ────────────────────────────
@router.post("/analyze")
async def run_quantum_analysis(req: AnalyzeRequest):
    if not _QUANTUM_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Quantum ML modules unavailable: {_QUANTUM_ERROR}"
        )

    t_total = time.perf_counter()
    ctx = _get_or_build_ctx(req.session_id)

    results: Dict[str, Any] = {}

    # ── Group A: 6 QUBO/annealing algorithms (run in parallel) ───────────────
    group_a = [
        ("segmentation",       lambda: _run_algorithm("QUBO Max-Cut Segmentation",
                                                       segmentation.run, ctx, req.n_clusters)),
        ("value_tiering",      lambda: _run_algorithm("QUBO Value Tiering",
                                                       value_tiering.run, ctx)),
        ("feature_interact",   lambda: _run_algorithm("Feature Interaction Discovery",
                                                       feature_interact.run, ctx)),
        ("loyalty_retention",  lambda: _run_algorithm("Retention Optimization",
                                                       loyalty_markov.run, ctx)),
        ("category_match",     lambda: _run_algorithm("Category Affinity Matching",
                                                       category_match.run, ctx)),
        ("feature_importance", lambda: _run_algorithm("Quantum-Assisted Feature Importance",
                                                       quantum_feature_importance.run, ctx)),
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {name: ex.submit(fn) for name, fn in group_a}
        for name, fut in futures.items():
            results[name] = fut.result()

    # ── Group B: lookalike (fast, uses query_customer_idx) ────────────────────
    results["lookalike"] = _run_algorithm(
        "Grover-Inspired Similarity Search",
        lookalike.run, ctx, req.query_customer_idx, 10)

    # ── Group C: QSVC (heavier Qiskit circuit) ────────────────────────────────
    results["qsvc"] = _run_algorithm("Quantum Kernel SVM", qsvc.run, ctx)

    total_elapsed = round(time.perf_counter() - t_total, 2)

    return JSONResponse(content={
        "status":               "success",
        "session_id":           req.session_id,
        "n_customers":          ctx.n_customers,
        "feature_names":        ctx.feature_names,
        "total_time_s":         total_elapsed,
        "preprocessing_time_s": round(ctx.build_time_s, 3),
        "algorithms":           results,
    })


# ── GET /quantum/lookalike — fast re-query using cached context ───────────────
@router.get("/lookalike")
def get_lookalike(
    session_id:          str = Query(..., description="Session ID from CSV upload"),
    query_customer_idx:  int = Query(0,   description="Index of the query customer"),
    top_k:               int = Query(10,  description="Number of look-alike results"),
):
    """
    Re-runs ONLY the Grover-Inspired Similarity Search using the cached
    QuantumContext — typical response time < 200 ms.
    """
    if not _QUANTUM_AVAILABLE:
        raise HTTPException(status_code=503, detail=f"Quantum ML unavailable: {_QUANTUM_ERROR}")

    ctx = _get_or_build_ctx(session_id)

    result = _run_algorithm(
        "Grover-Inspired Similarity Search",
        lookalike.run, ctx, query_customer_idx, top_k
    )
    return JSONResponse(content={"status": "success", "lookalike": result})


# ── GET /quantum/status ───────────────────────────────────────────────────────
@router.get("/status")
def quantum_status():
    return {
        "quantum_available": _QUANTUM_AVAILABLE,
        "session_count":     len(SESSION_STORE),
        "ctx_cache_count":   len(CTX_CACHE),
        "error":             _QUANTUM_ERROR or None,
    }
