import io
import uuid
from typing import Any, Dict
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

# ── Session store: keyed by UUID, holds normalised DataFrames ─────────────────
# Imported by quantum_analytics.py for stateless DataFrame retrieval
SESSION_STORE: Dict[str, Any] = {}


# ── Fuzzy column finder ────────────────────────────────────────────────────────
def find_col(df: pd.DataFrame, *keywords: str) -> str | None:
    """
    Return the first column whose lowercased name contains ALL of the
    given keyword substrings.  E.g. find_col(df, 'annual', 'income').
    """
    cols_lower = {c: c.lower().replace("_", " ").replace("-", " ") for c in df.columns}
    for col, low in cols_lower.items():
        if all(kw in low for kw in keywords):
            return col
    # Second pass: match any single keyword (most specific first)
    for kw in keywords:
        for col, low in cols_lower.items():
            if kw in low:
                return col
    return None


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw CSV columns to canonical internal names."""
    rename = {}

    mappings = [
        ("customer_id",        ["customer"]),
        ("gender",             ["gender"]),
        ("age_group",          ["age", "group"]),   # must come before plain "age"
        ("age",                ["age"]),
        ("annual_income",      ["annual", "income"]),
        ("spending_score",     ["spending", "score"]),
        ("credit_score",       ["credit", "score"]),
        ("loyalty_years",      ["loyalty"]),
        ("preferred_category", ["category"]),
    ]

    used = set()
    for canonical, keys in mappings:
        col = find_col(df, *keys)
        if col and col not in used:
            rename[col] = canonical
            used.add(col)

    df = df.rename(columns=rename)
    return df


# ── Analytics engine ──────────────────────────────────────────────────────────
def analyze(df: pd.DataFrame) -> dict:
    result: dict = {}
    result["row_count"] = len(df)
    result["columns"]   = df.columns.tolist()

    def col(name): return name if name in df.columns else None

    # ── Helper: safe numeric mean/max ──────────────────────────────────────────
    def safe_mean(c): return round(df[c].dropna().astype(float).mean(), 2) if c and col(c) else None
    def safe_max(c):  return df[c].dropna().astype(float).max() if c and col(c) else None

    # ── KPIs ───────────────────────────────────────────────────────────────────
    ai  = col("annual_income")
    ss  = col("spending_score")
    ly  = col("loyalty_years")
    cs  = col("credit_score")
    age = col("age")

    kpis = [
        {
            "label": "Total Customers",
            "value": f"{len(df):,}",
            "trend": f"{len(df):,} records loaded",
        },
        {
            "label": "Avg Annual Income",
            "value": f"${safe_mean(ai):,.0f}" if ai else "See Chart ↓",
            "trend": f"Max: ${safe_max(ai):,.0f}" if ai else "Income distribution below",
        },
        {
            "label": "Avg Spending Score",
            "value": f"{safe_mean(ss):.1f} / 100" if ss else "See Chart ↓",
            "trend": f"Min: {df[ss].min():.0f}  Max: {df[ss].max():.0f}" if ss else "Spending chart below",
        },
        {
            "label": "Avg Loyalty Years",
            "value": f"{safe_mean(ly):.1f} yrs" if ly else "See Chart ↓",
            "trend": f"Max: {safe_max(ly):.0f}" if ly else "Loyalty chart below",
        },
    ]
    result["kpis"] = kpis

    # ── Extra stat row ─────────────────────────────────────────────────────────
    result["extra_stats"] = {
        "avg_age":          round(safe_mean(age), 1) if age else None,
        "avg_credit_score": round(safe_mean(cs), 1)  if cs  else None,
        "gender_count":     df["gender"].value_counts().to_dict() if col("gender") else {},
        "age_range":        f"{int(df[age].min())}–{int(df[age].max())}" if age else None,
    }

    # ── Income histogram ───────────────────────────────────────────────────────
    income_hist = []
    if ai:
        series = df[ai].dropna().astype(float)
        hist, edges = pd.cut(series, bins=10, retbins=True)
        counts = hist.value_counts(sort=False)
        for interval, count in counts.items():
            lo, hi = int(interval.left / 1_000), int(interval.right / 1_000)
            income_hist.append({"name": f"${lo}k–${hi}k", "count": int(count)})
    result["income_hist"] = income_hist

    # ── Spending score by Age Group ────────────────────────────────────────────
    spending_by_age = []
    ag = col("age_group")
    if ag and ss:
        grp = df.groupby(ag)[ss].mean().reset_index()
        spending_by_age = [{"name": str(r[ag]), "value": round(r[ss], 1)} for _, r in grp.iterrows()]
    elif age and ss:
        # Fallback: bucket age into groups
        df["_age_bin"] = pd.cut(df[age], bins=[0,25,35,45,60,120],
                                labels=["18–25","26–35","36–45","46–60","60+"])
        grp = df.groupby("_age_bin")[ss].mean().reset_index()
        spending_by_age = [{"name": str(r["_age_bin"]), "value": round(r[ss], 1)}
                           for _, r in grp.iterrows()]
    result["spending_by_age"] = spending_by_age

    # ── Gender distribution ───────────────────────────────────────────────────
    gender_dist = []
    if col("gender"):
        g = df["gender"].value_counts()
        gender_dist = [{"name": str(k), "value": int(v)} for k, v in g.items()]
    result["gender_dist"] = gender_dist

    # ── Preferred category ────────────────────────────────────────────────────
    category_summary = []
    pc = col("preferred_category")
    if pc:
        cat = df[pc].value_counts().head(10)
        category_summary = [{"name": str(k), "value": int(v)} for k, v in cat.items()]
    result["category_summary"] = category_summary

    # ── Income vs Spending scatter ─────────────────────────────────────────────
    income_vs_spending = []
    if ai and ss:
        sample = df[[ai, ss]].dropna().head(300)
        income_vs_spending = [
            {"income": round(float(r[ai]), 0), "spending": round(float(r[ss]), 1)}
            for _, r in sample.iterrows()
        ]
    result["income_vs_spending"] = income_vs_spending

    # ── Credit score by Gender ─────────────────────────────────────────────────
    credit_by_gender = []
    if cs and col("gender"):
        grp = df.groupby("gender")[cs].mean().reset_index()
        credit_by_gender = [{"name": str(r["gender"]), "value": round(float(r[cs]), 1)}
                            for _, r in grp.iterrows()]
    result["credit_by_gender"] = credit_by_gender

    # ── Loyalty years by Age Group ────────────────────────────────────────────
    loyalty_by_age = []
    if ly:
        if ag:
            grp = df.groupby(ag)[ly].mean().reset_index()
            loyalty_by_age = [{"name": str(r[ag]), "value": round(float(r[ly]), 2)}
                               for _, r in grp.iterrows()]
        elif age:
            df["_age_bin2"] = pd.cut(df[age], bins=[0,25,35,45,60,120],
                                     labels=["18–25","26–35","36–45","46–60","60+"])
            grp = df.groupby("_age_bin2")[ly].mean().reset_index()
            loyalty_by_age = [{"name": str(r["_age_bin2"]), "value": round(float(r[ly]), 2)}
                               for _, r in grp.iterrows()]
    result["loyalty_by_age"] = loyalty_by_age

    # ── Income vs Credit Score ────────────────────────────────────────────────
    income_vs_credit = []
    if ai and cs:
        sample = df[[ai, cs]].dropna().head(300)
        income_vs_credit = [
            {"income": round(float(r[ai]), 0), "credit": round(float(r[cs]), 0)}
            for _, r in sample.iterrows()
        ]
    result["income_vs_credit"] = income_vs_credit

    # ── Spending score distribution histogram ──────────────────────────────────
    spending_hist = []
    if ss:
        series = df[ss].dropna().astype(float)
        hist, edges = pd.cut(series, bins=10, retbins=True)
        counts = hist.value_counts(sort=False)
        for interval, count in counts.items():
            spending_hist.append({"name": f"{int(interval.left)}–{int(interval.right)}", "count": int(count)})
    result["spending_hist"] = spending_hist

    # ── Data preview ──────────────────────────────────────────────────────────
    result["preview"] = df.head(8).fillna("").astype(str).to_dict(orient="records")

    return result


# ── Route ─────────────────────────────────────────────────────────────────────
@router.post("/csv-upload")
async def upload_csv_data(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8-sig")))   # utf-8-sig handles Excel BOM
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=422, detail="Uploaded CSV is empty.")

    # Log raw columns for debugging
    raw_cols = df.columns.tolist()
    df = map_columns(df)

    # ── Store DataFrame for quantum analysis ──────────────────────────────────
    session_id = str(uuid.uuid4())
    SESSION_STORE[session_id] = df
    # Keep session store bounded (max 20 sessions)
    if len(SESSION_STORE) > 20:
        oldest = next(iter(SESSION_STORE))
        del SESSION_STORE[oldest]

    analytics = analyze(df)
    analytics["raw_columns"] = raw_cols
    analytics["filename"]    = file.filename
    analytics["status"]      = "success"
    analytics["session_id"]  = session_id

    return JSONResponse(content=analytics)


@router.get("/status/{job_id}")
def get_ingestion_status(job_id: str):
    return {"job_id": job_id, "status": "completed"}
