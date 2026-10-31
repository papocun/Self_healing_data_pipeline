"""Read-only FastAPI service exposing pipeline results to the Next.js dashboard.

Run from repo root:  uvicorn api.main:app --port 8000
"""
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api import store
from src.utils.config import settings

app = FastAPI(title="SelfHealingPipeline API", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

SENTS = ["positive", "negative", "neutral"]
LOW_CONF = 0.6


def _f(x):
    """JSON-safe scalar."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    return x


def _pct(n, d):
    return round(100 * n / d, 1) if d else 0.0


def _top_aspects(df: pd.DataFrame, col: str, n=10):
    s = df[col].explode().dropna().astype(str).str.strip().str.lower()
    s = s[~s.isin(["", "none", "n/a", "na", "null", "nothing", "no negative aspects", "no positive aspects"])]
    return [{"aspect": k, "count": int(v)} for k, v in s.value_counts().head(n).items()]


def _range(df: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    if df.empty:
        return df
    if date_from:
        df = df[df["date_parsed"] >= pd.to_datetime(date_from, errors="coerce")]
    if date_to:
        df = df[df["date_parsed"] < pd.to_datetime(date_to, errors="coerce") + pd.Timedelta(days=1)]
    return df


@app.get("/api/overview")
def overview(date_from: str | None = None, date_to: str | None = None):
    full = store.predictions()
    df = _range(full, date_from, date_to)
    if df.empty:
        return {"empty": True, "total_all_time": len(full)}
    n = len(df)
    counts = df["sentiment"].value_counts()
    dist = [{"sentiment": s, "count": int(counts.get(s, 0)), "pct": _pct(counts.get(s, 0), n)} for s in SENTS]

    matrix = []
    for star in [1, 2, 3, 4, 5]:
        sub = df[df["stars"].round() == star]
        row = {"stars": star, "total": len(sub)}
        for s in SENTS:
            row[s] = int((sub["sentiment"] == s).sum())
        matrix.append(row)
    mismatch_pos_low = int(((df["stars"] <= 2) & (df["sentiment"] == "positive")).sum())
    mismatch_neg_high = int(((df["stars"] >= 4) & (df["sentiment"] == "negative")).sum())

    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(df["confidence"].dropna(), bins=bins)
    conf_hist = [{"lo": round(float(bins[i]), 1), "hi": round(float(bins[i + 1]), 1), "count": int(hist[i])}
                 for i in range(10)]
    low = int((df["confidence"] < LOW_CONF).sum())

    dated = df["date_parsed"].dropna()
    return {
        "empty": False,
        "kpis": {
            "total": n,
            "positive": {"count": int(counts.get("positive", 0)), "pct": _pct(counts.get("positive", 0), n)},
            "negative": {"count": int(counts.get("negative", 0)), "pct": _pct(counts.get("negative", 0), n)},
            "neutral": {"count": int(counts.get("neutral", 0)), "pct": _pct(counts.get("neutral", 0), n)},
            "avg_stars": _f(round(df["stars"].mean(), 2)),
            "avg_confidence": _f(round(df["confidence"].mean(), 3)),
        },
        "distribution": dist,
        "rating_matrix": matrix,
        "mismatches": {"positive_in_low_stars": mismatch_pos_low, "negative_in_high_stars": mismatch_neg_high},
        "confidence_hist": conf_hist,
        "low_confidence": {"threshold": LOW_CONF, "count": low, "pct": _pct(low, n)},
        "aspects": {"positive": _top_aspects(df, "positive_aspects"), "negative": _top_aspects(df, "negative_aspects")},
        "all_range": {"min": full["date_parsed"].min().isoformat(), "max": full["date_parsed"].max().isoformat()},
        "date_range": {"min": dated.min().isoformat() if len(dated) else None,
                       "max": dated.max().isoformat() if len(dated) else None},
        "model": df["model_name"].mode().iat[0] if "model_name" in df and df["model_name"].notna().any() else None,
    }


@app.get("/api/sentiment/trends")
def trends(granularity: str = Query("monthly", pattern="^(daily|weekly|monthly)$"),
           date_from: str | None = None, date_to: str | None = None):
    df = _range(store.predictions(), date_from, date_to)
    if df.empty:
        return {"points": [], "granularity": granularity}
    d = df.dropna(subset=["date_parsed"])
    if d.empty:
        return {"points": [], "granularity": granularity}
    freq = {"daily": "D", "weekly": "W", "monthly": "MS"}[granularity]
    g = d.groupby([pd.Grouper(key="date_parsed", freq=freq), "sentiment"]).size().unstack(fill_value=0)
    for s in SENTS:
        if s not in g:
            g[s] = 0
    return {
        "granularity": granularity,
        "points": [{"period": idx.strftime("%Y-%m-%d"), **{s: int(r[s]) for s in SENTS}} for idx, r in g.iterrows()],
    }


@app.get("/api/reviews")
def reviews(
    q: str | None = None,
    sentiment: str | None = None,
    stars: int | None = Query(None, ge=1, le=5),
    conf_min: float = 0.0,
    conf_max: float = 1.0,
    date_from: str | None = None,
    date_to: str | None = None,
    business_id: str | None = None,
    sort: str = "date",
    order: str = Query("desc", pattern="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    size: int = Query(25, ge=1, le=100),
):
    df = store.predictions()
    if df.empty:
        return {"total": 0, "page": page, "size": size, "items": [], "businesses": []}
    m = pd.Series(True, index=df.index)
    if q:
        m &= df["original_text"].fillna("").str.contains(q, case=False, regex=False) | (df["review_id"] == q)
    if sentiment:
        m &= df["sentiment"] == sentiment
    if stars:
        m &= df["stars"].round() == stars
    m &= df["confidence"].between(conf_min, conf_max)
    if date_from:
        m &= df["date_parsed"] >= pd.to_datetime(date_from, errors="coerce")
    if date_to:
        m &= df["date_parsed"] <= pd.to_datetime(date_to, errors="coerce") + pd.Timedelta(days=1)
    if business_id:
        m &= df["business_id"].str.contains(business_id, case=False, regex=False)
    sub = df[m]
    key = {"date": "date_parsed", "stars": "stars", "confidence": "confidence", "sentiment": "sentiment",
           "business_id": "business_id", "review_id": "review_id"}.get(sort, "date_parsed")
    sub = sub.sort_values(key, ascending=(order == "asc"), na_position="last")
    total = len(sub)
    page_df = sub.iloc[(page - 1) * size: page * size]
    items = [{
        "review_id": r.review_id, "business_id": r.business_id, "stars": _f(r.stars),
        "sentiment": r.sentiment, "confidence": _f(r.confidence), "date": r.date,
        "text": (r.original_text or "")[:300],
    } for r in page_df.itertuples()]
    return {"total": total, "page": page, "size": size, "items": items}


@app.get("/api/reviews/{review_id}")
def review_detail(review_id: str):
    df = store.predictions()
    hit = df[df["review_id"] == review_id] if not df.empty else df
    if hit.empty:
        raise HTTPException(404, "Review not found")
    r = hit.iloc[0]
    return {
        "review_id": r.review_id, "business_id": r.business_id, "user_id": r.user_id, "stars": _f(r.stars),
        "date": r.date, "sentiment": r.sentiment, "confidence": _f(r.confidence),
        "original_text": r.original_text, "cleaned_text": r.cleaned_text,
        "positive_aspects": list(r.positive_aspects), "negative_aspects": list(r.negative_aspects),
        "reasoning": r.reasoning, "model_name": r.model_name, "inference_timestamp": r.inference_timestamp,
        "batch_id": _f(r.batch_id), "useful": _f(r.useful), "funny": _f(r.funny), "cool": _f(r.cool),
    }


def _ollama_health() -> dict:
    base, model = settings.ollama.base_url, settings.ollama.model
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=2.0)
        resp.raise_for_status()
        names = [m.get("name", "").lower() for m in resp.json().get("models", [])]
        t = model.lower()
        avail = any(n == t or n == f"{t}:latest" for n in names)
        return {"reachable": True, "model": model, "model_available": avail, "installed_count": len(names)}
    except Exception:
        return {"reachable": False, "model": model, "model_available": False, "installed_count": 0}


def _bucket_freq(span: pd.Timedelta) -> str:
    if span <= pd.Timedelta(hours=3):
        return "1min"
    if span <= pd.Timedelta(days=3):
        return "1h"
    return "1D"


@app.get("/api/pipeline/status")
def pipeline_status():
    b = store.batches()
    cp = store.checkpoint()
    pred = store.predictions()
    inf = store.inference_failures()
    val = store.validation_failures()
    dup = store.duplicates()

    total_batches = len(b)
    status_counts = b["status"].value_counts().to_dict() if total_batches else {}
    ok = int(b["ok"].sum()) if total_batches else 0
    failed = int(b["failed"].sum()) if total_batches else 0
    dups = int(b["duplicates"].sum()) if total_batches else 0
    invalids = int(b["invalids"].sum()) if total_batches else 0
    quarantined = invalids + dups + failed
    seen = ok + failed + dups + invalids

    pred_ids = set(pred["review_id"]) if not pred.empty else set()
    recovered = len(inf["ids"] & pred_ids)
    unresolved = len(inf["ids"] - pred_ids)

    analytics_files = list(settings.paths.analytics_dir.glob("*")) if settings.paths.analytics_dir.exists() else []
    analytics_files = [f for f in analytics_files if f.name != ".gitkeep"]
    raw_files = [f.name for f in settings.paths.raw_data_dir.glob("*.json")] if settings.paths.raw_data_dir.exists() else []

    def stage(name, status, count=None, note=None):
        return {"stage": name, "status": status, "count": count, "note": note}

    has = total_batches > 0
    stages = [
        stage("Raw Data", "ok" if raw_files else "idle", len(raw_files), "source files"),
        stage("Ingestion", "ok" if has else "idle", seen if has else None, "records read"),
        stage("Validation", "warn" if invalids else ("ok" if has else "idle"), ok + failed + dups if has else None,
              f"{invalids} invalid" if has else None),
        stage("Cleaning", "ok" if has else "idle", ok + failed + dups if has else None, "records cleaned"),
        stage("Deduplication", "warn" if dups else ("ok" if has else "idle"), ok + failed if has else None,
              f"{dups} duplicates" if has else None),
        stage("Ollama Inference", "fail" if failed and not ok else ("warn" if failed else ("ok" if ok else "idle")),
              ok, f"{failed} failed" if has else None),
        stage("Persistence", "ok" if len(pred) else "idle", len(pred), "predictions stored"),
        stage("Analytics", "ok" if analytics_files else "missing", len(analytics_files) or None,
              "aggregates written" if analytics_files else "no analytics output produced"),
    ]

    # Throughput + checkpoint progression, bucketed in time
    throughput, progression = [], []
    if total_batches and b["ts"].notna().any():
        bb = b.dropna(subset=["ts"]).copy()
        freq = _bucket_freq(bb["ts"].max() - bb["ts"].min())
        bb["bucket"] = bb["ts"].dt.floor(freq)
        g = bb.groupby("bucket").agg(batches=("key", "count"), ok=("ok", "sum"), failed=("failed", "sum"),
                                     records=("records", "sum")).reset_index()
        throughput = [{"t": r.bucket.isoformat(), "ok": int(r.ok), "failed": int(r.failed), "batches": int(r.batches)}
                      for r in g.itertuples()]
        g["cum"] = g["batches"].cumsum()
        progression = [{"t": r.bucket.isoformat(), "completed_batches": int(r.cum)} for r in g.itertuples()]

    last_ok = None
    last_fail = None
    if total_batches:
        okb = b[b["ok"] > 0]
        fb = b[(b["failed"] > 0) | (b["status"] == "FAILED")]
        last_ok = okb["timestamp"].iat[-1] if len(okb) else None
        last_fail = fb["timestamp"].iat[-1] if len(fb) else None

    lc = cp.get("last_completed_batch") or {}
    return {
        "kpis": {
            "total_batches": total_batches,
            "completed_batches": int(status_counts.get("COMPLETED", 0)),
            "failed_batches": int(status_counts.get("FAILED", 0)),
            "successful_inferences": ok,
            "failed_inferences": failed,
            "invalid_records": invalids,
            "duplicate_records": dups,
            "quarantined_records": quarantined,
        },
        "batch_status": [{"status": k.title(), "count": int(v)} for k, v in status_counts.items()],
        "stages": stages,
        "throughput": throughput,
        "inference": {
            "successful": ok, "failed": failed, "error_classes": dict(inf["errors"]),
            "latency": None,  # not recorded by the pipeline
        },
        "recovery": {
            "retry_attempts": None,  # not persisted by the pipeline
            "failed_records_logged": inf["total"],
            "recovered": recovered,
            "unresolved": unresolved,
            "checkpoint_progression": progression,
        },
        "health": {
            "ollama": _ollama_health(),
            "latest_checkpoint": {"batch_id": lc.get("batch_id"), "timestamp": lc.get("timestamp"),
                                  "source_file": lc.get("source_file")} if lc else None,
            "checkpoint_updated_at": cp.get("updated_at"),
            "last_successful_execution": last_ok,
            "last_failed_execution": last_fail,
            "quarantine_count": quarantined,
        },
    }


@app.get("/api/pipeline/batches")
def pipeline_batches(page: int = Query(1, ge=1), size: int = Query(25, ge=1, le=100)):
    b = store.batches()
    if b.empty:
        return {"total": 0, "items": []}
    b = b.iloc[::-1]
    sl = b.iloc[(page - 1) * size: page * size]
    return {"total": len(b), "page": page, "size": size, "items": [
        {k: _f(v) for k, v in r.items() if k not in ("ts",)} for r in sl.to_dict("records")]}


@app.get("/api/data-quality")
def data_quality():
    b = store.batches()
    val = store.validation_failures()
    ok = int(b["ok"].sum()) if len(b) else 0
    failed = int(b["failed"].sum()) if len(b) else 0
    dups = int(b["duplicates"].sum()) if len(b) else 0
    invalids = int(b["invalids"].sum()) if len(b) else 0
    total = ok + failed + dups + invalids
    valid = ok + failed
    cats = val["cats"]
    missing = cats.get("Missing text", 0) + cats.get("Missing columns", 0) + cats.get("Invalid review ID", 0) \
        + cats.get("Invalid business ID", 0)

    over_time = []
    if len(b) and b["ts"].notna().any():
        bb = b.dropna(subset=["ts"]).copy()
        freq = _bucket_freq(bb["ts"].max() - bb["ts"].min())
        bb["bucket"] = bb["ts"].dt.floor(freq)
        g = bb.groupby("bucket")[["ok", "failed", "duplicates", "invalids"]].sum().reset_index()
        for r in g.itertuples():
            tot = r.ok + r.failed + r.duplicates + r.invalids
            over_time.append({"t": r.bucket.isoformat(), "valid": int(r.ok + r.failed), "invalid": int(r.invalids),
                              "duplicate": int(r.duplicates), "valid_pct": _pct(r.ok + r.failed, tot)})
    return {
        "kpis": {
            "valid": valid, "invalid": invalids, "duplicate": dups, "missing_fields": int(missing),
            "quarantined": invalids + dups + failed,
            "validation_success_rate": _pct(valid + dups, total),
            "total_seen": total,
        },
        "breakdown": [{"name": "Valid", "count": valid}, {"name": "Invalid", "count": invalids},
                      {"name": "Duplicate", "count": dups}],
        "error_categories": [{"category": k, "count": int(v)} for k, v in cats.most_common()],
        "over_time": over_time,
    }


@app.get("/api/quarantine")
def quarantine(kind: str = Query("invalid", pattern="^(invalid|duplicate|inference)$"),
               page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100)):
    return store.quarantine_page(kind, page, size)


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
