"""Read-only data access over pipeline outputs. Caches by directory signature."""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import settings

PATHS = settings.paths

_cache: dict[str, tuple[Any, Any]] = {}


def _sig(d: Path, pattern: str) -> tuple:
    if not d.exists():
        return ()
    fs = list(d.glob(pattern))
    return (len(fs), max((f.stat().st_mtime for f in fs), default=0))


def _cached(key: str, sig: tuple, build):
    hit = _cache.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    val = build()
    _cache[key] = (sig, val)
    return val


def _aspects(v) -> list[str]:
    if v is None:
        return []
    try:
        return [str(x).strip() for x in v if str(x).strip()]
    except TypeError:
        return []


def predictions() -> pd.DataFrame:
    def build():
        frames = []
        for f in sorted(PATHS.predictions_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                if not df.empty:
                    frames.append(df)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True).drop_duplicates("review_id", keep="last")
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        df["positive_aspects"] = df["positive_aspects"].map(_aspects)
        df["negative_aspects"] = df["negative_aspects"].map(_aspects)
        return df

    return _cached("pred", _sig(PATHS.predictions_dir, "*.parquet"), build)


def checkpoint() -> dict:
    p = PATHS.checkpoint_dir / "pipeline_checkpoint.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def batches() -> pd.DataFrame:
    cp = checkpoint()
    rows = []
    for key, b in (cp.get("completed_batches") or {}).items():
        m = b.get("metadata") or {}
        rows.append({
            "key": key,
            "batch_id": b.get("batch_id"),
            "source_file": b.get("source_file"),
            "timestamp": b.get("timestamp"),
            "status": str(b.get("processing_status", "")).upper(),
            "records": b.get("records_count", 0),
            "ok": m.get("successful_predictions", 0),
            "failed": m.get("failed_predictions", 0),
            "duplicates": m.get("duplicates", 0),
            "invalids": m.get("invalids", 0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def _failed_files(prefix: str) -> list[Path]:
    d = PATHS.failed_data_dir
    return sorted(d.glob(f"{prefix}_*.json")) if d.exists() else []


def _load(f: Path) -> list[dict]:
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def inference_failures() -> dict:
    """Summary (ids + error class) of inference-failed records."""
    def build():
        ids, errors, total = set(), Counter(), 0
        for f in _failed_files("inference_failed"):
            for r in _load(f):
                total += 1
                ids.add(r.get("review_id"))
                msg = str(r.get("error_message", "")).lower()
                if "not found" in msg:
                    errors["Model not found"] += 1
                elif "connect" in msg:
                    errors["Ollama unreachable"] += 1
                elif "timed out" in msg:
                    errors["Timeout"] += 1
                else:
                    errors["Other"] += 1
        return {"total": total, "ids": ids, "errors": errors}

    return _cached("inf_fail", _sig(PATHS.failed_data_dir, "inference_failed_*.json"), build)


VAL_CATS = [
    ("Missing required columns", "Missing columns"),
    ("Invalid review_id", "Invalid review ID"),
    ("Invalid business_id", "Invalid business ID"),
    ("Invalid stars", "Invalid rating"),
    ("Missing or empty review text", "Missing text"),
    ("Unparseable date", "Invalid date"),
]


def _category(msg: str) -> str:
    for pre, label in VAL_CATS:
        if msg.startswith(pre):
            return label
    return "Other"


def _batch_no(f: Path):
    m = re.search(r"batch_(\d+)", f.name)
    return int(m.group(1)) if m else None


def validation_failures() -> dict:
    def build():
        records, cats, per_batch = [], Counter(), {}
        for f in _failed_files("validation_failed"):
            bid = _batch_no(f)
            for r in _load(f):
                errs = r.get("validation_errors") or []
                for e in errs:
                    cats[_category(str(e))] += 1
                per_batch[bid] = per_batch.get(bid, 0) + 1
                records.append({"kind": "invalid", "batch_id": bid, "review_id": r.get("review_id"),
                                "business_id": r.get("business_id"), "reason": "; ".join(map(str, errs)),
                                "source": r})
        return {"records": records, "cats": cats, "per_batch": per_batch}

    return _cached("val_fail", _sig(PATHS.failed_data_dir, "validation_failed_*.json"), build)


def duplicates() -> dict:
    def build():
        records, per_batch = [], {}
        for f in _failed_files("duplicates"):
            bid = _batch_no(f)
            for r in _load(f):
                per_batch[bid] = per_batch.get(bid, 0) + 1
                records.append({"kind": "duplicate", "batch_id": bid, "review_id": r.get("review_id"),
                                "business_id": r.get("business_id"), "reason": "Duplicate review_id", "source": r})
        return {"records": records, "per_batch": per_batch}

    return _cached("dups", _sig(PATHS.failed_data_dir, "duplicates_*.json"), build)


def quarantine_page(kind: str, page: int, size: int) -> dict:
    """kind: invalid | duplicate | inference. Inference files are read lazily per page."""
    skip = (page - 1) * size
    if kind == "invalid":
        items = validation_failures()["records"]
        return {"total": len(items), "page": page, "size": size, "items": items[skip:skip + size]}
    if kind == "duplicate":
        items = duplicates()["records"]
        return {"total": len(items), "page": page, "size": size, "items": items[skip:skip + size]}
    out, seen = [], 0
    for f in _failed_files("inference_failed"):
        recs = _load(f)
        if seen + len(recs) <= skip:
            seen += len(recs)
            continue
        for r in recs[max(0, skip - seen):]:
            out.append({"kind": "inference", "batch_id": r.get("batch_id"), "review_id": r.get("review_id"),
                        "business_id": None, "reason": str(r.get("error_message", ""))[:300], "source": r})
            if len(out) >= size:
                break
        seen += len(recs)
        if len(out) >= size:
            break
    return {"total": inference_failures()["total"], "page": page, "size": size, "items": out}
