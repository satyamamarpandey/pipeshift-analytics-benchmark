import hashlib
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


_REQUIRED_MANIFEST = ["dataset", "snapshot_id", "snapshot_version", "reporting_period",
                      "file", "file_sha256", "row_count", "columns", "units",
                      "schema_changes", "description"]
_REQUIRED_COLS = ["record_id", "tpep_pickup_datetime", "trip_distance",
                  "payment_type", "fare_amount"]
_OPT_NUM = ["tip_amount", "total_amount", "VendorID", "PULocationID", "DOLocationID"]


def _to_local_naive(s, tz):
    if pd.api.types.is_datetime64_any_dtype(s):
        res = s
    else:
        try:
            res = pd.to_datetime(s)
        except Exception:
            res = pd.to_datetime(s, utc=True)
        if res.dtype == object:
            res = pd.to_datetime(s, utc=True)
    if getattr(res.dt, "tz", None) is not None:
        res = res.dt.tz_convert(tz).dt.tz_localize(None)
    return res.astype("datetime64[ns]")


def _num(s):
    return pd.to_numeric(s, errors="raise")


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except Exception as e:  # any unexpected failure -> no trustworthy result
        return _abstain("unexpected error: %s" % type(e).__name__)


def _analyze(data_path, manifest_path):
    # 1. manifest
    try:
        with open(manifest_path, "rb") as f:
            manifest = json.loads(f.read().decode("utf-8"))
    except Exception:
        return _abstain("manifest missing or invalid JSON")
    if not isinstance(manifest, dict):
        return _abstain("manifest is not a JSON object")
    missing = [k for k in _REQUIRED_MANIFEST if k not in manifest]
    if missing:
        return _abstain("manifest missing fields: %s" % ",".join(missing))
    rp = manifest["reporting_period"]
    if not isinstance(rp, dict) or not all(k in rp for k in ("start", "end", "timezone")):
        return _abstain("manifest reporting_period invalid")
    tz = rp["timezone"] or "America/New_York"
    try:
        def _bound(v):
            t = pd.Timestamp(v)
            if t.tzinfo is not None:
                t = t.tz_convert(tz).tz_localize(None)
            return t
        start, end = _bound(rp["start"]), _bound(rp["end"])
    except Exception:
        return _abstain("reporting_period not parseable")
    if not start < end:
        return _abstain("reporting_period empty")

    # units
    units_txt = json.dumps(manifest["units"]).lower()
    for bad in ("km", "kilomet", "meter", "cent"):
        if bad in units_txt and "mile" not in units_txt.replace("kilomet", "") or \
                (bad in ("cent",) and "cent" in units_txt and "percent" not in units_txt):
            return _abstain("units not miles/USD")

    # 2. hash
    try:
        with open(data_path, "rb") as f:
            raw = f.read()
    except Exception:
        return _abstain("data file unreadable")
    if hashlib.sha256(raw).hexdigest().lower() != str(manifest["file_sha256"]).strip().lower():
        return _abstain("sha256 mismatch")

    # 3. row count
    try:
        df = pq.read_table(data_path).to_pandas()
    except Exception:
        return _abstain("cannot read parquet")
    try:
        rc = int(manifest["row_count"])
    except Exception:
        return _abstain("row_count invalid")
    if len(df) != rc:
        return _abstain("row count mismatch")

    # 4. schema
    changes = manifest["schema_changes"] or []
    rename = {}
    for c in changes:
        if isinstance(c, dict) and c.get("type") == "rename":
            rename[c["from"]] = c["to"]
    for src, dst in rename.items():
        if src in df.columns:
            if dst in df.columns:
                return _abstain("rename collision")
            df = df.rename(columns={src: dst})
    cols = set(df.columns)
    mcols = manifest["columns"]
    if not isinstance(mcols, list):
        return _abstain("manifest columns invalid")
    for c in mcols:
        n = c.get("name") if isinstance(c, dict) else c
        if n not in cols and rename.get(n, n) not in cols:
            return _abstain("delivered columns disagree with manifest: %s" % n)
    for c in _REQUIRED_COLS:
        if c not in cols:
            return _abstain("missing required column %s" % c)
    if len(df) == 0:
        return _abstain("no rows")

    # 6. contract rules (nulls)
    for c in _REQUIRED_COLS:
        if df[c].isna().any():
            return _abstain("null in required column %s" % c)

    work = pd.DataFrame(index=df.index)
    try:
        work["record_id"] = _num(df["record_id"])
        work["trip_distance"] = _num(df["trip_distance"]).astype("float64")
        work["payment_type"] = _num(df["payment_type"])
        work["fare_amount"] = _num(df["fare_amount"]).astype("float64")
        work["tpep_pickup_datetime"] = _to_local_naive(df["tpep_pickup_datetime"], tz)
    except Exception:
        return _abstain("required column not convertible to contract type")
    if work["tpep_pickup_datetime"].isna().any():
        return _abstain("unparseable pickup timestamp")
    for c in ("record_id", "payment_type"):
        v = work[c].astype("float64")
        if not np.isfinite(v).all() or (v != np.floor(v)).any():
            return _abstain("non-integer %s" % c)
        work[c] = work[c].astype("int64")
    for c in ("trip_distance", "fare_amount"):
        if not np.isfinite(work[c]).all():
            return _abstain("non-finite %s" % c)

    # 5. period
    p = work["tpep_pickup_datetime"]
    if ((p < start) | (p >= end)).any():
        return _abstain("pickup outside reporting period")

    # ranges and codes
    if not ((work["trip_distance"] > 0) & (work["trip_distance"] <= 100)).all():
        return _abstain("trip_distance out of range")
    if not ((work["fare_amount"] >= 0) & (work["fare_amount"] <= 1000)).all():
        return _abstain("fare_amount out of range")
    if not work["payment_type"].isin([0, 1, 2, 3, 4, 5, 6]).all():
        return _abstain("undocumented payment_type")

    # optional columns for duplicate comparison
    for c in _OPT_NUM:
        if c in cols:
            work[c] = pd.to_numeric(df[c], errors="coerce")
    if "tpep_dropoff_datetime" in cols:
        try:
            work["tpep_dropoff_datetime"] = _to_local_naive(df["tpep_dropoff_datetime"], tz)
        except Exception:
            work["tpep_dropoff_datetime"] = df["tpep_dropoff_datetime"].astype(str)

    work = work.drop_duplicates()
    if work["record_id"].duplicated().any():
        return _abstain("conflicting records share a record_id")

    n = len(work)
    if n == 0:
        return _abstain("no trips")
    total_fare = float(work["fare_amount"].sum())
    metrics = {
        "trip_count": int(n),
        "total_fare": total_fare,
        "mean_fare": total_fare / n,
        "mean_trip_distance": float(work["trip_distance"].mean()),
        "card_share": float((work["payment_type"] == 1).sum()) / n,
    }
    for v in metrics.values():
        if isinstance(v, float) and not math.isfinite(v):
            return _abstain("non-finite metric")
    return {"status": "ok", "metrics": metrics, "reason": ""}
