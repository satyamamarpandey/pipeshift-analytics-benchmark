import json
import math

import numpy as np
import pandas as pd

NY = "America/New_York"

REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
CONTRACT_NUMERIC = ["record_id", "VendorID", "PULocationID", "DOLocationID", "trip_distance",
                    "payment_type", "fare_amount", "tip_amount", "total_amount"]
CONTRACT_DT = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]


class Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": str(reason)}


def _to_local_naive_series(s, name):
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(NY).dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        raise Abstain(f"{name} has ambiguous numeric datetime encoding")
    nonnull = s.notna()
    try:
        out = pd.to_datetime(s, errors="raise")
        if isinstance(out.dtype, pd.DatetimeTZDtype):
            return out.dt.tz_convert(NY).dt.tz_localize(None)
        return out
    except Exception:
        pass
    try:
        out = pd.to_datetime(s, errors="raise", utc=True)
    except Exception:
        raise Abstain(f"{name} cannot be parsed as datetime")
    # mixed naive/aware strings would be ambiguous
    strs = s[nonnull].astype(str)
    has_tz = strs.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})\s*$", regex=True)
    if not has_tz.all():
        raise Abstain(f"{name} mixes naive and timezone-aware values")
    return out.dt.tz_convert(NY).dt.tz_localize(None)


def _to_numeric(s, name):
    if pd.api.types.is_bool_dtype(s):
        raise Abstain(f"{name} has boolean values")
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
    else:
        out = pd.to_numeric(s.astype(object).where(s.notna(), None), errors="coerce")
    bad = out.isna() & s.notna()
    if bad.any():
        raise Abstain(f"{name} contains non-numeric values")
    return out.astype("float64")


def _parse_ts(v):
    t = pd.Timestamp(v)
    if t.tzinfo is not None:
        t = t.tz_convert(NY).tz_localize(None)
    return t


def _period(manifest):
    rp = manifest.get("reporting_period")
    if rp is None:
        raise Abstain("manifest lacks reporting_period")
    if isinstance(rp, dict):
        s = rp.get("start", rp.get("from"))
        e = rp.get("end", rp.get("to"))
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        s, e = rp
    else:
        raise Abstain("unrecognized reporting_period format")
    if s is None or e is None:
        raise Abstain("incomplete reporting_period")
    s, e = _parse_ts(s), _parse_ts(e)
    if not s < e:
        raise Abstain("invalid reporting_period")
    return s, e


def _analyze(data_path, manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise Abstain("manifest is not an object")
    start, end = _period(manifest)

    df = pd.read_parquet(data_path)

    for key in ("row_count", "num_rows", "rows"):
        if key in manifest and isinstance(manifest[key], (int, float)) and not isinstance(manifest[key], bool):
            if int(manifest[key]) != len(df):
                raise Abstain("row count does not match manifest")
            break

    # renames
    changes = manifest.get("schema_changes") or []
    for ch in changes:
        if not isinstance(ch, dict) or ch.get("type") != "rename":
            continue
        old, new = ch.get("from"), ch.get("to")
        if old is None or new is None:
            continue
        if new in df.columns:
            if old in df.columns:
                raise Abstain(f"both {old} and {new} present")
            df = df.rename(columns={new: old})
    if df.columns.duplicated().any():
        raise Abstain("duplicate column names")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise Abstain("missing required columns: " + ",".join(missing))

    work = pd.DataFrame(index=df.index)
    for c in CONTRACT_NUMERIC:
        if c in df.columns:
            work[c] = _to_numeric(df[c], c)
    for c in CONTRACT_DT:
        if c in df.columns:
            work[c] = _to_local_naive_series(df[c], c)

    if work[REQUIRED].isna().any().any():
        raise Abstain("nulls in required columns")

    for c in ("record_id", "payment_type"):
        v = work[c]
        if not np.isfinite(v).all() or (v != np.floor(v)).any():
            raise Abstain(f"{c} has non-integer values")

    # dedup / conflicts
    work = work.drop_duplicates()
    if work["record_id"].duplicated().any():
        raise Abstain("conflicting records share a record_id")

    if len(work) == 0:
        raise Abstain("no records")

    # validity
    d = work["trip_distance"]
    f = work["fare_amount"]
    if not (np.isfinite(d).all() and np.isfinite(f).all()):
        raise Abstain("non-finite values")
    if ((d <= 0) | (d > 100)).any():
        raise Abstain("trip_distance outside valid range")
    if ((f < 0) | (f > 1000)).any():
        raise Abstain("fare_amount outside valid range")
    if (~work["payment_type"].isin([0, 1, 2, 3, 4, 5, 6])).any():
        raise Abstain("invalid payment_type code")

    pu = work["tpep_pickup_datetime"]
    if ((pu < start) | (pu >= end)).any():
        raise Abstain("pickup outside reporting period")

    n = int(len(work))
    total = math.fsum(f.tolist())
    mean_fare = total / n
    mean_dist = math.fsum(d.tolist()) / n
    card = int((work["payment_type"] == 1).sum()) / n
    return {
        "status": "ok",
        "metrics": {
            "trip_count": n,
            "total_fare": float(total),
            "mean_fare": float(mean_fare),
            "mean_trip_distance": float(mean_dist),
            "card_share": float(card),
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except Abstain as e:
        return _abstain(e)
    except Exception as e:
        return _abstain(f"could not process input: {type(e).__name__}: {e}")
