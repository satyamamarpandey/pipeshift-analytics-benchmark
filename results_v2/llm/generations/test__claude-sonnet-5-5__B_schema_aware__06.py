import json
import math

import numpy as np
import pandas as pd

NY = "America/New_York"
REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
CONTRACT = ["record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "PULocationID", "DOLocationID", "trip_distance", "payment_type",
            "fare_amount", "tip_amount", "total_amount"]
NUMERIC = ["record_id", "VendorID", "PULocationID", "DOLocationID", "trip_distance",
           "payment_type", "fare_amount", "tip_amount", "total_amount"]
DATETIMES = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]


class Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": str(reason)}


def _to_local_naive(s):
    """Convert a series or scalar-like to naive New York local datetime64."""
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        parsed = s
    elif pd.api.types.is_numeric_dtype(s.dtype):
        raise Abstain("numeric datetime values are ambiguous")
    else:
        try:
            parsed = pd.to_datetime(s, format="ISO8601")
        except Exception:
            try:
                parsed = pd.to_datetime(s, utc=True)
            except Exception as e:
                raise Abstain(f"unparseable datetime: {e}")
    if getattr(parsed.dtype, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(NY).dt.tz_localize(None)
    return parsed.astype("datetime64[ns]")


def _period(manifest):
    rp = manifest.get("reporting_period")
    if rp is None:
        raise Abstain("manifest has no reporting_period")
    if isinstance(rp, dict):
        start, end = rp.get("start"), rp.get("end")
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        start, end = rp
    else:
        raise Abstain("unrecognized reporting_period")
    if start is None or end is None:
        raise Abstain("incomplete reporting_period")
    try:
        st = _to_local_naive(pd.Series([start])).iloc[0]
        en = _to_local_naive(pd.Series([end])).iloc[0]
    except Abstain:
        raise
    except Exception as e:
        raise Abstain(f"bad reporting_period: {e}")
    if pd.isna(st) or pd.isna(en) or st >= en:
        raise Abstain("invalid reporting_period")
    return st, en


def _analyze(data_path, manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise Abstain(f"cannot read manifest: {e}")
    if not isinstance(manifest, dict):
        raise Abstain("manifest is not an object")

    start, end = _period(manifest)

    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        raise Abstain(f"cannot read data: {e}")
    raw_rows = len(df)

    # renames
    changes = manifest.get("schema_changes") or []
    if not isinstance(changes, list):
        raise Abstain("invalid schema_changes")
    for ch in changes:
        if not isinstance(ch, dict) or ch.get("type") != "rename":
            continue
        src, dst = ch.get("from"), ch.get("to")
        if src is None or dst is None:
            continue
        if dst in df.columns and src in df.columns and src != dst:
            raise Abstain(f"rename conflict: both {src} and {dst} present")
        if src in df.columns and src != dst:
            df = df.rename(columns={src: dst})
    if df.columns.duplicated().any():
        raise Abstain("duplicate column names")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise Abstain(f"missing required columns: {missing}")

    cols = [c for c in CONTRACT if c in df.columns]
    df = df[cols].copy()

    for c in NUMERIC:
        if c in df.columns:
            s = df[c]
            if not pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype):
                try:
                    if pd.api.types.is_bool_dtype(s.dtype):
                        raise ValueError("boolean")
                    s = pd.to_numeric(s, errors="raise")
                except Exception:
                    raise Abstain(f"column {c} not numeric")
            df[c] = s.astype("float64") if c != "record_id" else s
    for c in DATETIMES:
        if c in df.columns:
            df[c] = _to_local_naive(df[c])

    # nulls in required
    for c in REQUIRED:
        if df[c].isna().any():
            raise Abstain(f"null values in required column {c}")

    # record_id numeric: keep exact; ensure integral
    rid = df["record_id"]
    if pd.api.types.is_float_dtype(rid.dtype):
        if not np.isfinite(rid).all() or (rid != np.floor(rid)).any():
            raise Abstain("record_id not integral")
        df["record_id"] = rid.astype("int64")

    for c in ["trip_distance", "fare_amount", "payment_type"]:
        if not np.isfinite(df[c].astype("float64")).all():
            raise Abstain(f"non-finite values in {c}")

    # dedup / conflicts
    df = df.drop_duplicates()
    if df["record_id"].duplicated().any():
        raise Abstain("conflicting records share a record_id")

    # manifest row count check
    for key in ("row_count", "num_rows", "record_count"):
        v = manifest.get(key)
        if isinstance(v, int) and not isinstance(v, bool):
            if v != raw_rows and v != len(df):
                raise Abstain("row count differs from manifest")
            break

    # validity
    pt = df["payment_type"]
    if ((pt != np.floor(pt)) | (pt < 0) | (pt > 6)).any():
        raise Abstain("invalid payment_type codes")
    td = df["trip_distance"]
    if ((td <= 0) | (td > 100)).any():
        raise Abstain("trip_distance out of range")
    fa = df["fare_amount"]
    if ((fa < 0) | (fa > 1000)).any():
        raise Abstain("fare_amount out of range")
    pk = df["tpep_pickup_datetime"]
    if ((pk < start) | (pk >= end)).any():
        raise Abstain("pickup outside reporting period")

    n = len(df)
    if n == 0:
        raise Abstain("no trips")

    total_fare = math.fsum(fa.tolist())
    total_dist = math.fsum(td.tolist())
    card = int((pt == 1).sum())
    return {
        "status": "ok",
        "metrics": {
            "trip_count": int(n),
            "total_fare": float(total_fare),
            "mean_fare": float(total_fare / n),
            "mean_trip_distance": float(total_dist / n),
            "card_share": float(card / n),
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except Abstain as a:
        return _abstain(a)
    except Exception as e:
        return _abstain(f"unexpected error: {e}")
