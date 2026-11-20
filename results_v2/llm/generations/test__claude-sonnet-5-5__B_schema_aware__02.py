import json
import math

import pandas as pd

CONTRACT_COLS = [
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]
REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance",
            "payment_type", "fare_amount"]
INT_COLS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"]
FLOAT_COLS = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
DT_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
TZ = "America/New_York"


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _to_ny_naive(ts):
    """Convert a scalar Timestamp to naive NY local time."""
    if ts is pd.NaT or ts is None:
        return pd.NaT
    if ts.tzinfo is not None:
        return ts.tz_convert(TZ).tz_localize(None)
    return ts


def _parse_dt(series):
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(TZ).dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        raise _Abstain("numeric datetime encoding is ambiguous")
    try:
        res = pd.to_datetime(series, errors="coerce", format="ISO8601")
        if isinstance(res.dtype, pd.DatetimeTZDtype):
            res = res.dt.tz_convert(TZ).dt.tz_localize(None)
        elif res.dtype == object:
            raise ValueError
    except Exception:
        # mixed formats / offsets: parse element-wise
        vals = []
        for v in series:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                vals.append(pd.NaT)
                continue
            try:
                vals.append(_to_ny_naive(pd.Timestamp(v)))
            except Exception:
                vals.append(pd.NaT)
        res = pd.Series(pd.to_datetime(vals), index=series.index)
    # unparsable non-null values
    bad = res.isna() & series.notna()
    if bool(bad.any()):
        raise _Abstain("unparsable datetime values")
    return res.astype("datetime64[ns]")


def _parse_num(series, name):
    num = pd.to_numeric(series, errors="coerce")
    bad = num.isna() & series.notna()
    if bool(bad.any()):
        raise _Abstain("non-numeric values in " + name)
    num = num.astype("float64")
    if bool(((num == float("inf")) | (num == float("-inf"))).any()):
        raise _Abstain("infinite values in " + name)
    return num


def _period(manifest):
    rp = manifest.get("reporting_period")
    if isinstance(rp, dict):
        s, e = rp.get("start"), rp.get("end")
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        s, e = rp
    else:
        raise _Abstain("manifest lacks reporting_period")
    if s is None or e is None:
        raise _Abstain("manifest reporting_period incomplete")
    try:
        s = _to_ny_naive(pd.Timestamp(s))
        e = _to_ny_naive(pd.Timestamp(e))
    except Exception:
        raise _Abstain("unparsable reporting_period")
    if pd.isna(s) or pd.isna(e) or not s < e:
        raise _Abstain("invalid reporting_period")
    return s, e


def _run(data_path, manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        raise _Abstain("cannot read manifest")
    if not isinstance(manifest, dict):
        raise _Abstain("manifest malformed")
    start, end = _period(manifest)

    try:
        df = pd.read_parquet(data_path)
    except Exception:
        raise _Abstain("cannot read data file")

    # renames (apply sequentially, only when source present)
    changes = manifest.get("schema_changes") or []
    if isinstance(changes, list):
        for ch in changes:
            if isinstance(ch, dict) and ch.get("type") == "rename":
                src, dst = ch.get("from"), ch.get("to")
                if src in df.columns and dst not in df.columns:
                    df = df.rename(columns={src: dst})
                elif src in df.columns and dst in df.columns and src != dst:
                    raise _Abstain("rename target already present: " + str(dst))

    if df.columns.duplicated().any():
        raise _Abstain("duplicate column names")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise _Abstain("missing required columns: " + ", ".join(missing))

    cols = [c for c in CONTRACT_COLS if c in df.columns]
    df = df[cols].copy()
    if len(df) == 0:
        raise _Abstain("no records")

    for c in DT_COLS:
        if c in df.columns:
            df[c] = _parse_dt(df[c])
    for c in INT_COLS + FLOAT_COLS:
        if c in df.columns:
            df[c] = _parse_num(df[c], c)

    for c in REQUIRED:
        if df[c].isna().any():
            raise _Abstain("null values in required column " + c)

    for c in INT_COLS:
        if c in df.columns:
            v = df[c].dropna()
            if bool((v != v.round()).any()):
                raise _Abstain("non-integer values in " + c)

    # dedupe: identical replays collapse, differing duplicates are conflicts
    df = df.drop_duplicates()
    if df["record_id"].duplicated().any():
        raise _Abstain("conflicting records share a record_id")

    # validity checks
    if not df["payment_type"].isin([0, 1, 2, 3, 4, 5, 6]).all():
        raise _Abstain("invalid payment_type codes")
    td = df["trip_distance"]
    if not ((td > 0) & (td <= 100)).all():
        raise _Abstain("trip_distance outside valid range")
    fa = df["fare_amount"]
    if not ((fa >= 0) & (fa <= 1000)).all():
        raise _Abstain("fare_amount outside valid range")
    pu = df["tpep_pickup_datetime"]
    if not ((pu >= start) & (pu < end)).all():
        raise _Abstain("pickup times outside reporting period")

    n = int(len(df))
    if n == 0:
        raise _Abstain("no records")
    total_fare = float(fa.sum())
    return {
        "status": "ok",
        "metrics": {
            "trip_count": n,
            "total_fare": total_fare,
            "mean_fare": total_fare / n,
            "mean_trip_distance": float(td.sum()) / n,
            "card_share": float((df["payment_type"] == 1).sum()) / n,
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _run(data_path, manifest_path)
    except _Abstain as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain("unexpected error: " + type(e).__name__)
