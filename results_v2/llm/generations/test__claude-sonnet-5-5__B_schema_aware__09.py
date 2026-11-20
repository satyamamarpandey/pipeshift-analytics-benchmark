import json
import numpy as np
import pandas as pd

_REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance",
             "payment_type", "fare_amount"]
_OPTIONAL = ["VendorID", "tpep_dropoff_datetime", "PULocationID",
             "DOLocationID", "tip_amount", "total_amount"]


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _to_local_naive(s):
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert("America/New_York").dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        p = pd.to_datetime(s, format="ISO8601")
    except Exception:
        try:
            p = pd.to_datetime(s, utc=True, format="ISO8601")
        except Exception:
            p = pd.to_datetime(s)
    if isinstance(p.dtype, pd.DatetimeTZDtype):
        p = p.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return p


def _ts_scalar(v):
    t = pd.Timestamp(v)
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    return t


def _numeric(s, name):
    try:
        n = pd.to_numeric(s, errors="raise")
    except Exception:
        raise _Abstain("column %s has non-numeric values" % name)
    n = n.astype("float64")
    if not np.isfinite(n.to_numpy()).all():
        raise _Abstain("column %s has non-finite values" % name)
    return n


def _period(manifest):
    rp = manifest.get("reporting_period")
    if rp is None:
        raise _Abstain("manifest lacks reporting_period")
    if isinstance(rp, dict):
        s, e = rp.get("start"), rp.get("end")
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        s, e = rp
    else:
        raise _Abstain("unrecognized reporting_period")
    if s is None or e is None:
        raise _Abstain("incomplete reporting_period")
    return _ts_scalar(s), _ts_scalar(e)


def _analyze(data_path, manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    start, end = _period(manifest)

    df = pd.read_parquet(data_path)

    renames = {}
    for ch in manifest.get("schema_changes") or []:
        if isinstance(ch, dict) and ch.get("type") == "rename":
            frm, to = ch.get("from"), ch.get("to")
            if frm and to and to in df.columns and frm not in df.columns:
                renames[to] = frm
    if renames:
        df = df.rename(columns=renames)

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise _Abstain("missing required columns: %s" % ",".join(missing))

    cols = _REQUIRED + [c for c in _OPTIONAL if c in df.columns]
    df = df[cols].copy()

    if df[_REQUIRED].isnull().any().any():
        raise _Abstain("nulls in required columns")

    df["record_id"] = _numeric(df["record_id"], "record_id")
    df["trip_distance"] = _numeric(df["trip_distance"], "trip_distance")
    df["payment_type"] = _numeric(df["payment_type"], "payment_type")
    df["fare_amount"] = _numeric(df["fare_amount"], "fare_amount")
    try:
        df["tpep_pickup_datetime"] = _to_local_naive(df["tpep_pickup_datetime"])
    except Exception:
        raise _Abstain("unparseable pickup timestamps")
    if df["tpep_pickup_datetime"].isnull().any():
        raise _Abstain("null pickup timestamps after parsing")
    if "tpep_dropoff_datetime" in df.columns:
        try:
            df["tpep_dropoff_datetime"] = _to_local_naive(df["tpep_dropoff_datetime"])
        except Exception:
            pass
    for c in ["VendorID", "PULocationID", "DOLocationID", "tip_amount", "total_amount"]:
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
            except Exception:
                pass

    # dedupe replays, detect conflicts
    df = df.drop_duplicates()
    if df["record_id"].duplicated().any():
        raise _Abstain("conflicting duplicate record_id values")

    if len(df) == 0:
        raise _Abstain("no records")

    pt = df["payment_type"]
    if ((pt % 1) != 0).any() or (~pt.isin([0, 1, 2, 3, 4, 5, 6])).any():
        raise _Abstain("invalid payment_type codes")
    d = df["trip_distance"]
    if ((d <= 0) | (d > 100)).any():
        raise _Abstain("trip_distance out of valid range")
    fa = df["fare_amount"]
    if ((fa < 0) | (fa > 1000)).any():
        raise _Abstain("fare_amount out of valid range")
    pu = df["tpep_pickup_datetime"]
    if ((pu < start) | (pu >= end)).any():
        raise _Abstain("pickups outside reporting period")

    n = len(df)
    total = float(fa.sum())
    return {
        "status": "ok",
        "metrics": {
            "trip_count": int(n),
            "total_fare": total,
            "mean_fare": float(fa.mean()),
            "mean_trip_distance": float(d.mean()),
            "card_share": float((pt == 1).sum() / n),
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except _Abstain as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain("failed to process inputs: %s" % type(e).__name__)
