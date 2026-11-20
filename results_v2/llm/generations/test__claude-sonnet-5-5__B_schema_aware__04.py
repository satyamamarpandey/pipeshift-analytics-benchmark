import json
import math

import numpy as np
import pandas as pd

NY = "America/New_York"
REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
CONTRACT_COLS = [
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _to_local_naive(values, name):
    """Parse to naive America/New_York local timestamps."""
    s = values
    try:
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            return s.dt.tz_convert(NY).dt.tz_localize(None).astype("datetime64[ns]")
        if pd.api.types.is_datetime64_any_dtype(s):
            return s.astype("datetime64[ns]")
        if pd.api.types.is_numeric_dtype(s):
            raise _Abstain(f"{name}: numeric timestamps are ambiguous")
        # strings / objects
        try:
            p = pd.to_datetime(s, errors="raise")
            if isinstance(p.dtype, pd.DatetimeTZDtype):
                return p.dt.tz_convert(NY).dt.tz_localize(None).astype("datetime64[ns]")
            return p.astype("datetime64[ns]")
        except (ValueError, TypeError):
            p = pd.to_datetime(s, errors="raise", utc=True)
            return p.dt.tz_convert(NY).dt.tz_localize(None).astype("datetime64[ns]")
    except _Abstain:
        raise
    except Exception as e:
        raise _Abstain(f"{name}: cannot parse timestamps ({type(e).__name__})")


def _to_numeric(s, name):
    try:
        out = pd.to_numeric(s, errors="raise")
    except Exception:
        raise _Abstain(f"{name}: non-numeric values")
    out = out.astype("float64")
    if not np.isfinite(out.to_numpy()).all():
        raise _Abstain(f"{name}: non-finite values")
    return out


def _parse_period(manifest):
    rp = manifest.get("reporting_period")
    if rp is None:
        raise _Abstain("manifest has no reporting_period")
    if isinstance(rp, dict):
        start = rp.get("start", rp.get("from"))
        end = rp.get("end", rp.get("to"))
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        start, end = rp
    else:
        raise _Abstain("unrecognized reporting_period format")
    if start is None or end is None:
        raise _Abstain("incomplete reporting_period")
    try:
        res = []
        for v in (start, end):
            t = pd.Timestamp(v)
            if t.tzinfo is not None:
                t = t.tz_convert(NY).tz_localize(None)
            res.append(t)
    except Exception:
        raise _Abstain("unparseable reporting_period")
    if not res[0] < res[1]:
        raise _Abstain("invalid reporting_period")
    return res[0], res[1]


def _analyze(data_path, manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise _Abstain("manifest is not an object")
    start, end = _parse_period(manifest)

    df = pd.read_parquet(data_path)

    # renames
    changes = manifest.get("schema_changes") or []
    for ch in changes:
        if not isinstance(ch, dict) or ch.get("type") != "rename":
            continue
        src, dst = ch.get("from"), ch.get("to")
        if src is None or dst is None:
            continue
        if dst in df.columns:
            if src in df.columns and src != dst:
                raise _Abstain(f"rename conflict for {dst}")
            continue
        if src in df.columns:
            df = df.rename(columns={src: dst})

    if df.columns.duplicated().any():
        raise _Abstain("duplicate column names")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise _Abstain("missing required columns: " + ",".join(missing))

    cols = [c for c in CONTRACT_COLS if c in df.columns]
    d = df[cols].copy()

    for c in REQUIRED:
        if d[c].isna().any():
            raise _Abstain(f"null values in required column {c}")

    # normalize types
    d["tpep_pickup_datetime"] = _to_local_naive(d["tpep_pickup_datetime"], "tpep_pickup_datetime")
    if d["tpep_pickup_datetime"].isna().any():
        raise _Abstain("null pickup after parsing")
    if "tpep_dropoff_datetime" in d.columns:
        try:
            d["tpep_dropoff_datetime"] = _to_local_naive(d["tpep_dropoff_datetime"], "tpep_dropoff_datetime")
        except _Abstain:
            d = d.drop(columns=["tpep_dropoff_datetime"])
    for c in ["record_id", "trip_distance", "payment_type", "fare_amount",
              "VendorID", "PULocationID", "DOLocationID", "tip_amount", "total_amount"]:
        if c in d.columns:
            if c in REQUIRED:
                d[c] = _to_numeric(d[c], c)
            else:
                d[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")

    for c in ("record_id", "payment_type"):
        if not (d[c] == np.floor(d[c])).all():
            raise _Abstain(f"{c}: non-integer values")

    # dedupe: exact replays collapse, differing values with same id conflict
    d = d.drop_duplicates()
    if d["record_id"].duplicated().any():
        raise _Abstain("conflicting records share a record_id")

    if len(d) == 0:
        raise _Abstain("no records")

    # validity
    pk = d["tpep_pickup_datetime"]
    if ((pk < start) | (pk >= end)).any():
        raise _Abstain("pickup outside reporting period")
    if not d["payment_type"].isin([0, 1, 2, 3, 4, 5, 6]).all():
        raise _Abstain("invalid payment_type codes")
    td = d["trip_distance"]
    if ((td <= 0) | (td > 100)).any():
        raise _Abstain("trip_distance outside valid range")
    fa = d["fare_amount"]
    if ((fa < 0) | (fa > 1000)).any():
        raise _Abstain("fare_amount outside valid range")

    n = int(len(d))
    total_fare = math.fsum(fa.tolist())
    mean_fare = total_fare / n
    mean_dist = math.fsum(td.tolist()) / n
    card = int((d["payment_type"] == 1).sum()) / n

    return {
        "status": "ok",
        "metrics": {
            "trip_count": n,
            "total_fare": float(total_fare),
            "mean_fare": float(mean_fare),
            "mean_trip_distance": float(mean_dist),
            "card_share": float(card),
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except _Abstain as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain(f"could not process inputs: {type(e).__name__}")
