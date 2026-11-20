import hashlib
import json
import os

import numpy as np
import pandas as pd


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _find(d, names):
    """Case-insensitive search for the first matching key, recursing into dicts."""
    if not isinstance(d, dict):
        return None
    low = {str(k).lower(): v for k, v in d.items()}
    for n in names:
        if n in low:
            return low[n]
    for v in d.values():
        if isinstance(v, dict):
            r = _find(v, names)
            if r is not None:
                return r
    return None


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return _abstain(f"manifest unreadable: {type(e).__name__}")
    if not isinstance(manifest, dict):
        return _abstain("manifest is not a JSON object")

    # Manifest integrity: status flags
    st = _find(manifest, ["status", "state", "snapshot_status"])
    if isinstance(st, str) and st.lower() in (
        "partial", "incomplete", "truncated", "corrupt", "failed", "draft", "invalid"
    ):
        return _abstain(f"manifest status is '{st}'")
    for k in ("partial", "truncated", "incomplete", "corrupt"):
        if _find(manifest, [k]) is True:
            return _abstain(f"manifest flags data as {k}")

    # Checksum verification
    try:
        for key, algo in (("sha256", "sha256"), ("md5", "md5"), ("sha1", "sha1")):
            exp = _find(manifest, [key, f"{key}_checksum", f"data_{key}"])
            if isinstance(exp, str) and exp:
                h = hashlib.new(algo)
                with open(data_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest().lower() != exp.strip().lower():
                    return _abstain("data checksum does not match manifest")
        exp_size = _find(manifest, ["size_bytes", "file_size", "bytes"])
        if isinstance(exp_size, (int, float)) and not isinstance(exp_size, bool):
            if os.path.getsize(data_path) != int(exp_size):
                return _abstain("data file size does not match manifest")
    except Exception as e:
        return _abstain(f"checksum verification failed: {type(e).__name__}")

    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        return _abstain(f"data unreadable: {type(e).__name__}")

    required = ["record_id", "tpep_pickup_datetime", "tpep_dropoff_datetime",
                "trip_distance", "payment_type", "fare_amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _abstain(f"missing columns: {missing}")

    # Row count vs manifest
    exp_rows = _find(manifest, ["row_count", "rows", "num_rows", "record_count", "n_rows", "trip_count"])
    if isinstance(exp_rows, (int, float)) and not isinstance(exp_rows, bool):
        if len(df) != int(exp_rows):
            return _abstain(f"row count {len(df)} differs from manifest {int(exp_rows)}")

    # Units / encoding hints that we cannot safely interpret
    unit = _find(manifest, ["fare_unit", "fare_units", "currency_unit", "currency"])
    if isinstance(unit, str) and unit.lower() not in ("usd", "dollars", "dollar", "$"):
        return _abstain(f"unsupported fare unit '{unit}'")
    dunit = _find(manifest, ["distance_unit", "distance_units", "trip_distance_unit"])
    if isinstance(dunit, str) and dunit.lower() not in ("mile", "miles", "mi"):
        return _abstain(f"unsupported distance unit '{dunit}'")

    if len(df) == 0:
        return _abstain("no records")

    # Duplicates
    if df["record_id"].isna().any():
        return _abstain("null record_id values")
    dup = df["record_id"].duplicated(keep=False)
    if dup.any():
        if df.duplicated(keep=False).sum() != dup.sum():
            return _abstain("conflicting duplicate record_ids")
        df = df.drop_duplicates(keep="first")

    n0 = len(df)
    fare = pd.to_numeric(df["fare_amount"], errors="coerce")
    dist = pd.to_numeric(df["trip_distance"], errors="coerce")
    pay = pd.to_numeric(df["payment_type"], errors="coerce")
    pu = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    do = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

    bad = (
        fare.isna() | dist.isna() | pay.isna() | pu.isna() | do.isna()
        | ~np.isfinite(fare.fillna(np.nan).astype(float))
        | ~np.isfinite(dist.astype(float))
        | (fare < 0) | (dist < 0) | (do < pu)
    )

    # Period check against manifest, if given (e.g. "2024-01")
    period = _find(manifest, ["month", "period", "year_month"])
    if isinstance(period, str):
        try:
            p = pd.Period(period[:7], freq="M")
            bad = bad | (pu.dt.to_period("M") != p)
        except Exception:
            pass

    if bad.sum() / n0 > 0.01:
        return _abstain(f"{int(bad.sum())} of {n0} records invalid or out of period")
    keep = ~bad
    fare, dist, pay = fare[keep], dist[keep], pay[keep]
    if len(fare) == 0:
        return _abstain("no valid records")

    n = int(len(fare))
    total = float(fare.sum())
    return {
        "status": "ok",
        "metrics": {
            "trip_count": n,
            "total_fare": total,
            "mean_fare": total / n,
            "mean_trip_distance": float(dist.mean()),
            "card_share": float((pay == 1).sum() / n),
        },
        "reason": "",
    }
