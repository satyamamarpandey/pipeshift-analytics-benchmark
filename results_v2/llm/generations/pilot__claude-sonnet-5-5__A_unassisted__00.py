import hashlib
import json
import math

import pandas as pd

REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance",
            "payment_type", "fare_amount"]

ROW_KEYS = {"row_count", "num_rows", "rows", "record_count", "n_rows", "total_rows", "trip_count"}
HASH_KEYS = {"sha256", "checksum_sha256", "file_sha256", "checksum"}
PERIOD_KEYS = {"month", "period", "year_month", "snapshot_month"}
DIST_UNIT_KEYS = {"distance_unit", "trip_distance_unit", "trip_distance_units"}
CURRENCY_KEYS = {"currency", "fare_currency"}


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k).lower(), v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _find(manifest, keys):
    return [v for k, v in _walk(manifest) if k in keys]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return _abstain(f"manifest unreadable: {type(e).__name__}")

    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        return _abstain(f"data unreadable: {type(e).__name__}")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return _abstain("missing required columns: " + ",".join(missing))

    raw_n = len(df)
    if raw_n == 0:
        return _abstain("no records")

    # Manifest consistency checks
    for v in _find(manifest, ROW_KEYS):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if int(v) != raw_n:
                return _abstain("row count does not match manifest")
            break
    for v in _find(manifest, HASH_KEYS):
        if isinstance(v, str) and len(v) == 64:
            try:
                if _sha256(data_path).lower() != v.lower():
                    return _abstain("checksum mismatch with manifest")
            except Exception:
                return _abstain("cannot verify checksum")
            break
    for v in _find(manifest, DIST_UNIT_KEYS):
        if isinstance(v, str) and v.strip().lower() not in ("mile", "miles", "mi"):
            return _abstain("trip distance unit is not miles")
    for v in _find(manifest, CURRENCY_KEYS):
        if isinstance(v, str) and v.strip().upper() != "USD":
            return _abstain("fare currency is not USD")

    # Deduplicate
    df = df.drop_duplicates()
    if df["record_id"].notna().any():
        dup = df["record_id"].duplicated(keep="first") & df["record_id"].notna()
        df = df[~dup]

    # Parse fields
    pickup = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    fare = pd.to_numeric(df["fare_amount"], errors="coerce")
    dist = pd.to_numeric(df["trip_distance"], errors="coerce")
    pay = pd.to_numeric(df["payment_type"], errors="coerce")

    valid = fare.notna() & dist.notna() & pickup.notna() & (fare >= 0) & (dist >= 0)
    valid &= fare.map(lambda x: math.isfinite(x) if pd.notna(x) else False)
    valid &= dist.map(lambda x: math.isfinite(x) if pd.notna(x) else False)

    # Restrict to manifest period if stated
    for v in _find(manifest, PERIOD_KEYS):
        if isinstance(v, str):
            try:
                p = pd.Period(v.strip()[:7], freq="M")
            except Exception:
                continue
            inperiod = (pickup >= p.start_time) & (pickup <= p.end_time)
            valid &= inperiod
            break

    kept = int(valid.sum())
    if kept == 0:
        return _abstain("no valid records")
    if kept < 0.9 * raw_n:
        return _abstain("too many invalid or duplicate records")

    fare = fare[valid]
    dist = dist[valid]
    pay = pay[valid]

    n = int(kept)
    total_fare = float(fare.sum())
    metrics = {
        "trip_count": n,
        "total_fare": total_fare,
        "mean_fare": total_fare / n,
        "mean_trip_distance": float(dist.mean()),
        "card_share": float((pay == 1).sum()) / n,
    }
    return {"status": "ok", "metrics": metrics, "reason": ""}
