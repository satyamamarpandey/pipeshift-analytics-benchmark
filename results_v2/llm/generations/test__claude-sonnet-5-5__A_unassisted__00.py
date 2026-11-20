import hashlib
import json
import math
import re


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _flatten(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_flatten(v, f"{prefix}[{i}]"))
    else:
        out.append((prefix.lower(), obj))
    return out


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        import pandas as pd
        import numpy as np
    except Exception as e:  # pragma: no cover
        return _abstain(f"imports failed: {e}")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return _abstain(f"manifest unreadable: {type(e).__name__}")

    flat = _flatten(manifest)

    # checksum verification
    try:
        for k, v in flat:
            if isinstance(v, str) and any(t in k for t in ("sha256", "md5", "checksum")):
                v = v.strip().lower()
                algo = "md5" if ("md5" in k or len(v) == 32) else "sha256"
                if not re.fullmatch(r"[0-9a-f]{32,64}", v):
                    continue
                if (algo == "md5" and len(v) != 32) or (algo == "sha256" and len(v) != 64):
                    continue
                h = hashlib.new(algo)
                with open(data_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != v:
                    return _abstain("data checksum does not match manifest")
    except Exception as e:
        return _abstain(f"checksum verification failed: {type(e).__name__}")

    # unit / currency sanity
    for k, v in flat:
        if isinstance(v, str):
            s = v.lower()
            if any(t in k for t in ("distance", "unit")) and ("km" in s or "kilomet" in s or "meter" in s and "mile" not in s):
                return _abstain("trip_distance not in miles per manifest")
            if "currency" in k and s.strip() not in ("usd", "us$", "$", "dollar", "dollars"):
                return _abstain("fare currency is not USD per manifest")

    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        return _abstain(f"data unreadable: {type(e).__name__}")

    required = ["record_id", "tpep_pickup_datetime", "trip_distance",
                "payment_type", "fare_amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _abstain(f"missing columns: {missing}")

    raw_rows = len(df)
    if raw_rows == 0:
        return _abstain("no records")

    # expected row count
    for k, v in flat:
        if isinstance(v, (int, float)) and not isinstance(v, bool) and \
                any(t in k for t in ("row_count", "num_rows", "rows", "record_count", "n_records", "trip_count")):
            if "group" in k or "column" in k:
                continue
            if int(v) != raw_rows:
                return _abstain(f"row count {raw_rows} differs from manifest {int(v)}")
            break

    # deduplicate
    df = df.drop_duplicates()
    if df["record_id"].isna().any():
        return _abstain("null record_id values")
    if df["record_id"].duplicated().any():
        return _abstain("conflicting duplicate record_ids")

    pick = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    fare = pd.to_numeric(df["fare_amount"], errors="coerce")
    dist = pd.to_numeric(df["trip_distance"], errors="coerce")
    pay = pd.to_numeric(df["payment_type"], errors="coerce")

    # month determination
    month = None
    for k, v in flat:
        if isinstance(v, str) and any(t in k for t in ("month", "period")):
            m = re.search(r"(\d{4})-(\d{2})", v)
            if m:
                month = (int(m.group(1)), int(m.group(2)))
                break
    valid_pick = pick.dropna()
    if valid_pick.empty:
        return _abstain("no valid pickup timestamps")
    if month is None:
        mode = (valid_pick.dt.year * 100 + valid_pick.dt.month).mode().iloc[0]
        month = (int(mode // 100), int(mode % 100))
    in_month = (pick.dt.year == month[0]) & (pick.dt.month == month[1])

    good = in_month & fare.notna() & dist.notna() & (fare >= 0) & (dist >= 0)
    if "tpep_dropoff_datetime" in df.columns:
        drop = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
        good &= drop.notna() & (drop >= pick)

    n_total = len(df)
    n_good = int(good.sum())
    if n_good == 0:
        return _abstain("no valid trips")
    if n_good < 0.9 * n_total:
        return _abstain("too many invalid or out-of-period records")

    fare = fare[good].astype("float64")
    dist = dist[good].astype("float64")
    pay = pay[good]

    if not (np.isfinite(fare).all() and np.isfinite(dist).all()):
        return _abstain("non-finite values")

    n = n_good
    total_fare = float(fare.sum())
    metrics = {
        "trip_count": int(n),
        "total_fare": total_fare,
        "mean_fare": float(fare.mean()),
        "mean_trip_distance": float(dist.mean()),
        "card_share": float((pay == 1).sum() / n),
    }
    for v in metrics.values():
        if isinstance(v, float) and math.isnan(v):
            return _abstain("NaN metric")
    return {"status": "ok", "metrics": metrics, "reason": ""}
