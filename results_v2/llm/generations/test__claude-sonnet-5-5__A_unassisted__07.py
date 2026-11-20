import json
import hashlib

import numpy as np
import pandas as pd


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _find(d, keys):
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    for v in d.values():
        if isinstance(v, dict):
            r = _find(v, keys)
            if r is not None:
                return r
    return None


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return _abstain("manifest unreadable: %s" % type(e).__name__)

    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        return _abstain("data unreadable: %s" % type(e).__name__)

    # checksum verification if provided
    expected_hash = _find(manifest, ["sha256", "checksum_sha256", "file_sha256"])
    if isinstance(expected_hash, str) and expected_hash:
        try:
            h = hashlib.sha256()
            with open(data_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest().lower() != expected_hash.strip().lower():
                return _abstain("data checksum does not match manifest")
        except Exception:
            return _abstain("could not verify checksum")

    required = ["record_id", "tpep_pickup_datetime", "trip_distance",
                "payment_type", "fare_amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _abstain("missing columns: %s" % ",".join(missing))

    # row count check
    exp_rows = _find(manifest, ["row_count", "num_rows", "rows", "record_count", "n_rows"])
    if isinstance(exp_rows, (int, float)) and not isinstance(exp_rows, bool):
        if int(exp_rows) != len(df):
            return _abstain("row count %d differs from manifest %d" % (len(df), int(exp_rows)))

    if len(df) == 0:
        return _abstain("no records")

    n_raw = len(df)

    df = df.copy()
    df["_pickup"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["_fare"] = pd.to_numeric(df["fare_amount"], errors="coerce")
    df["_dist"] = pd.to_numeric(df["trip_distance"], errors="coerce")
    df["_pay"] = pd.to_numeric(df["payment_type"], errors="coerce")

    # dedupe by record_id
    df = df[df["record_id"].isna() | ~df["record_id"].duplicated(keep="first")]

    # target month
    period = None
    mval = _find(manifest, ["month", "period", "reference_month", "year_month", "data_month"])
    if mval is not None:
        try:
            period = pd.Period(str(mval)[:7], freq="M")
        except Exception:
            period = None
    valid_pick = df["_pickup"].notna()
    if period is None:
        if not valid_pick.any():
            return _abstain("no valid pickup timestamps")
        period = df.loc[valid_pick, "_pickup"].dt.to_period("M").mode().iloc[0]

    in_month = valid_pick & (df["_pickup"].dt.to_period("M") == period)
    ok = in_month & df["_fare"].notna() & df["_dist"].notna() \
        & (df["_fare"] >= 0) & (df["_dist"] >= 0) \
        & np.isfinite(df["_fare"]) & np.isfinite(df["_dist"])
    clean = df[ok]

    if len(clean) == 0:
        return _abstain("no valid records after cleaning")
    if len(clean) < 0.8 * n_raw:
        return _abstain("too many invalid, duplicate or out-of-period records")

    n = int(len(clean))
    total_fare = float(clean["_fare"].sum())
    metrics = {
        "trip_count": n,
        "total_fare": total_fare,
        "mean_fare": total_fare / n,
        "mean_trip_distance": float(clean["_dist"].mean()),
        "card_share": float((clean["_pay"] == 1).sum()) / n,
    }
    return {"status": "ok", "metrics": metrics, "reason": ""}
