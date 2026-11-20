import hashlib
import json
import re

import numpy as np
import pandas as pd


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _flatten(obj, out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if isinstance(v, (dict, list)):
                _flatten(v, out)
            out.setdefault(lk, v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                _flatten(v, out)
    return out


def _find(flat, keys):
    for k in keys:
        if k in flat and flat[k] not in (None, ""):
            return flat[k]
    return None


def _file_hash(path, algo):
    h = hashlib.new(algo)
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
    flat = _flatten(manifest)

    # Integrity: file checksum
    try:
        for key, algo in (("sha256", "sha256"), ("md5", "md5"), ("sha1", "sha1"),
                          ("checksum", None), ("checksum_sha256", "sha256"),
                          ("file_sha256", "sha256"), ("file_md5", "md5")):
            val = flat.get(key)
            if isinstance(val, str) and val:
                a = algo
                if a is None:
                    a = {64: "sha256", 32: "md5", 40: "sha1"}.get(len(val))
                if a and _file_hash(data_path, a).lower() != val.lower():
                    return _abstain("data file checksum does not match manifest")
    except Exception as e:
        return _abstain(f"checksum verification failed: {type(e).__name__}")

    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        return _abstain(f"data unreadable: {type(e).__name__}")

    required = ["record_id", "tpep_pickup_datetime", "trip_distance",
                "payment_type", "fare_amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _abstain(f"missing columns: {missing}")

    # Row count check
    exp_rows = _find(flat, ["row_count", "num_rows", "rows", "record_count",
                            "n_rows", "total_rows", "total_records"])
    if exp_rows is not None:
        try:
            if int(exp_rows) != len(df):
                return _abstain("row count differs from manifest")
        except Exception:
            pass

    # Currency / units
    cur = _find(flat, ["currency", "fare_currency"])
    if isinstance(cur, str) and cur.strip().upper() not in ("USD", "US$", "$"):
        return _abstain(f"fare currency is {cur}, not USD")
    dist_scale = 1.0
    unit = _find(flat, ["distance_unit", "trip_distance_unit", "distance_units"])
    if isinstance(unit, str):
        u = unit.strip().lower()
        if u in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
            dist_scale = 0.621371192237334
        elif u in ("m", "meter", "meters", "metres"):
            dist_scale = 0.000621371192237334
        elif u not in ("mile", "miles", "mi"):
            return _abstain(f"unknown distance unit {unit}")
    fare_unit = _find(flat, ["fare_unit", "fare_amount_unit"])
    if isinstance(fare_unit, str) and fare_unit.strip().lower() in ("cents", "cent"):
        fare_scale = 0.01
    else:
        fare_scale = 1.0

    # Deduplicate
    df = df.copy()
    if df["record_id"].isna().any():
        df = df[df["record_id"].notna()]
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="record_id", keep="first")

    # Coerce types
    pick = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    fare = pd.to_numeric(df["fare_amount"], errors="coerce") * fare_scale
    dist = pd.to_numeric(df["trip_distance"], errors="coerce") * dist_scale

    # Period filter from manifest
    period = _find(flat, ["month", "period", "reporting_month", "data_month",
                          "reference_month"])
    ym = None
    if period is not None:
        m = re.search(r"(\d{4})[-/]?(\d{1,2})", str(period))
        if m and 1 <= int(m.group(2)) <= 12:
            ym = (int(m.group(1)), int(m.group(2)))
    keep = pd.Series(True, index=df.index)
    if ym is not None:
        keep &= (pick.dt.year == ym[0]) & (pick.dt.month == ym[1])

    # Validity
    valid = fare.notna() & dist.notna() & (fare >= 0) & (dist >= 0) & pick.notna()
    if "tpep_dropoff_datetime" in df.columns:
        drop = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
        valid &= drop.isna() | (drop >= pick)
    valid &= np.isfinite(fare) & np.isfinite(dist)

    candidates = keep.sum()
    if candidates == 0:
        return _abstain("no trips in the reported period")
    bad = (keep & ~valid).sum()
    if bad / candidates > 0.05:
        return _abstain("too many invalid records to trust the snapshot")

    sel = keep & valid
    n = int(sel.sum())
    if n == 0:
        return _abstain("no valid trips")

    f = fare[sel]
    d = dist[sel]
    pt = df.loc[sel, "payment_type"]
    ptn = pd.to_numeric(pt, errors="coerce")
    if ptn.isna().all():
        card = pt.astype(str).str.strip().str.lower().isin(
            ["credit", "credit card", "card", "credit_card", "1"])
    else:
        card = ptn == 1
    card_share = float(card.sum()) / n

    return {
        "status": "ok",
        "metrics": {
            "trip_count": n,
            "total_fare": float(f.sum()),
            "mean_fare": float(f.mean()),
            "mean_trip_distance": float(d.mean()),
            "card_share": card_share,
        },
        "reason": "",
    }
