import hashlib
import json
import math
import re

import pandas as pd


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _flatten(obj, out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                _flatten(v, out)
            out.setdefault(str(k).lower(), v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                _flatten(v, out)
    return out


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
        return _abstain("manifest unreadable: %s" % type(e).__name__)
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        return _abstain("data unreadable: %s" % type(e).__name__)

    flat = _flatten(manifest)

    # Integrity checks against the manifest
    for key in ("sha256", "md5", "sha1"):
        exp = flat.get(key)
        if isinstance(exp, str) and exp:
            try:
                if _file_hash(data_path, key).lower() != exp.strip().lower():
                    return _abstain("%s checksum mismatch with manifest" % key)
            except Exception:
                return _abstain("could not verify checksum")
    for key in ("row_count", "num_rows", "rows", "n_rows", "record_count",
                "total_rows", "total_records"):
        exp = flat.get(key)
        if isinstance(exp, (int, float)) and not isinstance(exp, bool):
            if int(exp) != len(df):
                return _abstain("row count %d differs from manifest %d" % (len(df), int(exp)))
            break

    # Unit / currency sanity
    for key in ("fare_unit", "fare_units", "currency", "unit", "units", "fare_amount_unit"):
        v = flat.get(key)
        if isinstance(v, str) and v.strip():
            s = v.strip().lower()
            if s not in ("usd", "$", "dollar", "dollars", "us dollar", "miles", "mile", "mi"):
                if "cent" in s or (key.startswith(("fare", "currency")) and s != "usd"):
                    return _abstain("unexpected unit in manifest: %s" % v)

    required = ["record_id", "tpep_pickup_datetime", "trip_distance",
                "payment_type", "fare_amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _abstain("missing columns: %s" % ",".join(missing))
    if len(df) == 0:
        return _abstain("no records")

    df = df.copy()
    # Duplicates
    df = df.drop_duplicates(subset="record_id", keep="first")

    # Period filter
    pick = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    period = None
    for key in ("month", "period", "year_month", "reporting_month", "data_month"):
        v = flat.get(key)
        if isinstance(v, str):
            m = re.search(r"(\d{4})-(\d{2})", v)
            if m:
                period = (int(m.group(1)), int(m.group(2)))
                break
    if period is not None:
        mask = (pick.dt.year == period[0]) & (pick.dt.month == period[1])
        df = df[mask.values]
        pick = pick[mask.values]
        if len(df) == 0:
            return _abstain("no records in manifest period")

    fare = pd.to_numeric(df["fare_amount"], errors="coerce")
    dist = pd.to_numeric(df["trip_distance"], errors="coerce")
    pay = pd.to_numeric(df["payment_type"], errors="coerce")
    n_before = len(df)
    valid = fare.notna() & dist.notna() & (fare >= 0) & (dist >= 0)
    fare, dist, pay = fare[valid], dist[valid], pay[valid]
    n = len(fare)
    if n == 0:
        return _abstain("no valid records")
    if (n_before - n) / n_before > 0.2:
        return _abstain("too many invalid records")

    total = float(fare.sum())
    mean_fare = float(fare.mean())
    mean_dist = float(dist.mean())
    card = float((pay == 1).sum()) / n
    for x in (total, mean_fare, mean_dist, card):
        if not math.isfinite(x):
            return _abstain("non-finite metric")

    return {
        "status": "ok",
        "metrics": {
            "trip_count": int(n),
            "total_fare": total,
            "mean_fare": mean_fare,
            "mean_trip_distance": mean_dist,
            "card_share": card,
        },
        "reason": "",
    }
