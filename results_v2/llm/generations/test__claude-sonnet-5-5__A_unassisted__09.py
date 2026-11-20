import json
import hashlib
import math


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _find(d, keys):
    """Search nested dict for first matching key (case-insensitive)."""
    keys = {k.lower() for k in keys}
    stack = [d]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(k, str) and k.lower() in keys and not isinstance(v, (dict, list)):
                    return v
            stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            stack.extend(v for v in cur if isinstance(v, (dict, list)))
    return None


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        import pandas as pd
        import numpy as np

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return _abstain("manifest unreadable: %s" % type(e).__name__)
        if not isinstance(manifest, dict):
            return _abstain("manifest has unexpected structure")

        # Checksum verification
        for name, algo in (("sha256", "sha256"), ("md5", "md5"), ("sha1", "sha1")):
            expected = _find(manifest, [name])
            if isinstance(expected, str) and expected:
                h = hashlib.new(algo)
                with open(data_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest().lower() != expected.strip().lower():
                    return _abstain("data file checksum does not match manifest")

        # Currency / unit checks
        cur = _find(manifest, ["currency", "fare_currency"])
        if isinstance(cur, str) and cur.strip().upper() not in ("USD", "US$", "$"):
            return _abstain("fare currency is not USD")
        dist_scale = 1.0
        unit = _find(manifest, ["distance_unit", "trip_distance_unit", "distance_units"])
        if isinstance(unit, str):
            u = unit.strip().lower()
            if u in ("mile", "miles", "mi"):
                pass
            elif u in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
                dist_scale = 0.621371192237334
            else:
                return _abstain("unknown distance unit in manifest")

        try:
            df = pd.read_parquet(data_path)
        except Exception as e:
            return _abstain("data unreadable: %s" % type(e).__name__)

        required = ["record_id", "tpep_pickup_datetime", "trip_distance",
                    "payment_type", "fare_amount"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return _abstain("missing columns: %s" % ",".join(missing))

        # Row count check against manifest
        exp_rows = _find(manifest, ["row_count", "num_rows", "rows", "record_count", "n_rows"])
        if isinstance(exp_rows, (int, float)) and not isinstance(exp_rows, bool):
            if int(exp_rows) != len(df):
                return _abstain("row count differs from manifest")

        # Deduplicate on record_id
        if df["record_id"].isna().any():
            df = df[df["record_id"].notna()]
        df = df.drop_duplicates(subset="record_id", keep="first")

        # Restrict to the manifest's month if available
        pick = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        period = _find(manifest, ["month", "period", "year_month", "snapshot_month"])
        if isinstance(period, str):
            try:
                p = pd.Period(period.strip()[:7], freq="M")
                mask = (pick >= p.start_time) & (pick <= p.end_time)
                df = df[mask.values]
            except Exception:
                pass

        n = len(df)
        if n == 0:
            return _abstain("no trips after cleaning")

        fare = pd.to_numeric(df["fare_amount"], errors="coerce")
        dist = pd.to_numeric(df["trip_distance"], errors="coerce") * dist_scale
        pay = pd.to_numeric(df["payment_type"], errors="coerce")

        if fare.isna().mean() > 0.05 or dist.isna().mean() > 0.05:
            return _abstain("too many missing fare or distance values")

        total_fare = float(fare.sum())
        mean_fare = float(fare.mean())
        mean_dist = float(dist.mean())
        card_share = float((pay == 1).sum()) / n

        vals = [total_fare, mean_fare, mean_dist, card_share]
        if any(not math.isfinite(v) for v in vals):
            return _abstain("non-finite metric computed")

        return {
            "status": "ok",
            "metrics": {
                "trip_count": int(n),
                "total_fare": total_fare,
                "mean_fare": mean_fare,
                "mean_trip_distance": mean_dist,
                "card_share": card_share,
            },
            "reason": "",
        }
    except Exception as e:
        return _abstain("unexpected error: %s" % type(e).__name__)
