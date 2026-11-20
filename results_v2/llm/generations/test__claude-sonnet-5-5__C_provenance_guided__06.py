import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": str(reason)}


_MANIFEST_FIELDS = ["dataset", "snapshot_id", "snapshot_version", "reporting_period",
                    "file", "file_sha256", "row_count", "columns", "units",
                    "schema_changes", "description"]
_REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
_CONTRACT = ["record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
             "PULocationID", "DOLocationID", "trip_distance", "payment_type",
             "fare_amount", "tip_amount", "total_amount"]
_TZ = "America/New_York"


def _to_local_naive(s, tz):
    """Convert a series to naive local timestamps in the given timezone."""
    def fix(ser):
        if getattr(ser.dtype, "tz", None) is not None:
            return ser.dt.tz_convert(tz).dt.tz_localize(None)
        return ser

    if pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype):
        raise ValueError("numeric timestamps are ambiguous")
    try:
        out = pd.to_datetime(s)
        if pd.api.types.is_datetime64_any_dtype(out.dtype):
            return fix(out).astype("datetime64[ns]")
    except Exception:
        pass
    vals = []
    for v in s:
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            vals.append(pd.NaT)
            continue
        t = pd.Timestamp(v)
        if t.tzinfo is not None:
            t = t.tz_convert(tz).tz_localize(None)
        vals.append(t)
    return pd.Series(pd.to_datetime(vals), index=s.index).astype("datetime64[ns]")


def _to_numeric(s):
    out = pd.to_numeric(s, errors="raise")
    return out


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # 1. manifest
        if not os.path.isfile(manifest_path):
            return _abstain("manifest missing")
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                man = json.load(f)
        except Exception as e:
            return _abstain("manifest is not valid JSON: %s" % e)
        if not isinstance(man, dict):
            return _abstain("manifest is not a JSON object")
        missing = [k for k in _MANIFEST_FIELDS if k not in man]
        if missing:
            return _abstain("manifest missing fields: %s" % missing)
        rp = man["reporting_period"]
        if not isinstance(rp, dict) or "start" not in rp or "end" not in rp:
            return _abstain("manifest reporting_period invalid")
        if not isinstance(man["file_sha256"], str):
            return _abstain("manifest file_sha256 invalid")
        rc = man["row_count"]
        if isinstance(rc, bool) or not isinstance(rc, int):
            if isinstance(rc, float) and rc == int(rc):
                rc = int(rc)
            else:
                return _abstain("manifest row_count invalid")
        if not isinstance(man["columns"], list):
            return _abstain("manifest columns invalid")
        changes = man["schema_changes"]
        if changes is None:
            changes = []
        if not isinstance(changes, list):
            return _abstain("manifest schema_changes invalid")

        # 2. checksum
        if not os.path.isfile(data_path):
            return _abstain("data file missing")
        h = hashlib.sha256()
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest().lower() != man["file_sha256"].strip().lower():
            return _abstain("sha256 mismatch")

        # 3. row count
        try:
            table = pq.read_table(data_path)
        except Exception as e:
            return _abstain("cannot read parquet: %s" % e)
        if table.num_rows != rc:
            return _abstain("row count mismatch")
        df = table.to_pandas()
        if df.columns.duplicated().any():
            return _abstain("duplicate column names")

        # 4. columns / schema changes
        renames = {}
        for ch in changes:
            if not isinstance(ch, dict):
                return _abstain("invalid schema change entry")
            if ch.get("type") == "rename":
                renames[ch.get("from")] = ch.get("to")
            else:
                return _abstain("unsupported schema change: %s" % ch.get("type"))
        cols = set(df.columns)
        for old, new in renames.items():
            if old in cols and new not in cols:
                df = df.rename(columns={old: new})
                cols = set(df.columns)
            elif old in cols and new in cols:
                return _abstain("rename conflict for column %s" % old)
        for c in man["columns"]:
            if not isinstance(c, dict) or "name" not in c:
                return _abstain("manifest column entry invalid")
            name = renames.get(c["name"], c["name"])
            if name not in cols and c["name"] not in cols:
                return _abstain("manifest column %s not delivered" % c["name"])
        absent = [c for c in _REQUIRED if c not in cols]
        if absent:
            return _abstain("required columns missing: %s" % absent)

        # normalize
        for c in ["record_id", "VendorID", "PULocationID", "DOLocationID",
                  "trip_distance", "payment_type", "fare_amount", "tip_amount", "total_amount"]:
            if c in df.columns:
                try:
                    df[c] = _to_numeric(df[c])
                except Exception:
                    return _abstain("column %s not numeric" % c)
        for c in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
            if c in df.columns:
                try:
                    df[c] = _to_local_naive(df[c], _TZ)
                except Exception:
                    if c in _REQUIRED:
                        return _abstain("column %s not parseable as datetime" % c)
                    return _abstain("column %s not parseable as datetime" % c)

        # required non-null
        for c in _REQUIRED:
            if df[c].isna().any():
                return _abstain("nulls in required column %s" % c)

        # 5. reporting period
        tzname = rp.get("timezone") or _TZ
        try:
            start = pd.Timestamp(rp["start"])
            end = pd.Timestamp(rp["end"])
            if start.tzinfo is not None:
                start = start.tz_convert(tzname).tz_localize(None)
            if end.tzinfo is not None:
                end = end.tz_convert(tzname).tz_localize(None)
        except Exception:
            return _abstain("reporting_period unparseable")
        pk = df["tpep_pickup_datetime"]
        if ((pk < start) | (pk >= end)).any():
            return _abstain("pickup timestamps outside reporting period")

        # 6. contract rules
        for c in ["trip_distance", "fare_amount"]:
            if not np.isfinite(df[c].astype("float64")).all():
                return _abstain("non-finite values in %s" % c)
        for c in ["tip_amount", "total_amount"]:
            if c in df.columns:
                v = df[c].astype("float64")
                if np.isinf(v).any():
                    return _abstain("infinite values in %s" % c)
        if ((df["trip_distance"] <= 0) | (df["trip_distance"] > 100)).any():
            return _abstain("trip_distance out of range")
        if ((df["fare_amount"] < 0) | (df["fare_amount"] > 1000)).any():
            return _abstain("fare_amount out of range")
        pt = df["payment_type"].astype("float64")
        if (pt != np.floor(pt)).any() or (~pt.isin([0, 1, 2, 3, 4, 5, 6])).any():
            return _abstain("undocumented payment_type code")
        rid = df["record_id"].astype("float64")
        if (rid != np.floor(rid)).any():
            return _abstain("record_id not integer")

        # replays / conflicts
        try:
            dedup = df.drop_duplicates()
        except Exception as e:
            return _abstain("cannot compare records: %s" % e)
        if dedup["record_id"].duplicated().any():
            return _abstain("conflicting duplicate record_id values")

        n = len(dedup)
        if n == 0:
            return _abstain("no trips")
        fare = dedup["fare_amount"].astype("float64")
        dist = dedup["trip_distance"].astype("float64")
        total_fare = float(fare.sum())
        metrics = {
            "trip_count": int(n),
            "total_fare": total_fare,
            "mean_fare": float(fare.mean()),
            "mean_trip_distance": float(dist.mean()),
            "card_share": float((dedup["payment_type"] == 1).sum() / n),
        }
        return {"status": "ok", "metrics": metrics, "reason": ""}
    except Exception as e:
        return _abstain("unexpected error: %s" % e)
