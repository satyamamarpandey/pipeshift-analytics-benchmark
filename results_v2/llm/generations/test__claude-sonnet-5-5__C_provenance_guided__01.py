import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NY = "America/New_York"
REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
MANIFEST_FIELDS = ["dataset", "snapshot_id", "snapshot_version", "reporting_period", "file",
                   "file_sha256", "row_count", "columns", "units", "schema_changes", "description"]


class Abstain(Exception):
    pass


def _abs(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _to_local_naive(series, name):
    """Convert a datetime-like series to naive New York local time."""
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        if getattr(series.dt, "tz", None) is not None:
            return series.dt.tz_convert(NY).dt.tz_localize(None)
        return series
    if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise Abstain(f"{name} has unsupported numeric type")
    try:
        s = pd.to_datetime(series, format="ISO8601", errors="raise")
    except Exception:
        try:
            s = pd.to_datetime(series, format="ISO8601", errors="raise", utc=True)
        except Exception:
            raise Abstain(f"{name} cannot be parsed as datetime")
    if s.dtype == object:
        try:
            s = pd.to_datetime(series, format="ISO8601", errors="raise", utc=True)
        except Exception:
            raise Abstain(f"{name} cannot be parsed as datetime")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(NY).dt.tz_localize(None)
    return s


def _to_numeric(series, name):
    if pd.api.types.is_bool_dtype(series.dtype):
        raise Abstain(f"{name} has boolean type")
    try:
        s = pd.to_numeric(series, errors="raise")
    except Exception:
        raise Abstain(f"{name} not numeric")
    s = s.astype("float64") if not pd.api.types.is_integer_dtype(s.dtype) else s
    if not np.all(np.isfinite(s.to_numpy(dtype="float64"))):
        raise Abstain(f"{name} has non-finite values")
    return s


def _parse_ts(v, name):
    try:
        t = pd.Timestamp(v)
    except Exception:
        raise Abstain(f"invalid reporting_period {name}")
    if pd.isna(t):
        raise Abstain(f"invalid reporting_period {name}")
    if t.tzinfo is not None:
        t = t.tz_convert(NY).tz_localize(None)
    return t


def _run(data_path, manifest_path):
    # 1. manifest
    if not os.path.isfile(manifest_path):
        raise Abstain("manifest missing")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)
    except Exception:
        raise Abstain("manifest is not valid JSON")
    if not isinstance(man, dict):
        raise Abstain("manifest is not an object")
    missing = [k for k in MANIFEST_FIELDS if k not in man]
    if missing:
        raise Abstain("manifest missing fields: " + ",".join(missing))
    rp = man["reporting_period"]
    if not isinstance(rp, dict) or not all(k in rp for k in ("start", "end", "timezone")):
        raise Abstain("manifest reporting_period invalid")
    if rp["timezone"] != NY:
        raise Abstain("unsupported reporting_period timezone")
    if not isinstance(man["columns"], list) or not isinstance(man["schema_changes"], list):
        raise Abstain("manifest columns/schema_changes invalid")
    rc = man["row_count"]
    if isinstance(rc, bool) or not isinstance(rc, int):
        raise Abstain("manifest row_count invalid")
    units_txt = json.dumps(man["units"]).lower()
    if "kilomet" in units_txt or "km" in units_txt or "cent" in units_txt or "meter" in units_txt.replace("kilometer", ""):
        raise Abstain("units differ from contract")

    start = _parse_ts(rp["start"], "start")
    end = _parse_ts(rp["end"], "end")
    if not start < end:
        raise Abstain("reporting_period empty")

    # 2. checksum
    if not os.path.isfile(data_path):
        raise Abstain("data file missing")
    h = hashlib.sha256()
    with open(data_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if not isinstance(man["file_sha256"], str) or h.hexdigest().lower() != man["file_sha256"].strip().lower():
        raise Abstain("sha256 mismatch")

    # 3. row count
    try:
        table = pq.read_table(data_path)
    except Exception:
        raise Abstain("data file unreadable")
    if table.num_rows != rc:
        raise Abstain("row_count mismatch")
    try:
        df = table.to_pandas()
    except Exception:
        raise Abstain("data cannot be converted")
    if len(set(df.columns)) != len(df.columns):
        raise Abstain("duplicate column names")

    # 4. schema
    renames = {}
    for ch in man["schema_changes"]:
        if not isinstance(ch, dict) or ch.get("type") != "rename" or "from" not in ch or "to" not in ch:
            raise Abstain("unsupported schema change")
        renames[ch["from"]] = ch["to"]
    cols = list(df.columns)
    newcols = []
    for c in cols:
        if c in renames:
            if renames[c] in cols and renames[c] != c:
                raise Abstain("rename target already present")
            newcols.append(renames[c])
        else:
            newcols.append(c)
    if len(set(newcols)) != len(newcols):
        raise Abstain("duplicate columns after rename")
    df.columns = newcols
    have = set(newcols)
    for c in man["columns"]:
        if not isinstance(c, dict) or "name" not in c:
            raise Abstain("manifest columns invalid")
        n = c["name"]
        if n not in have and renames.get(n, n) not in have:
            raise Abstain(f"column {n} missing from data")
    for c in REQUIRED:
        if c not in have:
            raise Abstain(f"required column {c} missing")

    # nulls
    for c in REQUIRED:
        if df[c].isna().any():
            raise Abstain(f"null in required column {c}")
    if len(df) == 0:
        raise Abstain("no rows")

    # normalize
    df["tpep_pickup_datetime"] = _to_local_naive(df["tpep_pickup_datetime"], "tpep_pickup_datetime")
    if "tpep_dropoff_datetime" in have and not df["tpep_dropoff_datetime"].isna().all():
        try:
            df["tpep_dropoff_datetime"] = _to_local_naive(df["tpep_dropoff_datetime"], "tpep_dropoff_datetime")
        except Abstain:
            raise
    for c in ("record_id", "payment_type"):
        s = _to_numeric(df[c], c)
        if not np.all(np.mod(s.to_numpy(dtype="float64"), 1) == 0):
            raise Abstain(f"{c} not integer-valued")
        if c == "record_id":
            if not pd.api.types.is_integer_dtype(s.dtype):
                s = s.astype("int64")
        else:
            s = s.astype("int64")
        df[c] = s
    for c in ("trip_distance", "fare_amount"):
        df[c] = _to_numeric(df[c], c).astype("float64")
    for c in ("tip_amount", "total_amount"):
        if c in have:
            try:
                df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
            except Exception:
                raise Abstain(f"{c} not numeric")

    # 5. period
    p = df["tpep_pickup_datetime"]
    if ((p < start) | (p >= end)).any():
        raise Abstain("pickup outside reporting_period")

    # 6. contract rules
    if not df["payment_type"].between(0, 6).all():
        raise Abstain("undocumented payment_type code")
    d = df["trip_distance"]
    if not ((d > 0) & (d <= 100)).all():
        raise Abstain("trip_distance out of range")
    fa = df["fare_amount"]
    if not ((fa >= 0) & (fa <= 1000)).all():
        raise Abstain("fare_amount out of range")

    try:
        df = df.drop_duplicates()
    except Exception:
        raise Abstain("cannot compare duplicate records")
    if df["record_id"].duplicated().any():
        raise Abstain("conflicting duplicate record_id")

    n = len(df)
    if n == 0:
        raise Abstain("no trips")
    total_fare = math.fsum(df["fare_amount"].tolist())
    total_dist = math.fsum(df["trip_distance"].tolist())
    card = int((df["payment_type"] == 1).sum())
    return {
        "trip_count": int(n),
        "total_fare": float(total_fare),
        "mean_fare": float(total_fare / n),
        "mean_trip_distance": float(total_dist / n),
        "card_share": float(card / n),
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        metrics = _run(data_path, manifest_path)
    except Abstain as e:
        return _abs(str(e))
    except Exception as e:
        return _abs("unexpected error: " + type(e).__name__)
    return {"status": "ok", "metrics": metrics, "reason": ""}
