import hashlib
import json
import math
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_REQUIRED_MANIFEST = ["dataset", "snapshot_id", "snapshot_version", "reporting_period",
                      "file", "file_sha256", "row_count", "columns", "units",
                      "schema_changes", "description"]
_CONTRACT_COLS = ["record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
                  "PULocationID", "DOLocationID", "trip_distance", "payment_type",
                  "fare_amount", "tip_amount", "total_amount"]
_NOT_NULL = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
_TZ = "America/New_York"


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": str(reason)}


def _to_local_naive(series, tz):
    s = series
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(tz).dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        raise _Abstain("numeric timestamps are ambiguous")
    try:
        r = pd.to_datetime(s, format="ISO8601")
    except Exception:
        try:
            r = pd.to_datetime(s, format="ISO8601", utc=True)
        except Exception:
            raise _Abstain("unparseable pickup timestamps")
        # mixed offsets/naive: naive values would be wrongly treated as UTC
        txt = s.dropna().astype(str)
        if not txt.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all():
            raise _Abstain("mixed naive and aware timestamps")
    if isinstance(r.dtype, pd.DatetimeTZDtype):
        return r.dt.tz_convert(tz).dt.tz_localize(None)
    return r


def _to_num(s, name):
    try:
        r = pd.to_numeric(s, errors="raise")
    except Exception:
        raise _Abstain(f"non-numeric values in {name}")
    r = r.astype("float64")
    if not np.isfinite(r.dropna().to_numpy()).all():
        raise _Abstain(f"non-finite values in {name}")
    return r


def _analyze(data_path, manifest_path):
    # 1. manifest
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)
    except Exception as e:
        raise _Abstain(f"manifest missing or invalid JSON: {e}")
    if not isinstance(man, dict):
        raise _Abstain("manifest is not an object")
    for k in _REQUIRED_MANIFEST:
        if k not in man:
            raise _Abstain(f"manifest missing field {k}")
    rp = man["reporting_period"]
    if not isinstance(rp, dict) or "start" not in rp or "end" not in rp:
        raise _Abstain("invalid reporting_period")
    if not isinstance(man["columns"], list) or not isinstance(man["schema_changes"], list):
        raise _Abstain("invalid columns/schema_changes")
    tz = rp.get("timezone") or _TZ
    if tz != _TZ:
        raise _Abstain("unexpected reporting timezone")

    # units
    units = json.dumps(man["units"]).lower()
    if re.search(r"kilomet|\bkm\b|\bcents?\b|\bmeters?\b|\bmetres?\b|\bkph\b", units):
        raise _Abstain("units differ from contract")

    # 2. hash
    try:
        h = hashlib.sha256()
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except Exception as e:
        raise _Abstain(f"cannot read data file: {e}")
    if h.hexdigest().lower() != str(man["file_sha256"]).strip().lower():
        raise _Abstain("SHA-256 mismatch")

    # 3. row count
    try:
        table = pq.read_table(data_path)
    except Exception as e:
        raise _Abstain(f"cannot read parquet: {e}")
    try:
        expected_rows = int(man["row_count"])
    except Exception:
        raise _Abstain("invalid row_count")
    if table.num_rows != expected_rows:
        raise _Abstain("row count mismatch")
    df = table.to_pandas()

    # 4. columns
    renames = {}
    for ch in man["schema_changes"]:
        if not isinstance(ch, dict):
            raise _Abstain("invalid schema_changes entry")
        if ch.get("type") == "rename":
            renames[ch["from"]] = ch["to"]
        else:
            raise _Abstain("unsupported schema change")
    if len(set(df.columns)) != len(df.columns):
        raise _Abstain("duplicate column names")
    df = df.rename(columns=renames)
    if len(set(df.columns)) != len(df.columns):
        raise _Abstain("rename produces duplicate columns")
    try:
        man_cols = [renames.get(c["name"], c["name"]) for c in man["columns"]]
    except Exception:
        raise _Abstain("invalid manifest columns")
    missing = [c for c in man_cols if c not in df.columns]
    if missing:
        raise _Abstain(f"delivered columns disagree with manifest: {missing}")
    missing = [c for c in _NOT_NULL if c not in df.columns]
    if missing:
        raise _Abstain(f"required columns missing: {missing}")

    # normalize
    work = pd.DataFrame(index=df.index)
    work["tpep_pickup_datetime"] = _to_local_naive(df["tpep_pickup_datetime"], tz)
    if "tpep_dropoff_datetime" in df.columns:
        try:
            work["tpep_dropoff_datetime"] = _to_local_naive(df["tpep_dropoff_datetime"], tz)
        except _Abstain:
            raise
    for c in ["record_id", "payment_type", "trip_distance", "fare_amount"]:
        work[c] = _to_num(df[c], c)
    for c in ["VendorID", "PULocationID", "DOLocationID", "tip_amount", "total_amount"]:
        if c in df.columns:
            work[c] = _to_num(df[c], c)

    # required non-null
    for c in _NOT_NULL:
        if work[c].isna().any():
            raise _Abstain(f"null values in required column {c}")

    # 5. period
    try:
        start = pd.Timestamp(rp["start"])
        end = pd.Timestamp(rp["end"])
        if start.tzinfo is not None:
            start = start.tz_convert(tz).tz_localize(None)
        if end.tzinfo is not None:
            end = end.tz_convert(tz).tz_localize(None)
    except Exception:
        raise _Abstain("invalid reporting_period bounds")
    p = work["tpep_pickup_datetime"]
    if ((p < start) | (p >= end)).any():
        raise _Abstain("pickup timestamp outside reporting period")

    # 6. contract rules
    rid = work["record_id"]
    if (rid != np.floor(rid)).any():
        raise _Abstain("non-integer record_id")
    pt = work["payment_type"]
    if (pt != np.floor(pt)).any() or (~pt.isin([0, 1, 2, 3, 4, 5, 6])).any():
        raise _Abstain("undocumented payment_type code")
    td = work["trip_distance"]
    if ((td <= 0) | (td > 100)).any():
        raise _Abstain("trip_distance out of range")
    fa = work["fare_amount"]
    if ((fa < 0) | (fa > 1000)).any():
        raise _Abstain("fare_amount out of range")

    # replay / conflict
    work = work.drop_duplicates()
    if work["record_id"].duplicated().any():
        raise _Abstain("conflicting duplicate record_id")
    if len(work) == 0:
        raise _Abstain("no trips")

    n = int(len(work))
    total = float(work["fare_amount"].sum())
    metrics = {
        "trip_count": n,
        "total_fare": total,
        "mean_fare": float(work["fare_amount"].mean()),
        "mean_trip_distance": float(work["trip_distance"].mean()),
        "card_share": float((work["payment_type"] == 1).sum() / n),
    }
    for v in metrics.values():
        if isinstance(v, float) and math.isnan(v):
            raise _Abstain("non-computable metric")
    return {"status": "ok", "metrics": metrics, "reason": ""}


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except _Abstain as e:
        return _abstain(e)
    except Exception as e:
        return _abstain(f"unexpected error: {e}")
