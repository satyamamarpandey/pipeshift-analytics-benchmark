import hashlib
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_REQUIRED_MANIFEST = ["dataset", "snapshot_id", "snapshot_version", "reporting_period",
                      "file", "file_sha256", "row_count", "columns", "units",
                      "schema_changes", "description"]
_NUMERIC = ["record_id", "VendorID", "PULocationID", "DOLocationID", "trip_distance",
            "payment_type", "fare_amount", "tip_amount", "total_amount"]
_DATETIME = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
_REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
_TZ = "America/New_York"


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _to_local_naive(series, tz=_TZ):
    s = series
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(tz).dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        raise _Abstain("numeric timestamps are ambiguous")
    try:
        r = pd.to_datetime(s, errors="raise")
        if isinstance(r.dtype, pd.DatetimeTZDtype):
            return r.dt.tz_convert(tz).dt.tz_localize(None)
        return r
    except Exception:
        pass
    try:
        r = pd.to_datetime(s, errors="raise", utc=True)
        return r.dt.tz_convert(tz).dt.tz_localize(None)
    except Exception:
        raise _Abstain("unparseable datetime values")


def _to_numeric(series, name):
    try:
        r = pd.to_numeric(series, errors="raise")
    except Exception:
        raise _Abstain("non-numeric values in " + name)
    if r.dtype == object or not pd.api.types.is_numeric_dtype(r):
        raise _Abstain("non-numeric values in " + name)
    if pd.api.types.is_bool_dtype(r):
        raise _Abstain("boolean values in " + name)
    return r.astype("float64") if not pd.api.types.is_integer_dtype(r) else r


def _unit_factors(units):
    """Return (distance_factor_to_miles, money_factor_to_usd)."""
    dist, money = 1.0, 1.0
    if units in (None, "", {}, []):
        return dist, money
    items = []
    if isinstance(units, dict):
        items = list(units.items())
    elif isinstance(units, str):
        items = [("*", units)]
    else:
        raise _Abstain("unrecognized units declaration")
    for k, v in items:
        key = str(k).lower()
        val = str(v).lower().strip()
        if "distance" in key or key == "*" and any(t in val for t in ("mile", "km", "kilomet", "meter")):
            if "kilomet" in val or val in ("km", "kms"):
                dist = 0.621371192237334
            elif "mile" in val or val in ("mi", "miles"):
                pass
            elif "meter" in val or val in ("m",):
                dist = 1.0 / 1609.344
            elif "distance" in key:
                raise _Abstain("unrecognized distance unit: " + val)
        if any(t in key for t in ("fare", "amount", "money", "currency", "tip", "total")) or key == "*":
            if "cent" in val:
                money = 0.01
            elif "usd" in val or "dollar" in val or "$" in val:
                pass
            elif key != "*":
                raise _Abstain("unrecognized money unit: " + val)
    return dist, money


def _analyze(data_path, manifest_path):
    # 1. manifest
    try:
        with open(manifest_path, "rb") as f:
            manifest = json.loads(f.read().decode("utf-8"))
    except Exception:
        raise _Abstain("manifest missing or invalid JSON")
    if not isinstance(manifest, dict):
        raise _Abstain("manifest is not a JSON object")
    missing = [k for k in _REQUIRED_MANIFEST if k not in manifest]
    if missing:
        raise _Abstain("manifest missing fields: " + ",".join(missing))
    rp = manifest["reporting_period"]
    if not isinstance(rp, dict) or "start" not in rp or "end" not in rp:
        raise _Abstain("invalid reporting_period")

    # 2. checksum
    try:
        h = hashlib.sha256()
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except Exception:
        raise _Abstain("data file unreadable")
    if h.hexdigest().lower() != str(manifest["file_sha256"]).strip().lower():
        raise _Abstain("SHA-256 mismatch")

    # 3. row count
    try:
        pf = pq.ParquetFile(data_path)
        nrows = pf.metadata.num_rows
        df = pf.read().to_pandas()
    except Exception:
        raise _Abstain("data file is not a readable Parquet file")
    try:
        expected_rows = int(manifest["row_count"])
    except Exception:
        raise _Abstain("invalid row_count in manifest")
    if nrows != expected_rows or len(df) != expected_rows:
        raise _Abstain("row count mismatch")

    # 4. columns / schema changes
    renames = {}
    changes = manifest["schema_changes"]
    if changes is None:
        changes = []
    if not isinstance(changes, list):
        raise _Abstain("invalid schema_changes")
    for c in changes:
        if not isinstance(c, dict):
            raise _Abstain("invalid schema_changes entry")
        if c.get("type") == "rename":
            if "from" not in c or "to" not in c:
                raise _Abstain("invalid rename entry")
            renames[c["from"]] = c["to"]
        else:
            raise _Abstain("unsupported schema change: " + str(c.get("type")))
    if len(set(df.columns)) != len(df.columns):
        raise _Abstain("duplicate column names")
    # apply renames (chains not expected)
    new_cols = [renames.get(c, c) for c in df.columns]
    if len(set(new_cols)) != len(new_cols):
        raise _Abstain("rename produces duplicate columns")
    df.columns = new_cols
    mcols = manifest["columns"]
    if not isinstance(mcols, list):
        raise _Abstain("invalid manifest columns")
    mnames = []
    for c in mcols:
        if not isinstance(c, dict) or "name" not in c:
            raise _Abstain("invalid manifest column entry")
        mnames.append(renames.get(c["name"], c["name"]))
    absent = [n for n in mnames if n not in df.columns]
    if absent:
        raise _Abstain("delivered columns disagree with manifest: " + ",".join(map(str, absent)))
    absent = [n for n in _REQUIRED if n not in df.columns]
    if absent:
        raise _Abstain("required columns missing: " + ",".join(absent))

    # normalize types
    for c in _NUMERIC:
        if c in df.columns:
            df[c] = _to_numeric(df[c], c)
    for c in _DATETIME:
        if c in df.columns:
            df[c] = _to_local_naive(df[c])

    # required non-null
    for c in _REQUIRED:
        if df[c].isna().any():
            raise _Abstain("null values in required column " + c)

    # units
    dist_f, money_f = _unit_factors(manifest["units"])
    if dist_f != 1.0:
        df["trip_distance"] = df["trip_distance"].astype("float64") * dist_f
    if money_f != 1.0:
        for c in ("fare_amount", "tip_amount", "total_amount"):
            if c in df.columns:
                df[c] = df[c].astype("float64") * money_f

    # 5. reporting period
    try:
        tz = rp.get("timezone") or _TZ
        start = pd.Timestamp(rp["start"])
        end = pd.Timestamp(rp["end"])
        if start.tzinfo is not None:
            start = start.tz_convert(tz).tz_localize(None)
        if end.tzinfo is not None:
            end = end.tz_convert(tz).tz_localize(None)
    except Exception:
        raise _Abstain("unparseable reporting_period")
    pk = df["tpep_pickup_datetime"]
    if ((pk < start) | (pk >= end)).any():
        raise _Abstain("pickup timestamps outside reporting period")

    # 6. contract rules
    for c in ("trip_distance", "fare_amount"):
        if not np.isfinite(df[c].astype("float64")).all():
            raise _Abstain("non-finite values in " + c)
    rid = df["record_id"].astype("float64")
    if not np.isfinite(rid).all() or (rid != np.floor(rid)).any():
        raise _Abstain("invalid record_id values")
    pt = df["payment_type"].astype("float64")
    if not np.isfinite(pt).all() or (pt != np.floor(pt)).any() or (~pt.isin([0, 1, 2, 3, 4, 5, 6])).any():
        raise _Abstain("undocumented payment_type codes")
    td = df["trip_distance"].astype("float64")
    if ((td <= 0) | (td > 100)).any():
        raise _Abstain("trip_distance out of valid range")
    fa = df["fare_amount"].astype("float64")
    if ((fa < 0) | (fa > 1000)).any():
        raise _Abstain("fare_amount out of valid range")

    # replay / conflict handling
    contract_cols = [c for c in _NUMERIC + _DATETIME if c in df.columns]
    sub = df[contract_cols]
    dedup = sub.drop_duplicates()
    if dedup["record_id"].duplicated().any():
        raise _Abstain("conflicting duplicate record_id values")
    n = len(dedup)
    if n == 0:
        raise _Abstain("no trips")

    fares = dedup["fare_amount"].astype("float64")
    dists = dedup["trip_distance"].astype("float64")
    total = float(fares.sum())
    metrics = {
        "trip_count": int(n),
        "total_fare": total,
        "mean_fare": float(total / n),
        "mean_trip_distance": float(dists.mean()),
        "card_share": float((dedup["payment_type"] == 1).sum() / n),
    }
    return {"status": "ok", "metrics": metrics, "reason": ""}


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except _Abstain as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain("unexpected error: " + type(e).__name__)
