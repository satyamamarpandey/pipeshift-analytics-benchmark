"""Monthly summary of NYC yellow taxi trips with contract validation."""

from __future__ import annotations

import datetime as _dt
import json
import math
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_TZ = "America/New_York"

_CONTRACT_COLUMNS = [
    "record_id",
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "total_amount",
]
_REQUIRED = [
    "record_id",
    "tpep_pickup_datetime",
    "trip_distance",
    "payment_type",
    "fare_amount",
]
_INT_COLS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"]
_FLOAT_COLS = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
_TS_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
_VALID_PAYMENT = {0, 1, 2, 3, 4, 5, 6}
_BENIGN_CHANGES = {"add", "add_column", "added", "reorder", "reorder_columns", "type_change",
                   "retype", "cast", "timezone", "tz_aware"}


class _Abstain(Exception):
    pass


def _abstain(reason: str) -> dict:
    return {"status": "abstain", "metrics": None, "reason": reason}


# ----------------------------------------------------------------------------- time helpers
def _epoch_to_local(value: float) -> pd.Timestamp:
    av = abs(value)
    if av >= 1e17:
        unit = "ns"
    elif av >= 1e14:
        unit = "us"
    elif av >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    ts = pd.Timestamp(value, unit=unit, tz="UTC")
    return ts.tz_convert(_TZ).tz_localize(None)


def _scalar_to_local(v):
    if v is None:
        return pd.NaT
    if isinstance(v, float) and math.isnan(v):
        return pd.NaT
    if v is pd.NaT:
        return pd.NaT
    if isinstance(v, (bool, np.bool_)):
        raise ValueError("boolean timestamp")
    if isinstance(v, (int, float, np.integer, np.floating)):
        return _epoch_to_local(float(v))
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"nan", "nat", "none", "null"}:
            return pd.NaT
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", s):
            return _epoch_to_local(float(s))
        ts = pd.Timestamp(s)
    elif isinstance(v, (_dt.datetime, _dt.date, np.datetime64, pd.Timestamp)):
        ts = pd.Timestamp(v)
    else:
        raise ValueError(f"unsupported timestamp value type {type(v)!r}")
    if ts is pd.NaT or pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_TZ).tz_localize(None)
    return ts


def _to_local_naive(series: pd.Series, name: str) -> pd.Series:
    s = series
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(_TZ).dt.tz_localize(None).astype("datetime64[ns]")
    if pd.api.types.is_datetime64_dtype(s.dtype):
        return s.astype("datetime64[ns]")
    if pd.api.types.is_bool_dtype(s.dtype):
        raise _Abstain(f"{name} has boolean type")
    if pd.api.types.is_numeric_dtype(s.dtype):
        out = [pd.NaT if pd.isna(x) else _epoch_to_local(float(x)) for x in s.to_numpy()]
        return pd.Series(pd.DatetimeIndex(out), index=s.index).astype("datetime64[ns]")
    cache: dict = {}
    out = []
    for v in s.to_numpy(dtype=object):
        try:
            key = (type(v), v)
            hash(key)
        except TypeError:
            key = None
        if key is not None and key in cache:
            out.append(cache[key])
            continue
        try:
            r = _scalar_to_local(v)
        except Exception:
            raise _Abstain(f"unparseable value in {name}")
        if key is not None:
            cache[key] = r
        out.append(r)
    try:
        res = pd.Series(pd.DatetimeIndex(out), index=s.index).astype("datetime64[ns]")
    except Exception:
        raise _Abstain(f"unparseable values in {name}")
    return res


# -------------------------------------------------------------------------- numeric helpers
def _to_numeric(series: pd.Series, name: str) -> pd.Series:
    s = series
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    if pd.api.types.is_bool_dtype(s.dtype):
        raise _Abstain(f"{name} has boolean type")
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        raise _Abstain(f"{name} has datetime type")
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.astype("float64")
    obj = s.astype(object)
    orig_null = obj.map(
        lambda x: x is None
        or (isinstance(x, float) and math.isnan(x))
        or (isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null"})
    )
    cleaned = obj.map(lambda x: x.strip() if isinstance(x, str) else x)
    if cleaned.map(lambda x: isinstance(x, (bool, np.bool_))).any():
        raise _Abstain(f"{name} contains boolean values")
    num = pd.to_numeric(cleaned.where(~orig_null, np.nan), errors="coerce")
    if (num.isna() & ~orig_null).any():
        raise _Abstain(f"unparseable numeric value in {name}")
    return num.astype("float64")


def _check_integral(s: pd.Series, name: str) -> None:
    vals = s.dropna().to_numpy()
    if len(vals) and (not np.all(np.isfinite(vals)) or np.any(vals != np.round(vals))):
        raise _Abstain(f"{name} has non-integer values")


# ------------------------------------------------------------------------- manifest helpers
def _parse_bound(v) -> pd.Timestamp:
    ts = _scalar_to_local(v)
    if ts is pd.NaT or pd.isna(ts):
        raise _Abstain("invalid reporting_period bound")
    return ts


def _reporting_period(manifest: dict):
    rp = manifest.get("reporting_period")
    if rp is None:
        raise _Abstain("manifest missing reporting_period")
    if isinstance(rp, dict):
        start = end = None
        for k in ("start", "start_date", "from", "begin", "start_inclusive"):
            if k in rp:
                start = rp[k]
                break
        for k in ("end", "end_date", "to", "stop", "end_exclusive"):
            if k in rp:
                end = rp[k]
                break
        if start is None or end is None:
            raise _Abstain("reporting_period lacks start/end")
        s, e = _parse_bound(start), _parse_bound(end)
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        s, e = _parse_bound(rp[0]), _parse_bound(rp[1])
    elif isinstance(rp, str):
        m = re.fullmatch(r"\s*(\d{4})-(\d{1,2})\s*", rp)
        if not m:
            raise _Abstain("unrecognized reporting_period format")
        y, mo = int(m.group(1)), int(m.group(2))
        s = pd.Timestamp(year=y, month=mo, day=1)
        e = s + pd.DateOffset(months=1)
    else:
        raise _Abstain("unrecognized reporting_period format")
    if not s < e:
        raise _Abstain("reporting_period start not before end")
    return s, e


def _apply_schema_changes(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    changes = manifest.get("schema_changes") or []
    if not isinstance(changes, list):
        raise _Abstain("schema_changes is not a list")
    for ch in changes:
        if not isinstance(ch, dict):
            raise _Abstain("malformed schema_changes entry")
        typ = str(ch.get("type", "")).strip().lower()
        if typ == "rename":
            src, dst = ch.get("from"), ch.get("to")
            if not isinstance(src, str) or not isinstance(dst, str):
                raise _Abstain("malformed rename entry")
            if dst in df.columns:
                if src in df.columns and src != dst:
                    raise _Abstain(f"both {src} and renamed {dst} present")
                df = df.rename(columns={dst: src})
            # if dst absent but src present, data already uses the contract name
        elif typ in _BENIGN_CHANGES:
            continue
        else:
            raise _Abstain(f"unsupported schema change type: {typ or '<missing>'}")
    return df


# ----------------------------------------------------------------------------------- main
def _run(data_path: str, manifest_path: str) -> dict:
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception:
        raise _Abstain("cannot read manifest")
    if not isinstance(manifest, dict):
        raise _Abstain("manifest is not an object")

    start, end = _reporting_period(manifest)

    try:
        table = pq.read_table(data_path)
    except Exception:
        raise _Abstain("cannot read parquet data")
    names = list(table.column_names)
    if len(set(names)) != len(names):
        raise _Abstain("duplicate column names in data")
    try:
        df = table.to_pandas()
    except Exception:
        raise _Abstain("cannot convert data to dataframe")

    raw_rows = len(df)
    df = _apply_schema_changes(df, manifest)

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise _Abstain(f"missing required columns: {', '.join(missing)}")

    present = [c for c in _CONTRACT_COLUMNS if c in df.columns]
    norm = pd.DataFrame(index=df.index)
    for c in present:
        if c in _TS_COLS:
            norm[c] = _to_local_naive(df[c], c)
        else:
            norm[c] = _to_numeric(df[c], c)
            if c in _INT_COLS:
                _check_integral(norm[c], c)

    for c in _REQUIRED:
        if norm[c].isna().any():
            raise _Abstain(f"null values in required column {c}")

    # Replays vs conflicts
    norm = norm.drop_duplicates(subset=present, keep="first")
    if norm["record_id"].duplicated().any():
        raise _Abstain("conflicting records share a record_id")

    dedup_rows = len(norm)
    for k in ("row_count", "record_count", "num_rows", "expected_row_count", "n_rows"):
        if k in manifest:
            v = manifest[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if int(v) != raw_rows and int(v) != dedup_rows:
                    raise _Abstain("row count does not match manifest")
            break

    if dedup_rows == 0:
        raise _Abstain("no trips in data")

    dist = norm["trip_distance"]
    if not np.all(np.isfinite(dist)) or ((dist <= 0) | (dist > 100)).any():
        raise _Abstain("trip_distance outside valid range")
    fare = norm["fare_amount"]
    if not np.all(np.isfinite(fare)) or ((fare < 0) | (fare > 1000)).any():
        raise _Abstain("fare_amount outside valid range")
    pt = norm["payment_type"]
    if not pt.isin(list(_VALID_PAYMENT)).all():
        raise _Abstain("invalid payment_type code")
    pu = norm["tpep_pickup_datetime"]
    if ((pu < start) | (pu >= end)).any():
        raise _Abstain("pickup outside reporting_period")

    n = int(dedup_rows)
    fare_vals = fare.to_numpy(dtype="float64")
    total_fare = float(math.fsum(fare_vals))
    metrics = {
        "trip_count": n,
        "total_fare": total_fare,
        "mean_fare": total_fare / n,
        "mean_trip_distance": float(math.fsum(dist.to_numpy(dtype="float64"))) / n,
        "card_share": float(int((pt == 1).sum())) / n,
    }
    return {"status": "ok", "metrics": metrics, "reason": ""}


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _run(data_path, manifest_path)
    except _Abstain as exc:
        return _abstain(str(exc))
    except Exception as exc:  # any unexpected failure means the result is not trustworthy
        return _abstain(f"unexpected error: {type(exc).__name__}")
