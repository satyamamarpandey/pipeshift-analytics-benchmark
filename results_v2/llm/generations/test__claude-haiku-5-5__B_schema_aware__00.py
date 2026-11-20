import json
import math
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_TZ = "America/New_York"
_CANON = [
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]
_REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
_INT_COLS = {"record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"}
_DT_COLS = {"tpep_pickup_datetime", "tpep_dropoff_datetime"}
_PAYMENT_CODES = [0, 1, 2, 3, 4, 5, 6]
_CARD_CODE = 1
_OFFSET_RE = re.compile(
    r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|[+-]\d{2}(?::?\d{2})?)$",
    re.IGNORECASE,
)


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _strip_or_none(v):
    if isinstance(v, str):
        v = v.strip()
        return v if v else None
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _parse_bound(value, label):
    if not isinstance(value, str) or not value.strip():
        raise _Abstain(f"reporting_period {label} is not a timestamp string")
    try:
        ts = pd.Timestamp(value.strip())
    except (ValueError, TypeError) as exc:
        raise _Abstain(f"reporting_period {label} is unparseable") from exc
    if pd.isna(ts):
        raise _Abstain(f"reporting_period {label} is missing")
    if ts.tz is not None:
        ts = ts.tz_convert(_TZ).tz_localize(None)
    return ts


def _load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise _Abstain("manifest is not a JSON object")
    period = manifest.get("reporting_period")
    if not isinstance(period, dict) or "start" not in period or "end" not in period:
        raise _Abstain("manifest lacks reporting_period start/end")
    start = _parse_bound(period["start"], "start")
    end = _parse_bound(period["end"], "end")
    if not start < end:
        raise _Abstain("reporting_period start is not before end")
    changes = manifest.get("schema_changes", [])
    if changes is None:
        changes = []
    if not isinstance(changes, list):
        raise _Abstain("schema_changes is malformed")
    return start, end, changes


def _apply_schema_changes(df, changes):
    mapping = {}
    for ch in changes:
        if not isinstance(ch, dict) or ch.get("type") != "rename":
            raise _Abstain("manifest declares an unsupported schema change")
        frm, to = ch.get("from"), ch.get("to")
        if not isinstance(frm, str) or not isinstance(to, str):
            raise _Abstain("malformed rename in schema_changes")
        if frm in df.columns and to in df.columns and frm != to:
            raise _Abstain(f"both '{frm}' and '{to}' present; rename is ambiguous")
        if to in df.columns and frm != to:
            mapping[to] = frm
    if mapping:
        df = df.rename(columns=mapping)
    if df.columns.duplicated().any():
        raise _Abstain("duplicate column names after applying renames")
    return df


def _numeric_column(s, name, integral):
    if pd.api.types.is_bool_dtype(s):
        raise _Abstain(f"{name} has boolean values")
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        s = s.map(_strip_or_none)
    num = pd.to_numeric(s, errors="coerce")
    if (s.notna() & num.isna()).any():
        raise _Abstain(f"{name} contains non-numeric values")
    if not (pd.api.types.is_integer_dtype(num) or pd.api.types.is_float_dtype(num)):
        raise _Abstain(f"{name} is not numeric")
    if pd.api.types.is_float_dtype(num):
        vals = num.to_numpy(dtype="float64", na_value=np.nan)
        present = vals[~np.isnan(vals)]
        if not np.all(np.isfinite(present)):
            raise _Abstain(f"{name} contains non-finite values")
        if integral and not np.all(np.mod(present, 1.0) == 0):
            raise _Abstain(f"{name} contains non-integral values")
    return num


def _to_ny_naive(series):
    if getattr(series.dt, "tz", None) is not None:
        return series.dt.tz_convert(_TZ).dt.tz_localize(None)
    return series


def _datetime_column(s, name):
    if pd.api.types.is_datetime64_any_dtype(s):
        return _to_ny_naive(s)
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        raise _Abstain(f"{name} has unsupported type {s.dtype}")
    raw = s.map(_strip_or_none)
    texts = raw.dropna()
    if texts.empty:
        return pd.to_datetime(raw, errors="coerce")
    if not texts.map(lambda v: isinstance(v, str)).all():
        parsed = pd.to_datetime(raw, errors="coerce")
        if (raw.notna() & parsed.isna()).any():
            raise _Abstain(f"{name} contains unparseable timestamps")
        if pd.api.types.is_datetime64_any_dtype(parsed):
            return _to_ny_naive(parsed)
        raise _Abstain(f"{name} contains unparseable timestamps")
    has_off = texts.map(lambda v: bool(_OFFSET_RE.search(v)))
    if has_off.all():
        parsed = pd.to_datetime(raw, errors="coerce", utc=True, format="ISO8601")
        if (raw.notna() & parsed.isna()).any():
            raise _Abstain(f"{name} contains unparseable timestamps")
        return parsed.dt.tz_convert(_TZ).dt.tz_localize(None)
    if has_off.any():
        raise _Abstain(f"{name} mixes timezone-aware and naive timestamps")
    parsed = pd.to_datetime(raw, errors="coerce", format="ISO8601")
    if (raw.notna() & parsed.isna()).any():
        raise _Abstain(f"{name} contains unparseable timestamps")
    return parsed


def _as_float(s):
    return s.to_numpy(dtype="float64", na_value=np.nan)


def _analyze(data_path, manifest_path):
    start, end, changes = _load_manifest(manifest_path)

    df = pq.read_table(data_path).to_pandas().reset_index(drop=True)
    if df.columns.duplicated().any():
        raise _Abstain("input has duplicate column names")
    df = _apply_schema_changes(df, changes)

    for col in _REQUIRED:
        if col not in df.columns:
            raise _Abstain(f"missing required column {col}")

    coerced = {}
    for col in _CANON:
        if col not in df.columns:
            continue
        s = df[col]
        if col in _DT_COLS:
            coerced[col] = _datetime_column(s, col)
        else:
            coerced[col] = _numeric_column(s, col, integral=col in _INT_COLS)
    work = pd.DataFrame(coerced)

    for col in _REQUIRED:
        if work[col].isna().any():
            raise _Abstain(f"required column {col} contains nulls")

    pay = _as_float(work["payment_type"])
    if not np.isin(pay, _PAYMENT_CODES).all():
        raise _Abstain("payment_type contains codes outside 0-6")

    dist = _as_float(work["trip_distance"])
    if not ((dist > 0) & (dist <= 100)).all():
        raise _Abstain("trip_distance outside valid range (0, 100]")

    fare = _as_float(work["fare_amount"])
    if not ((fare >= 0) & (fare <= 1000)).all():
        raise _Abstain("fare_amount outside valid range [0, 1000]")

    key_cols = [c for c in _CANON if c in work.columns]
    rows = work.drop_duplicates(subset=key_cols).reset_index(drop=True)
    if rows["record_id"].duplicated().any():
        raise _Abstain("conflicting records share the same record_id")

    pu = rows["tpep_pickup_datetime"]
    if not ((pu >= start) & (pu < end)).all():
        raise _Abstain("some trips have pickup outside the reporting_period")

    n = len(rows)
    if n == 0:
        raise _Abstain("no trips in reporting_period")

    fares = _as_float(rows["fare_amount"]).tolist()
    distances = _as_float(rows["trip_distance"]).tolist()
    total_fare = math.fsum(fares)
    mean_fare = total_fare / n
    mean_trip_distance = math.fsum(distances) / n
    card_count = int(np.sum(_as_float(rows["payment_type"]) == _CARD_CODE))
    card_share = card_count / n

    return {
        "status": "ok",
        "metrics": {
            "trip_count": int(n),
            "total_fare": float(total_fare),
            "mean_fare": float(mean_fare),
            "mean_trip_distance": float(mean_trip_distance),
            "card_share": float(card_share),
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except _Abstain as exc:
        return _abstain(str(exc))
    except Exception as exc:
        return _abstain(f"could not process inputs: {type(exc).__name__}")
