"""Monthly-independent summary metrics for NYC TLC yellow taxi trip snapshots.

analyze() validates a Parquet snapshot against the data contract and returns
either status "ok" with the five metrics or status "abstain" with a reason.
It never prints, prompts, or writes files.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_NY = ZoneInfo("America/New_York")
_TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$", re.IGNORECASE)

_CONTRACT_COLUMNS = (
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
)
_REQUIRED = ("record_id", "tpep_pickup_datetime", "trip_distance",
             "payment_type", "fare_amount")
_INT_COLUMNS = ("record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type")
_FLOAT_COLUMNS = ("trip_distance", "fare_amount", "tip_amount", "total_amount")
_DATETIME_COLUMNS = ("tpep_pickup_datetime", "tpep_dropoff_datetime")
_VALID_PAYMENT_TYPES = [0, 1, 2, 3, 4, 5, 6]


class _Abstain(Exception):
    """Raised when the input violates the contract in a way that blocks a correct result."""


def _abstain(reason: str) -> dict:
    return {"status": "abstain", "metrics": None, "reason": reason}


def _strip(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        return series.map(lambda v: v.strip() if isinstance(v, str) else v)
    return series


def _to_numeric(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        raise _Abstain(f"{name}: boolean values are not allowed")
    if pd.api.types.is_numeric_dtype(series):
        return series
    raw = _strip(series)
    out = pd.to_numeric(raw, errors="coerce")
    if bool((raw.notna() & out.isna()).any()):
        raise _Abstain(f"{name}: unparseable numeric values")
    return out


def _coerce_int(series: pd.Series, name: str) -> pd.Series:
    out = _to_numeric(series, name)
    if out.dtype.kind == "f":
        vals = out.dropna().to_numpy()
        if not bool(np.all(np.mod(vals, 1) == 0)):
            raise _Abstain(f"{name}: non-integral values")
    return out.astype("Int64")


def _coerce_float(series: pd.Series, name: str) -> pd.Series:
    out = _to_numeric(series, name).astype("float64")
    vals = out.dropna().to_numpy()
    if not bool(np.all(np.isfinite(vals))):
        raise _Abstain(f"{name}: non-finite values")
    return out


def _to_naive_local(series: pd.Series, name: str) -> pd.Series:
    """Return naive New York local datetimes (datetime64[ns])."""
    if pd.api.types.is_datetime64_any_dtype(series):
        if series.dt.tz is not None:
            return series.dt.tz_convert(_NY).dt.tz_localize(None)
        return series.astype("datetime64[ns]")

    raw = _strip(series)
    notna = raw.notna()
    is_aware = raw.map(
        lambda v: (isinstance(v, str) and bool(_TZ_SUFFIX.search(v)))
        or (isinstance(v, datetime) and v.tzinfo is not None)
    ).astype(bool)

    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    naive_sel = notna & ~is_aware
    aware_sel = notna & is_aware

    if bool(naive_sel.any()):
        parsed = pd.to_datetime(raw[naive_sel], errors="coerce", format="ISO8601")
        result[naive_sel] = parsed.to_numpy()
    if bool(aware_sel.any()):
        parsed = pd.to_datetime(raw[aware_sel], errors="coerce",
                                format="ISO8601", utc=True)
        parsed = parsed.dt.tz_convert(_NY).dt.tz_localize(None)
        result[aware_sel] = parsed.to_numpy()

    if bool((notna & result.isna()).any()):
        raise _Abstain(f"{name}: unparseable timestamps")
    return result


def _parse_bound(value, label: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise _Abstain(f"manifest {label} must be an ISO-8601 string")
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise _Abstain(f"manifest {label} is not a valid timestamp") from exc
    if ts is pd.NaT:
        raise _Abstain(f"manifest {label} is not a valid timestamp")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_NY).tz_localize(None)
    return ts


def _load_manifest(path: str):
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise _Abstain("manifest is not a JSON object")

    period = manifest.get("reporting_period")
    if not isinstance(period, dict) or "start" not in period or "end" not in period:
        raise _Abstain("manifest lacks reporting_period start/end")
    start = _parse_bound(period["start"], "reporting_period.start")
    end = _parse_bound(period["end"], "reporting_period.end")
    if not start < end:
        raise _Abstain("reporting_period start must be before end")

    changes = manifest.get("schema_changes", [])
    if changes is None:
        changes = []
    if not isinstance(changes, list):
        raise _Abstain("manifest schema_changes must be a list")
    return start, end, changes


def _apply_renames(df: pd.DataFrame, changes: list) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for change in changes:
        if not isinstance(change, dict) or change.get("type") != "rename":
            raise _Abstain("unsupported or malformed schema change")
        frm, to = change.get("from"), change.get("to")
        if not isinstance(frm, str) or not isinstance(to, str):
            raise _Abstain("malformed rename in schema_changes")
        if to in df.columns:
            if frm in df.columns:
                raise _Abstain(f"ambiguous rename: both {frm!r} and {to!r} present")
            rename_map[to] = frm
    out = df.rename(columns=rename_map)
    if out.columns.duplicated().any():
        raise _Abstain("duplicate column names after applying renames")
    return out


def _analyze(data_path: str, manifest_path: str) -> dict:
    start, end, changes = _load_manifest(manifest_path)

    table = pq.read_table(data_path)
    df = table.to_pandas()
    if df.columns.duplicated().any():
        raise _Abstain("duplicate column names in data")
    df = _apply_renames(df, changes)

    missing = [c for c in _CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        raise _Abstain("missing contract columns: " + ", ".join(missing))

    norm = pd.DataFrame(index=df.index)
    for col in _INT_COLUMNS:
        norm[col] = _coerce_int(df[col], col)
    for col in _FLOAT_COLUMNS:
        norm[col] = _coerce_float(df[col], col)
    for col in _DATETIME_COLUMNS:
        norm[col] = _to_naive_local(df[col], col)

    for col in _REQUIRED:
        if bool(norm[col].isna().any()):
            raise _Abstain(f"required column {col} contains nulls")

    payment = norm["payment_type"]
    if not bool(payment.isin(_VALID_PAYMENT_TYPES).all()):
        raise _Abstain("payment_type contains codes outside 0-6")

    dist = norm["trip_distance"]
    if not bool(((dist > 0) & (dist <= 100)).all()):
        raise _Abstain("trip_distance outside valid range (0, 100]")

    fare = norm["fare_amount"]
    if not bool(((fare >= 0) & (fare <= 1000)).all()):
        raise _Abstain("fare_amount outside valid range [0, 1000]")

    # Identical replays collapse to one row; same record_id with different values is a conflict.
    unique = norm.drop_duplicates()
    if bool(unique["record_id"].duplicated().any()):
        raise _Abstain("conflicting records share a record_id")

    pickup = unique["tpep_pickup_datetime"]
    if bool(((pickup < start) | (pickup >= end)).any()):
        raise _Abstain("trips with pickup outside the reporting period")

    n = len(unique)
    if n == 0:
        raise _Abstain("no trips in data")

    fares = unique["fare_amount"].to_numpy(dtype="float64").tolist()
    dists = unique["trip_distance"].to_numpy(dtype="float64").tolist()
    total_fare = math.fsum(fares)
    mean_fare = total_fare / n
    mean_trip_distance = math.fsum(dists) / n
    card_share = int((unique["payment_type"] == 1).sum()) / n

    return {
        "trip_count": n,
        "total_fare": total_fare,
        "mean_fare": mean_fare,
        "mean_trip_distance": mean_trip_distance,
        "card_share": card_share,
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = _analyze(data_path, manifest_path)
    except _Abstain as exc:
        return _abstain(str(exc))
    except Exception as exc:  # any failure to validate or read means no trustworthy result
        return _abstain(f"could not validate input: {type(exc).__name__}")
    return {"status": "ok", "metrics": metrics, "reason": ""}
