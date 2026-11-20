"""Summarize NYC TLC yellow taxi trips with strict data-contract checks.

analyze(data_path, manifest_path) returns an "ok" result with metrics when the
inputs satisfy the contract, or an "abstain" result when they do not.
Pure function: no printing, prompting, or file writes.
"""

import datetime as _dt
import json
import math
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_NY = "America/New_York"
_REQUIRED = ("record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount")
_OPTIONAL_TS = ("tpep_dropoff_datetime",)
_OPTIONAL_NUM = ("VendorID", "PULocationID", "DOLocationID", "tip_amount", "total_amount")
_TZ_SUFFIX = re.compile(r"(?:[Zz]|[+-]\d{2}:?\d{2})$")


class _Abstain(Exception):
    """Raised internally when the data cannot yield a trustworthy result."""


def _fail(reason):
    raise _Abstain(reason)


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _naive_local(value):
    """Parse a manifest timestamp into a naive New York local Timestamp."""
    if not isinstance(value, str):
        _fail("reporting_period value is not a string")
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        _fail("reporting_period value is invalid")
    if pd.isna(ts):
        _fail("reporting_period value is invalid")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_NY).tz_localize(None)
    return ts


def _load_period(manifest):
    period = manifest.get("reporting_period")
    if not isinstance(period, dict) or "start" not in period or "end" not in period:
        _fail("manifest lacks reporting_period")
    start = _naive_local(period["start"])
    end = _naive_local(period["end"])
    if start >= end:
        _fail("reporting_period start is not before end")
    return start, end


def _rename_map(manifest, columns):
    """Map delivered column names back to canonical names declared as renames."""
    changes = manifest.get("schema_changes", [])
    if not isinstance(changes, list):
        _fail("schema_changes is malformed")
    mapping = {}
    for change in changes:
        if (not isinstance(change, dict)
                or change.get("type") != "rename"
                or not isinstance(change.get("from"), str)
                or not isinstance(change.get("to"), str)):
            _fail("unsupported schema change declared")
        original, delivered = change["from"], change["to"]
        if delivered in mapping:
            _fail("conflicting rename declarations")
        if delivered in columns:
            if original in columns:
                _fail("ambiguous renamed column")
            mapping[delivered] = original
    return mapping


def _load_frame(data_path, manifest):
    table = pq.read_table(data_path)
    columns = table.column_names
    if len(set(columns)) != len(columns):
        _fail("duplicate column names in data")
    renames = _rename_map(manifest, set(columns))
    frame = table.to_pandas().rename(columns=renames)
    missing = [c for c in _REQUIRED if c not in frame.columns]
    if missing:
        _fail(f"missing required column {missing[0]}")
    return frame


def _numeric(series, name, *, integral=False, allow_null=False):
    if not allow_null and series.isna().any():
        _fail(f"{name} has null values")
    if pd.api.types.is_bool_dtype(series):
        _fail(f"{name} has boolean type")
    if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
        series = series.map(lambda v: v.strip() if isinstance(v, str) else v)
    values = pd.to_numeric(series, errors="coerce")
    if (series.notna() & values.isna()).any():
        _fail(f"{name} has non-numeric values")
    present = values[values.notna()].astype("float64")
    if not np.isfinite(present.to_numpy()).all():
        _fail(f"{name} has non-finite values")
    if integral:
        if values.isna().any() or (present % 1 != 0).any():
            _fail(f"{name} has non-integer values")
        values = values.astype("int64")
    return values


def _timestamps(series, name, *, allow_null=False):
    if not allow_null and series.isna().any():
        _fail(f"{name} has null values")
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(_NY).dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
        series = series.map(lambda v: v.strip() if isinstance(v, str) else v)
    flags = []
    for v in series[series.notna()]:
        if isinstance(v, str):
            flags.append(bool(_TZ_SUFFIX.search(v)))
        elif isinstance(v, _dt.datetime):
            flags.append(v.tzinfo is not None)
        else:
            _fail(f"{name} has unsupported timestamp values")
    if all(flags):
        # Aware values: normalize to New York local naive time.
        parsed = pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
        parsed = parsed.dt.tz_convert(_NY).dt.tz_localize(None)
    elif not any(flags):
        # Naive values are already New York local time.
        parsed = pd.to_datetime(series, format="ISO8601", errors="coerce")
    else:
        _fail(f"{name} mixes timezone-aware and naive values")
    if (series.notna() & parsed.isna()).any():
        _fail(f"{name} has unparseable timestamps")
    return parsed


def _analyze(data_path, manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        _fail("manifest is not a JSON object")

    start, end = _load_period(manifest)
    frame = _load_frame(data_path, manifest)

    trips = pd.DataFrame(index=frame.index)
    trips["record_id"] = _numeric(frame["record_id"], "record_id", integral=True)
    trips["tpep_pickup_datetime"] = _timestamps(frame["tpep_pickup_datetime"], "tpep_pickup_datetime")
    trips["trip_distance"] = _numeric(frame["trip_distance"], "trip_distance")
    trips["payment_type"] = _numeric(frame["payment_type"], "payment_type", integral=True)
    trips["fare_amount"] = _numeric(frame["fare_amount"], "fare_amount")
    for name in _OPTIONAL_TS:
        if name in frame.columns:
            trips[name] = _timestamps(frame[name], name, allow_null=True)
    for name in _OPTIONAL_NUM:
        if name in frame.columns:
            trips[name] = _numeric(frame[name], name, allow_null=True)

    # Domain rules from the data contract.
    dist = trips["trip_distance"]
    if not ((dist > 0) & (dist <= 100)).all():
        _fail("trip_distance outside (0, 100]")
    fare = trips["fare_amount"]
    if not ((fare >= 0) & (fare <= 1000)).all():
        _fail("fare_amount outside [0, 1000]")
    if not trips["payment_type"].isin(list(range(7))).all():
        _fail("payment_type code outside 0-6")

    # Every trip must fall inside the reporting period (start inclusive, end exclusive).
    pickup = trips["tpep_pickup_datetime"]
    if not ((pickup >= start) & (pickup < end)).all():
        _fail("trip pickup outside reporting period")

    # Identical redelivered records count once; differing values for one record_id conflict.
    if trips["record_id"].duplicated().any():
        collapsed = trips.drop_duplicates()
        if collapsed["record_id"].duplicated().any():
            _fail("conflicting duplicate record_id")
        trips = collapsed

    count = len(trips)
    if count == 0:
        _fail("no trips in reporting period")

    total_fare = math.fsum(trips["fare_amount"].tolist())
    total_distance = math.fsum(trips["trip_distance"].tolist())
    card_trips = int((trips["payment_type"] == 1).sum())

    metrics = {
        "trip_count": int(count),
        "total_fare": float(total_fare),
        "mean_fare": float(total_fare / count),
        "mean_trip_distance": float(total_distance / count),
        "card_share": float(card_trips / count),
    }
    return {"status": "ok", "metrics": metrics, "reason": ""}


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except _Abstain as exc:
        return _abstain(str(exc))
    except Exception as exc:  # any unexpected failure means no trustworthy result
        return _abstain(f"could not validate inputs ({type(exc).__name__})")
