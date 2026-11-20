"""Monthly summary of NYC TLC yellow taxi trips with contract validation.

analyze(data_path, manifest_path) returns either a metrics dict with status "ok"
or an "abstain" result when the inputs cannot support a trustworthy answer.
The module performs no printing, prompting, or file writes.
"""

import json
import math
import re
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

_LOCAL_TZ = "America/New_York"

_COLUMNS = [
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
_INT_COLS = {"record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"}
_FLOAT_COLS = {"trip_distance", "fare_amount", "tip_amount", "total_amount"}
_DT_COLS = {"tpep_pickup_datetime", "tpep_dropoff_datetime"}
_REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
_PAYMENT_CODES = [0, 1, 2, 3, 4, 5, 6]
_CARD_CODE = 1


class _Abstain(Exception):
    """Raised internally when the data cannot yield a trustworthy result."""


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _parse_manifest_ts(value):
    if not isinstance(value, str):
        raise _Abstain("manifest reporting_period bound is not a string")
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise _Abstain("manifest reporting_period bound is not a valid timestamp") from exc
    if dt.tzinfo is not None:
        dt = dt.astimezone(ZoneInfo(_LOCAL_TZ)).replace(tzinfo=None)
    return pd.Timestamp(dt)


def _load_manifest(manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        raise _Abstain("manifest cannot be read as JSON") from exc
    if not isinstance(manifest, dict):
        raise _Abstain("manifest is not a JSON object")

    period = manifest.get("reporting_period")
    if not isinstance(period, dict) or "start" not in period or "end" not in period:
        raise _Abstain("manifest lacks reporting_period start and end")
    start = _parse_manifest_ts(period["start"])
    end = _parse_manifest_ts(period["end"])
    if start >= end:
        raise _Abstain("manifest reporting_period start is not before end")

    changes = manifest.get("schema_changes", [])
    if changes is None:
        changes = []
    if not isinstance(changes, list):
        raise _Abstain("manifest schema_changes is not a list")
    return start, end, changes


def _rename_plan(names, changes):
    """Map physical column names to contract names using declared renames."""
    plan = {}
    for change in changes:
        if not isinstance(change, dict) or change.get("type") != "rename":
            raise _Abstain("manifest contains an unsupported schema change")
        a, b = change.get("from"), change.get("to")
        if not isinstance(a, str) or not isinstance(b, str):
            raise _Abstain("manifest rename entry has non-string names")
        # The contract name is whichever side matches a contract column.
        if a in _COLUMNS and b not in _COLUMNS:
            canonical, source = a, b
        elif b in _COLUMNS and a not in _COLUMNS:
            canonical, source = b, a
        else:
            raise _Abstain("manifest rename cannot be matched to the data contract")
        if source in names and canonical in names:
            raise _Abstain("both original and renamed column are present")
        if source in names:
            plan[source] = canonical
    return plan


def _has_offset(value):
    if isinstance(value, datetime):
        return value.tzinfo is not None
    if isinstance(value, str):
        return re.search(r"(?:Z|[+-]\d{2}:\d{2}|[+-]\d{4})$", value) is not None
    return False


def _to_datetime(series, col):
    before = int(series.isna().sum())
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        conv = series.dt.tz_convert(_LOCAL_TZ).dt.tz_localize(None)
    elif pd.api.types.is_datetime64_any_dtype(series):
        conv = series
    else:
        cleaned = series.map(lambda v: v.strip() if isinstance(v, str) else v)
        notnull = cleaned.notna()
        flags = cleaned.map(_has_offset).astype(bool)
        n_off = int((flags & notnull).sum())
        n_naive = int((~flags & notnull).sum())
        if n_off and n_naive:
            raise _Abstain(f"mixed timezone-aware and naive timestamps in {col}")
        if n_off:
            parsed = pd.to_datetime(cleaned, format="ISO8601", utc=True, errors="coerce")
            conv = parsed.dt.tz_convert(_LOCAL_TZ).dt.tz_localize(None)
        else:
            conv = pd.to_datetime(cleaned, format="ISO8601", errors="coerce")
    if int(conv.isna().sum()) > before:
        raise _Abstain(f"unparseable timestamps in {col}")
    return conv


def _to_numeric(series, col, integer):
    if pd.api.types.is_bool_dtype(series):
        raise _Abstain(f"boolean values in numeric column {col}")
    before = int(series.isna().sum())
    if pd.api.types.is_numeric_dtype(series):
        conv = series
    else:
        cleaned = series.map(lambda v: v.strip() if isinstance(v, str) else v)
        conv = pd.to_numeric(cleaned, errors="coerce")
    if int(conv.isna().sum()) > before:
        raise _Abstain(f"unparseable numeric values in {col}")
    if integer:
        present = conv.dropna().astype("float64")
        if not ((present % 1) == 0).all():
            raise _Abstain(f"non-integer values in integer column {col}")
        return conv.astype("Int64")
    return conv.astype("float64")


def _analyze(data_path, manifest_path):
    start, end, changes = _load_manifest(manifest_path)

    try:
        table = pq.read_table(data_path)
    except Exception as exc:
        raise _Abstain("data file cannot be read as Parquet") from exc

    names = table.column_names
    if len(set(names)) != len(names):
        raise _Abstain("duplicate column names in data")

    plan = _rename_plan(names, changes)
    if plan:
        table = table.rename_columns([plan.get(n, n) for n in names])

    present = set(table.column_names)
    missing = [c for c in _COLUMNS if c not in present]
    if missing:
        raise _Abstain(f"missing contract columns: {', '.join(missing)}")

    raw = table.select(_COLUMNS).to_pandas()

    out = pd.DataFrame(index=raw.index)
    for col in _COLUMNS:
        s = raw[col]
        if col in _DT_COLS:
            out[col] = _to_datetime(s, col)
        elif col in _INT_COLS:
            out[col] = _to_numeric(s, col, integer=True)
        else:
            out[col] = _to_numeric(s, col, integer=False)

    for col in _REQUIRED:
        if out[col].isna().any():
            raise _Abstain(f"null values in required column {col}")

    dist = out["trip_distance"]
    if not ((dist > 0) & (dist <= 100)).all():
        raise _Abstain("trip_distance outside valid range 0 < x <= 100")

    fare = out["fare_amount"]
    if not ((fare >= 0) & (fare <= 1000)).all():
        raise _Abstain("fare_amount outside valid range 0 <= x <= 1000")

    if not out["payment_type"].isin(_PAYMENT_CODES).all():
        raise _Abstain("payment_type contains codes outside 0..6")

    pickup = out["tpep_pickup_datetime"]
    if not ((pickup >= start) & (pickup < end)).all():
        raise _Abstain("trips with pickup outside the manifest reporting_period")

    # Identical replays count once; same record_id with different values is a conflict.
    dedup = out.drop_duplicates(keep="first")
    if dedup["record_id"].duplicated().any():
        raise _Abstain("conflicting records share a record_id")

    n = len(dedup)
    if n == 0:
        raise _Abstain("no trips in data; metrics undefined")

    total_fare = math.fsum(dedup["fare_amount"].tolist())
    mean_fare = total_fare / n
    mean_trip_distance = math.fsum(dedup["trip_distance"].tolist()) / n
    card_trips = int((dedup["payment_type"] == _CARD_CODE).sum())
    card_share = card_trips / n

    return {
        "status": "ok",
        "metrics": {
            "trip_count": int(n),
            "total_fare": total_fare,
            "mean_fare": mean_fare,
            "mean_trip_distance": mean_trip_distance,
            "card_share": card_share,
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    """Compute the monthly yellow taxi summary, or abstain if the contract is violated."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _analyze(data_path, manifest_path)
    except _Abstain as exc:
        return _abstain(str(exc))
    except Exception:
        return _abstain("unexpected error while validating or computing from inputs")
