"""NYC TLC yellow taxi monthly summary with strict contract verification.

analyze(data_path, manifest_path) verifies the manifest and the Parquet data file
against the published data contract. Metrics are computed only if every check
passes; otherwise the function abstains with a reason. The module reads only the
two given paths and never prints, prompts, or writes files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import warnings

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_TZ = "America/New_York"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_INT_RE = re.compile(r"[+-]?\d+")
_INT_DEC_RE = re.compile(r"[+-]?\d+\.0+")

# Canonical contract column -> kind ("int", "double", "ts")
_CONTRACT = {
    "record_id": "int",
    "VendorID": "int",
    "tpep_pickup_datetime": "ts",
    "tpep_dropoff_datetime": "ts",
    "PULocationID": "int",
    "DOLocationID": "int",
    "trip_distance": "double",
    "payment_type": "int",
    "fare_amount": "double",
    "tip_amount": "double",
    "total_amount": "double",
}
_REQUIRED = ("record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount")
_MANIFEST_TYPES = {
    "int": {"int64", "int32", "integer"},
    "double": {"double", "float64"},
    "ts": {"timestamp", "timestamp[ns]", "timestamp[us]", "timestamp[ms]", "timestamp[s]"},
}
_PAYMENT_CODES = [0, 1, 2, 3, 4, 5, 6]
_MANIFEST_FIELDS = (
    "dataset", "snapshot_id", "snapshot_version", "reporting_period", "file",
    "file_sha256", "row_count", "columns", "units", "schema_changes", "description",
)
_UNIT_EXPECTED = {
    "fare_amount": {"usd", "$"},
    "tip_amount": {"usd", "$"},
    "total_amount": {"usd", "$"},
    "trip_distance": {"miles", "mile", "mi"},
}
_REQUIRED_UNITS = ("fare_amount", "trip_distance")


class _Abstain(Exception):
    """Raised when a verification step fails; carries the abstain reason."""


def _abstain(reason: str) -> dict:
    return {"status": "abstain", "metrics": None, "reason": reason}


def _parse_manifest_ts(value) -> pd.Timestamp:
    if not isinstance(value, str):
        raise _Abstain("reporting_period bounds must be strings")
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise _Abstain("reporting_period bound is not a valid timestamp")
    if pd.isna(ts):
        raise _Abstain("reporting_period bound is not a valid timestamp")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_TZ).tz_localize(None)
    return ts


def _load_manifest(manifest_path: str) -> dict:
    if not os.path.isfile(manifest_path):
        raise _Abstain("manifest file not found")
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError):
        raise _Abstain("manifest is not valid JSON")
    if not isinstance(m, dict):
        raise _Abstain("manifest root is not a JSON object")

    missing = [f for f in _MANIFEST_FIELDS if f not in m]
    if missing:
        raise _Abstain("manifest missing fields: " + ", ".join(missing))

    for f in ("dataset", "snapshot_id", "file", "description"):
        if not isinstance(m[f], str) or not m[f].strip():
            raise _Abstain(f"manifest field {f} must be a non-empty string")

    sv = m["snapshot_version"]
    if isinstance(sv, bool) or not isinstance(sv, (str, int)):
        raise _Abstain("manifest snapshot_version is invalid")

    sha = m["file_sha256"]
    if not isinstance(sha, str) or not _HEX64.fullmatch(sha.lower()):
        raise _Abstain("manifest file_sha256 is not a SHA-256 hex digest")

    rc = m["row_count"]
    if isinstance(rc, bool) or not isinstance(rc, int) or rc < 0:
        raise _Abstain("manifest row_count must be a non-negative integer")

    rp = m["reporting_period"]
    if not isinstance(rp, dict) or any(k not in rp for k in ("start", "end", "timezone")):
        raise _Abstain("manifest reporting_period must contain start, end, timezone")
    if rp["timezone"] != _TZ:
        raise _Abstain("reporting_period timezone must be America/New_York")
    start = _parse_manifest_ts(rp["start"])
    end = _parse_manifest_ts(rp["end"])
    if not start < end:
        raise _Abstain("reporting_period start must be before end")

    cols = m["columns"]
    if not isinstance(cols, list) or not cols:
        raise _Abstain("manifest columns must be a non-empty list")
    col_pairs = []
    for c in cols:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) \
                or not isinstance(c.get("type"), str):
            raise _Abstain("manifest column entries must have string name and type")
        col_pairs.append((c["name"], c["type"].strip().lower()))
    names = [n for n, _ in col_pairs]
    if len(set(names)) != len(names):
        raise _Abstain("manifest column names are not unique")
    for n, t in col_pairs:
        if n in _CONTRACT and t not in _MANIFEST_TYPES[_CONTRACT[n]]:
            raise _Abstain(f"manifest type for {n} conflicts with the data contract")
    missing_cols = [c for c in _CONTRACT if c not in names]
    if missing_cols:
        raise _Abstain("manifest missing contract columns: " + ", ".join(missing_cols))

    units = m["units"]
    if not isinstance(units, dict):
        raise _Abstain("manifest units must be an object")
    for field in _REQUIRED_UNITS:
        if field not in units:
            raise _Abstain(f"manifest units missing {field}")
    for field, ok in _UNIT_EXPECTED.items():
        if field in units and str(units[field]).strip().lower() not in ok:
            raise _Abstain(f"unit for {field} violates the data contract")

    if not isinstance(m["schema_changes"], list):
        raise _Abstain("manifest schema_changes must be a list")

    return {
        "sha256": sha.lower(),
        "row_count": rc,
        "start": start,
        "end": end,
        "columns": names,
        "schema_changes": m["schema_changes"],
    }


def _map_columns(manifest: dict, delivered: list) -> dict:
    """Map each contract column to the delivered column name after schema_changes."""
    names = manifest["columns"]
    rename: dict[str, str] = {}
    for ch in manifest["schema_changes"]:
        if (not isinstance(ch, dict) or ch.get("type") != "rename"
                or not isinstance(ch.get("from"), str) or not isinstance(ch.get("to"), str)):
            raise _Abstain("unsupported schema change in manifest")
        if ch["from"] in rename:
            raise _Abstain("duplicate rename declared in manifest")
        if ch["from"] not in names:
            raise _Abstain("rename source is not a manifest column")
        rename[ch["from"]] = ch["to"]

    expected = {n: rename.get(n, n) for n in names}
    if len(set(expected.values())) != len(expected):
        raise _Abstain("schema changes produce duplicate column names")
    if len(set(delivered)) != len(delivered):
        raise _Abstain("delivered file has duplicate column names")
    absent = [d for d in expected.values() if d not in delivered]
    if absent:
        raise _Abstain("delivered file is missing columns: " + ", ".join(absent))
    return {c: expected[c] for c in _CONTRACT}


def _py_int(v, name: str):
    if v is None:
        return None
    if isinstance(v, bool):
        raise _Abstain(f"{name} has a boolean value")
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if math.isinf(v) or not v.is_integer():
            raise _Abstain(f"{name} has a non-integral or non-finite value")
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if _INT_RE.fullmatch(s):
            return int(s)
        if _INT_DEC_RE.fullmatch(s):
            return int(s.split(".")[0])
        raise _Abstain(f"{name} has a non-integer string value")
    raise _Abstain(f"{name} has an unsupported value")


def _to_int(col, name: str) -> pd.Series:
    t = col.type
    if pa.types.is_integer(t):
        values = col.to_pylist()
    elif pa.types.is_floating(t) or pa.types.is_string(t) or pa.types.is_large_string(t):
        values = [_py_int(v, name) for v in col.to_pylist()]
    else:
        raise _Abstain(f"{name} has an unsupported type")
    try:
        return pd.Series(values, dtype="Int64")
    except (OverflowError, TypeError, ValueError):
        raise _Abstain(f"{name} values are out of int64 range or malformed")


def _to_double(col, name: str) -> pd.Series:
    t = col.type
    if pa.types.is_boolean(t):
        raise _Abstain(f"{name} has a boolean type")
    try:
        if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t):
            casted = pc.cast(col, pa.float64(), safe=False)
        elif pa.types.is_string(t) or pa.types.is_large_string(t):
            casted = pc.cast(col, pa.float64())
        else:
            raise _Abstain(f"{name} has an unsupported type")
    except pa.ArrowException:
        raise _Abstain(f"{name} has values that are not numeric")
    arr = pd.Series(casted.to_pandas(), dtype="float64").to_numpy()
    if pd.Series(arr).isin([float("inf"), float("-inf")]).any():
        raise _Abstain(f"{name} contains infinite values")
    return pd.Series(arr, dtype="float64")


def _to_ts(col, name: str) -> pd.Series:
    t = col.type
    if pa.types.is_timestamp(t):
        try:
            s = pd.Series(col.to_pandas())
        except (pa.ArrowException, ValueError, OverflowError):
            raise _Abstain(f"{name} timestamps are not representable")
    elif pa.types.is_string(t) or pa.types.is_large_string(t):
        raw = pd.Series(col.to_pylist(), dtype=object)
        try:
            s = pd.Series(pd.to_datetime(raw, format="ISO8601"))
        except (ValueError, TypeError, OverflowError):
            raise _Abstain(f"{name} contains unparseable timestamps")
    else:
        raise _Abstain(f"{name} has an unsupported timestamp type")

    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert(_TZ).dt.tz_localize(None)
    if not pd.api.types.is_datetime64_dtype(s.dtype):
        raise _Abstain(f"{name} has inconsistent or mixed timezone offsets")
    try:
        s = s.astype("datetime64[ns]")
    except (OverflowError, ValueError, TypeError):
        raise _Abstain(f"{name} timestamps are out of range")
    return s.reset_index(drop=True)


_NORMALIZERS = {"int": _to_int, "double": _to_double, "ts": _to_ts}


def _read_bytes(data_path: str) -> bytes:
    if not os.path.isfile(data_path):
        raise _Abstain("data file not found")
    try:
        with open(data_path, "rb") as fh:
            return fh.read()
    except OSError:
        raise _Abstain("data file could not be read")


def _read_table(data: bytes) -> pa.Table:
    try:
        pf = pq.ParquetFile(pa.BufferReader(pa.py_buffer(data)))
        meta_rows = pf.metadata.num_rows
        table = pf.read()
    except (pa.ArrowException, OSError, ValueError):
        raise _Abstain("data file is not valid Parquet")
    if meta_rows != table.num_rows:
        raise _Abstain("Parquet footer row count disagrees with data")
    return table


def _analyze(data_path: str, manifest_path: str) -> dict:
    if not isinstance(data_path, str) or not isinstance(manifest_path, str):
        raise _Abstain("paths must be strings")

    # 1. Manifest exists, is valid JSON, and has the required fields.
    man = _load_manifest(manifest_path)

    # 2. SHA-256 of the exact delivered bytes must match the manifest.
    data = _read_bytes(data_path)
    if hashlib.sha256(data).hexdigest() != man["sha256"]:
        raise _Abstain("file SHA-256 does not match manifest file_sha256")

    # 3. Row count must match the manifest.
    table = _read_table(data)
    if table.num_rows != man["row_count"]:
        raise _Abstain("row count does not match manifest row_count")

    # 4. Delivered columns must agree with manifest columns after schema_changes.
    colmap = _map_columns(man, table.column_names)
    series = {}
    for canon, kind in _CONTRACT.items():
        series[canon] = _NORMALIZERS[kind](table.column(colmap[canon]), canon)
    df = pd.DataFrame(series)

    # 6a. Required fields must not be null.
    for req in _REQUIRED:
        if df[req].isna().any():
            raise _Abstain(f"required field {req} contains nulls")

    # 5. Every pickup must lie within the reporting period [start, end).
    pu = df["tpep_pickup_datetime"]
    if ((pu < man["start"]) | (pu >= man["end"])).any():
        raise _Abstain("pickup timestamp outside reporting_period")

    # 6b. Documented codes and valid ranges.
    if not df["payment_type"].isin(_PAYMENT_CODES).all():
        raise _Abstain("payment_type contains undocumented codes")
    dist = df["trip_distance"]
    if not ((dist > 0) & (dist <= 100)).all():
        raise _Abstain("trip_distance outside valid range (0, 100]")
    fare = df["fare_amount"]
    if not ((fare >= 0) & (fare <= 1000)).all():
        raise _Abstain("fare_amount outside valid range [0, 1000]")

    # 6c. Replays (identical rows) count once; same record_id with different values is a conflict.
    unique = df[list(_CONTRACT)].drop_duplicates()
    if unique["record_id"].duplicated().any():
        raise _Abstain("conflicting duplicate record_id with different values")

    n = len(unique)
    if n == 0:
        raise _Abstain("no trips in the delivered data")

    fares = unique["fare_amount"].to_numpy(dtype="float64").tolist()
    distances = unique["trip_distance"].to_numpy(dtype="float64").tolist()
    total_fare = math.fsum(fares)
    mean_fare = total_fare / n
    mean_trip_distance = math.fsum(distances) / n
    card_trips = int((unique["payment_type"] == 1).sum())
    card_share = card_trips / n

    return {
        "status": "ok",
        "metrics": {
            "trip_count": n,
            "total_fare": float(total_fare),
            "mean_fare": float(mean_fare),
            "mean_trip_distance": float(mean_trip_distance),
            "card_share": float(card_share),
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _analyze(data_path, manifest_path)
    except _Abstain as exc:
        return _abstain(str(exc))
    except Exception:
        return _abstain("unexpected error while validating or computing metrics")
