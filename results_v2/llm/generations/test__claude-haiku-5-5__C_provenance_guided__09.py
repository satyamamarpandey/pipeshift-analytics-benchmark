"""Monthly summary of NYC TLC yellow taxi trips with contract verification.

analyze(data_path, manifest_path) verifies the delivered snapshot against its
manifest and the data contract. It returns status "ok" with metrics only when
every verification step passes; otherwise it returns status "abstain" with a
short reason. The module neither prints nor writes files.
"""

import hashlib
import json
import math
import re
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NY_TZ = "America/New_York"

CONTRACT_COLUMNS = (
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
)
REQUIRED_COLUMNS = (
    "record_id",
    "tpep_pickup_datetime",
    "trip_distance",
    "payment_type",
    "fare_amount",
)
REQUIRED_INT_COLUMNS = ("record_id", "payment_type")
DATETIME_COLUMNS = ("tpep_pickup_datetime", "tpep_dropoff_datetime")
FLOAT_COLUMNS = ("trip_distance", "fare_amount", "tip_amount", "total_amount")
COERCIBLE_ONLY_COLUMNS = ("VendorID", "PULocationID", "DOLocationID")
VALID_PAYMENT_TYPES = (0, 1, 2, 3, 4, 5, 6)
MANIFEST_FIELDS = (
    "dataset",
    "snapshot_id",
    "snapshot_version",
    "reporting_period",
    "file",
    "file_sha256",
    "row_count",
    "columns",
    "units",
    "schema_changes",
    "description",
)


class _Abstain(Exception):
    """Raised internally to stop verification with a reason."""


def _fail(reason):
    raise _Abstain(reason)


def _abstain_result(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


# ---------------------------------------------------------------- manifest

def _parse_manifest_ts(value, label):
    if not isinstance(value, str):
        _fail(f"reporting_period.{label} must be a string")
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        _fail(f"reporting_period.{label} is not a valid timestamp")
    if ts is pd.NaT:
        _fail(f"reporting_period.{label} is not a valid timestamp")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    return ts


def _load_manifest(manifest_path):
    try:
        with open(manifest_path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        _fail(f"manifest cannot be read ({exc.__class__.__name__})")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        _fail("manifest is not valid JSON")
    if not isinstance(manifest, dict):
        _fail("manifest top level is not a JSON object")

    missing = [f for f in MANIFEST_FIELDS if f not in manifest]
    if missing:
        _fail("manifest is missing fields: " + ", ".join(missing))

    for f in ("dataset", "snapshot_id", "snapshot_version", "description"):
        if manifest[f] is None:
            _fail(f"manifest field {f} is null")
    if not isinstance(manifest["file"], str) or not manifest["file"]:
        _fail("manifest field file must be a non-empty string")
    if not isinstance(manifest["file_sha256"], str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", manifest["file_sha256"]
    ):
        _fail("manifest file_sha256 is not a SHA-256 hex digest")
    rc = manifest["row_count"]
    if not isinstance(rc, int) or isinstance(rc, bool) or rc < 0:
        _fail("manifest row_count must be a non-negative integer")
    if not isinstance(manifest["columns"], list):
        _fail("manifest columns must be a list")
    if not isinstance(manifest["schema_changes"], list):
        _fail("manifest schema_changes must be a list")
    units = manifest["units"]
    if not isinstance(units, dict):
        _fail("manifest units must be an object")
    td = units.get("trip_distance")
    if td is not None and "mile" not in str(td).lower():
        _fail("manifest units do not declare trip_distance in miles")
    fa = units.get("fare_amount")
    if fa is not None and not any(k in str(fa).lower() for k in ("usd", "dollar")):
        _fail("manifest units do not declare fare_amount in USD")

    period = manifest["reporting_period"]
    if not isinstance(period, dict) or any(
        k not in period for k in ("start", "end", "timezone")
    ):
        _fail("reporting_period must contain start, end and timezone")
    if period["timezone"] != NY_TZ:
        _fail("reporting_period timezone is not America/New_York")
    return manifest


def _period_bounds(manifest):
    period = manifest["reporting_period"]
    start = _parse_manifest_ts(period["start"], "start")
    end = _parse_manifest_ts(period["end"], "end")
    if start >= end:
        _fail("reporting_period start is not before end")
    return start, end


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ schema

def _schema_mapping(manifest, delivered):
    """Return {delivered_name: contract_name}, verifying schema agreement."""
    renames = {}
    for change in manifest["schema_changes"]:
        if (
            not isinstance(change, dict)
            or change.get("type") != "rename"
            or not isinstance(change.get("from"), str)
            or not isinstance(change.get("to"), str)
        ):
            _fail("unsupported schema change in manifest")
        if change["from"] in renames:
            _fail("duplicate rename declared in manifest")
        renames[change["from"]] = change["to"]

    declared = []
    for col in manifest["columns"]:
        if not isinstance(col, dict) or not isinstance(col.get("name"), str) or "type" not in col:
            _fail("manifest columns entries must have name and type")
        declared.append(col["name"])
    if len(set(declared)) != len(declared):
        _fail("manifest declares duplicate column names")
    if any(src not in declared for src in renames):
        _fail("schema_changes refers to a column not declared in manifest")

    expected = [renames.get(name, name) for name in declared]
    if len(set(expected)) != len(expected):
        _fail("schema_changes produce duplicate column names")
    if set(expected) != set(delivered):
        _fail("delivered columns do not match manifest after schema_changes")

    back = {to: frm for frm, to in renames.items()}
    mapping = {name: back.get(name, name) for name in delivered}
    canon_names = list(mapping.values())
    if len(set(canon_names)) != len(canon_names):
        _fail("normalized column names are not unique")
    missing = [c for c in CONTRACT_COLUMNS if c not in canon_names]
    if missing:
        _fail("delivered data lacks contract columns: " + ", ".join(missing))
    return mapping


# ------------------------------------------------------- normalization

def _required_int(s, name):
    if pd.api.types.is_bool_dtype(s):
        _fail(f"{name} has boolean type")
    if pd.api.types.is_integer_dtype(s):
        return s.astype("int64")
    num = pd.to_numeric(s, errors="coerce")
    if num.isna().any():
        _fail(f"{name} has null or non-numeric values")
    if pd.api.types.is_integer_dtype(num):
        return num.astype("int64")
    arr = num.to_numpy(dtype="float64")
    if (
        not np.isfinite(arr).all()
        or not (arr == np.floor(arr)).all()
        or (np.abs(arr) >= 2.0 ** 63).any()
    ):
        _fail(f"{name} has non-integral values")
    return pd.Series(arr.astype("int64"), index=s.index)


def _float_column(s, name):
    if pd.api.types.is_bool_dtype(s):
        _fail(f"{name} has boolean type")
    num = pd.to_numeric(s, errors="coerce")
    arr = pd.Series(num, index=s.index).to_numpy(dtype="float64", na_value=np.nan)
    out = pd.Series(arr, index=s.index)
    if (out.isna() & ~s.isna()).any():
        _fail(f"{name} has non-numeric values")
    if np.isinf(arr).any():
        _fail(f"{name} has infinite values")
    return out


def _check_coercible(s, name):
    if pd.api.types.is_bool_dtype(s):
        _fail(f"{name} has boolean type")
    num = pd.to_numeric(s, errors="coerce")
    if (pd.isna(num) & ~s.isna()).any():
        _fail(f"{name} has non-numeric values")
    return s


def _datetime_column(s, name):
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(NY_TZ).dt.tz_localize(None)
        return s.astype("datetime64[ns]")
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        _fail(f"{name} is numeric, not a timestamp")

    values = []
    for v in s.tolist():
        if v is None or v is pd.NaT or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
            values.append(pd.NaT)
            continue
        if isinstance(v, (bool, int, float, np.number)):
            _fail(f"{name} contains numeric values instead of timestamps")
        try:
            ts = pd.Timestamp(v)
        except (ValueError, TypeError, OverflowError):
            _fail(f"{name} contains unparseable timestamps")
        if ts is pd.NaT:
            values.append(pd.NaT)
            continue
        if ts.tzinfo is not None:
            ts = ts.tz_convert(NY_TZ).tz_localize(None)
        values.append(ts)
    arr = pd.DatetimeIndex(values).to_numpy().astype("datetime64[ns]")
    return pd.Series(arr, index=s.index)


def _normalize_column(s, name):
    if name in REQUIRED_INT_COLUMNS:
        return _required_int(s, name)
    if name in DATETIME_COLUMNS:
        return _datetime_column(s, name)
    if name in FLOAT_COLUMNS:
        return _float_column(s, name)
    if name in COERCIBLE_ONLY_COLUMNS:
        return _check_coercible(s, name)
    return s  # additional declared columns are kept as delivered


# ------------------------------------------------------------- analysis

def _analyze(data_path, manifest_path):
    manifest = _load_manifest(manifest_path)
    start, end = _period_bounds(manifest)

    try:
        actual_sha = _sha256_file(data_path)
    except OSError as exc:
        _fail(f"data file cannot be read ({exc.__class__.__name__})")
    if actual_sha != manifest["file_sha256"].lower():
        _fail("SHA-256 of data file does not match manifest")

    try:
        table = pq.read_table(data_path)
    except Exception:
        _fail("data file cannot be read as Parquet")

    if table.num_rows != manifest["row_count"]:
        _fail("row count does not match manifest")

    names = table.schema.names
    if len(set(names)) != len(names):
        _fail("data file has duplicate column names")
    mapping = _schema_mapping(manifest, names)

    raw = table.to_pandas()
    frame = {canon: _normalize_column(raw[delivered], canon)
             for delivered, canon in mapping.items()}
    df = pd.DataFrame(frame)

    for col in REQUIRED_COLUMNS:
        if df[col].isna().any():
            _fail(f"required column {col} has null values")

    if not df["payment_type"].isin(VALID_PAYMENT_TYPES).all():
        _fail("payment_type contains undocumented codes")

    dist = df["trip_distance"]
    if not ((dist > 0) & (dist <= 100)).all():
        _fail("trip_distance outside valid range (0, 100]")

    fare = df["fare_amount"]
    if not ((fare >= 0) & (fare <= 1000)).all():
        _fail("fare_amount outside valid range [0, 1000]")

    try:
        unique = df.drop_duplicates()
    except TypeError:
        _fail("rows cannot be compared for replay detection")
    if unique["record_id"].duplicated().any():
        _fail("conflicting records share the same record_id")

    pickup = unique["tpep_pickup_datetime"]
    if ((pickup < start) | (pickup >= end)).any():
        _fail("pickup timestamps fall outside reporting_period")

    n = len(unique)
    if n == 0:
        _fail("no trips in reporting period")

    fares = [float(x) for x in unique["fare_amount"].tolist()]
    dists = [float(x) for x in unique["trip_distance"].tolist()]
    total_fare = math.fsum(fares)
    mean_fare = total_fare / n
    mean_trip_distance = math.fsum(dists) / n
    card_trips = int((unique["payment_type"] == 1).sum())
    card_share = card_trips / n

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
    """Verify the snapshot and compute monthly taxi trip metrics.

    Returns {"status": "ok", "metrics": {...}, "reason": ""} when every
    verification step passes, otherwise {"status": "abstain", "metrics": None,
    "reason": "..."}.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _analyze(data_path, manifest_path)
    except _Abstain as exc:
        return _abstain_result(str(exc))
    except Exception as exc:
        return _abstain_result(
            f"unexpected failure during verification ({exc.__class__.__name__})"
        )
