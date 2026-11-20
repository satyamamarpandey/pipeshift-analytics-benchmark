"""Monthly summary of NYC TLC yellow taxi trips with contract verification.

analyze(data_path, manifest_path) returns either a status "ok" result with
metrics, or a status "abstain" result with a reason. It never prints, prompts,
or writes files.
"""

import hashlib
import json
import math
import warnings

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
_REQUIRED_COLUMNS = [
    "record_id",
    "tpep_pickup_datetime",
    "trip_distance",
    "payment_type",
    "fare_amount",
]
_INT_COLUMNS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"]
_FLOAT_COLUMNS = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
_DATETIME_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
_PAYMENT_CODES = [0, 1, 2, 3, 4, 5, 6]
_CARD_PAYMENT = 1

_MANIFEST_FIELDS = (
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
_ACCEPTED_DISTANCE_UNITS = {"mile", "miles", "mi"}
_ACCEPTED_CURRENCY_UNITS = {"usd", "dollars", "$"}


class _Abstain(Exception):
    """Raised internally when a verification step fails."""


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _load_manifest(manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError:
        raise _Abstain("manifest file not found")
    except (OSError, ValueError):
        raise _Abstain("manifest is not readable valid JSON")

    if not isinstance(manifest, dict):
        raise _Abstain("manifest is not a JSON object")
    missing = [f for f in _MANIFEST_FIELDS if f not in manifest]
    if missing:
        raise _Abstain("manifest missing fields: " + ", ".join(missing))

    period = manifest["reporting_period"]
    if not isinstance(period, dict) or not all(
        k in period for k in ("start", "end", "timezone")
    ):
        raise _Abstain("manifest reporting_period is malformed")
    if period["timezone"] != _TZ:
        raise _Abstain("manifest reporting_period timezone is not America/New_York")

    columns = manifest["columns"]
    if not isinstance(columns, list) or not all(
        isinstance(c, dict) and isinstance(c.get("name"), str) for c in columns
    ):
        raise _Abstain("manifest columns are malformed")

    if not isinstance(manifest["schema_changes"], list):
        raise _Abstain("manifest schema_changes is malformed")

    row_count = manifest["row_count"]
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise _Abstain("manifest row_count is not a non-negative integer")

    if not isinstance(manifest["file_sha256"], str):
        raise _Abstain("manifest file_sha256 is malformed")

    units = manifest["units"]
    if not isinstance(units, dict):
        raise _Abstain("manifest units cannot be verified")
    distance_unit = units.get("trip_distance")
    fare_unit = units.get("fare_amount")
    if not isinstance(distance_unit, str) or not isinstance(fare_unit, str):
        raise _Abstain("manifest units for trip_distance or fare_amount are missing")
    if distance_unit.strip().lower() not in _ACCEPTED_DISTANCE_UNITS:
        raise _Abstain("manifest trip_distance unit is not miles")
    if fare_unit.strip().lower() not in _ACCEPTED_CURRENCY_UNITS:
        raise _Abstain("manifest fare_amount unit is not USD")

    return manifest


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_mapping(manifest, delivered):
    """Return {delivered_name: canonical_name} or abstain on any mismatch."""
    manifest_cols = [c["name"] for c in manifest["columns"]]
    if len(set(manifest_cols)) != len(manifest_cols):
        raise _Abstain("manifest lists duplicate column names")

    renames = {}
    for change in manifest["schema_changes"]:
        if not isinstance(change, dict) or change.get("type") != "rename":
            raise _Abstain("unsupported schema change in manifest")
        src, dst = change.get("from"), change.get("to")
        if not isinstance(src, str) or not isinstance(dst, str):
            raise _Abstain("malformed rename in schema_changes")
        if src not in manifest_cols or src in renames:
            raise _Abstain("rename source is not a unique manifest column")
        renames[src] = dst
    if len(set(renames.values())) != len(renames):
        raise _Abstain("schema_changes rename targets are not unique")

    expected = [renames.get(c, c) for c in manifest_cols]
    if len(set(expected)) != len(expected):
        raise _Abstain("column names collide after applying schema_changes")
    if sorted(expected) != sorted(delivered):
        raise _Abstain("delivered columns do not match manifest after schema_changes")

    inverse = {dst: src for src, dst in renames.items()}
    mapping = {d: inverse.get(d, d) for d in delivered}
    canonical = set(mapping.values())
    if not set(_CONTRACT_COLUMNS).issubset(canonical):
        raise _Abstain("delivered data lacks required contract columns")
    return mapping


def _coerce_numeric(s, name):
    if pd.api.types.is_datetime64_any_dtype(s):
        raise _Abstain(f"{name}: expected numeric values, found timestamps")
    out = pd.to_numeric(s, errors="coerce")
    if (out.isna() & s.notna()).any():
        raise _Abstain(f"{name}: contains non-numeric values")
    return out


def _int_col(s, name):
    out = _coerce_numeric(s, name)
    if pd.api.types.is_float_dtype(out):
        vals = out.dropna().to_numpy(dtype="float64")
        if not np.isfinite(vals).all() or not np.all(vals == np.floor(vals)):
            raise _Abstain(f"{name}: contains non-integer values")
    return out.astype("Int64")


def _float_col(s, name):
    out = _coerce_numeric(s, name).astype("float64")
    vals = out.dropna().to_numpy(dtype="float64")
    if not np.isfinite(vals).all():
        raise _Abstain(f"{name}: contains non-finite values")
    return out


def _datetime_col(s, name):
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(_TZ).dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        parsed = pd.to_datetime(s, errors="coerce")
    except (TypeError, ValueError):
        raise _Abstain(f"{name}: timestamps cannot be parsed")
    if (parsed.isna() & s.notna()).any():
        raise _Abstain(f"{name}: contains unparseable timestamps")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert(_TZ).dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(parsed):
        return parsed
    raise _Abstain(f"{name}: timestamps have mixed or unsupported types")


def _naive_local(value, label):
    if not isinstance(value, str):
        raise _Abstain(f"reporting_period {label} is not a string timestamp")
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise _Abstain(f"reporting_period {label} is invalid")
    if pd.isna(ts):
        raise _Abstain(f"reporting_period {label} is invalid")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_TZ).tz_localize(None)
    return ts


def _analyze(data_path, manifest_path):
    manifest = _load_manifest(manifest_path)

    # Step 2: file integrity.
    if _sha256(data_path) != manifest["file_sha256"].strip().lower():
        raise _Abstain("SHA-256 of data file does not match manifest")

    # Step 3: row count. Step 4: columns after schema_changes.
    table = pq.read_table(data_path)
    if table.num_rows != manifest["row_count"]:
        raise _Abstain("row count does not match manifest")
    mapping = _column_mapping(manifest, table.column_names)

    raw = table.to_pandas()
    raw.columns = [mapping[c] for c in table.column_names]

    # Normalize schema (types may be strings, timestamps may be tz-aware).
    cols = {}
    for c in _CONTRACT_COLUMNS:
        series = raw[c]
        if c in _INT_COLUMNS:
            cols[c] = _int_col(series, c)
        elif c in _FLOAT_COLUMNS:
            cols[c] = _float_col(series, c)
        elif c in _DATETIME_COLUMNS:
            cols[c] = _datetime_col(series, c)
        else:
            raise _Abstain(f"no normalization rule for column {c}")
    df = pd.DataFrame(cols)

    # Step 6: required, non-null fields.
    for c in _REQUIRED_COLUMNS:
        if df[c].isna().any():
            raise _Abstain(f"required field {c} contains nulls")

    # Documented codes and valid ranges.
    if not df["payment_type"].isin(_PAYMENT_CODES).all():
        raise _Abstain("payment_type contains undocumented codes")
    td = df["trip_distance"]
    if not ((td > 0) & (td <= 100)).all():
        raise _Abstain("trip_distance outside valid range (0, 100]")
    fare = df["fare_amount"]
    if not ((fare >= 0) & (fare <= 1000)).all():
        raise _Abstain("fare_amount outside valid range [0, 1000]")

    # Step 5: pickup timestamps within the reporting period (local time).
    period = manifest["reporting_period"]
    start = _naive_local(period["start"], "start")
    end = _naive_local(period["end"], "end")
    if not start < end:
        raise _Abstain("reporting_period start is not before end")
    pickup = df["tpep_pickup_datetime"]
    if not ((pickup >= start) & (pickup < end)).all():
        raise _Abstain("pickup timestamps fall outside reporting_period")

    # Replays count once; same record_id with different values is a conflict.
    dedup = df.drop_duplicates()
    if dedup["record_id"].duplicated().any():
        raise _Abstain("conflicting duplicate record_id with differing values")

    n = len(dedup)
    if n == 0:
        raise _Abstain("no trips remain after replay handling")

    fares = dedup["fare_amount"].tolist()
    distances = dedup["trip_distance"].tolist()
    payments = dedup["payment_type"].tolist()

    total_fare = math.fsum(fares)
    mean_fare = total_fare / n
    mean_trip_distance = math.fsum(distances) / n
    card_trips = sum(1 for p in payments if p == _CARD_PAYMENT)
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
    except Exception as exc:  # any unexpected failure means no trustworthy result
        return _abstain(f"verification failed unexpectedly: {type(exc).__name__}")
