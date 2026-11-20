"""Monthly summary metrics for NYC TLC yellow taxi trip snapshots.

analyze(data_path, manifest_path) runs the verification protocol, normalizes the
schema, and returns either metrics or an abstention. It never prints, prompts,
or writes files.
"""

import hashlib
import json
import math
import os
import re
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TZ_NAME = "America/New_York"

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
CONTRACT_SET = frozenset(CONTRACT_COLUMNS)
REQUIRED_COLUMNS = (
    "record_id",
    "tpep_pickup_datetime",
    "trip_distance",
    "payment_type",
    "fare_amount",
)
INT64_COLS = frozenset({"record_id", "payment_type"})
INT32_COLS = frozenset({"VendorID", "PULocationID", "DOLocationID"})
FLOAT_COLS = frozenset({"trip_distance", "fare_amount", "tip_amount", "total_amount"})
DATETIME_COLS = frozenset({"tpep_pickup_datetime", "tpep_dropoff_datetime"})
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

_INT_TYPES = {"int64", "int32", "int", "integer", "long", "bigint"}
_FLOAT_TYPES = {"double", "float64", "float", "number", "numeric"}
_DISTANCE_UNITS = {"miles", "mile", "mi"}
_CURRENCY_UNITS = {"usd", "dollar", "dollars", "us dollars"}
_SHA_RE = re.compile(r"[0-9a-fA-F]{64}")


class _Abstain(Exception):
    """Raised internally to abstain with a short reason."""


def _fail(reason):
    raise _Abstain(reason)


def _parse_local(value):
    """Parse a manifest timestamp into a naive New York local time, or None."""
    if not isinstance(value, str):
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert(TZ_NAME).tz_localize(None)
    return ts


def _load_manifest(manifest_path):
    if not isinstance(manifest_path, (str, os.PathLike)) or not os.path.isfile(manifest_path):
        _fail("manifest does not exist")
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        _fail("manifest is not valid JSON")
    if not isinstance(manifest, dict):
        _fail("manifest is not a JSON object")
    missing = [f for f in MANIFEST_FIELDS if f not in manifest]
    if missing:
        _fail("manifest is missing fields: " + ", ".join(missing))
    return manifest


def _validate_manifest(m):
    rp = m["reporting_period"]
    if not isinstance(rp, dict) or any(k not in rp for k in ("start", "end", "timezone")):
        _fail("reporting_period is malformed")
    if rp["timezone"] != TZ_NAME:
        _fail("reporting_period timezone is not America/New_York")
    start = _parse_local(rp["start"])
    end = _parse_local(rp["end"])
    if start is None or end is None or start >= end:
        _fail("reporting_period is invalid")

    if not isinstance(m["file"], str) or not m["file"]:
        _fail("manifest file field is malformed")

    sha = m["file_sha256"]
    if not isinstance(sha, str) or not _SHA_RE.fullmatch(sha):
        _fail("manifest file_sha256 is malformed")

    rc = m["row_count"]
    if isinstance(rc, bool) or not isinstance(rc, int) or rc < 0:
        _fail("manifest row_count is malformed")

    cols = m["columns"]
    if (not isinstance(cols, list) or not cols
            or not all(isinstance(c, dict) and isinstance(c.get("name"), str)
                       and isinstance(c.get("type"), str) for c in cols)):
        _fail("manifest columns are malformed")

    units = m["units"]
    if not isinstance(units, dict):
        _fail("manifest units must be an object")
    required_units = {"trip_distance": _DISTANCE_UNITS, "fare_amount": _CURRENCY_UNITS}
    optional_units = {"tip_amount": _CURRENCY_UNITS, "total_amount": _CURRENCY_UNITS}
    for key, allowed in required_units.items():
        if key not in units:
            _fail(f"manifest units do not declare {key}")
        if not isinstance(units[key], str) or units[key].strip().lower() not in allowed:
            _fail(f"unexpected unit for {key}")
    for key, allowed in optional_units.items():
        if key in units and (not isinstance(units[key], str)
                             or units[key].strip().lower() not in allowed):
            _fail(f"unexpected unit for {key}")

    changes = m["schema_changes"]
    if not isinstance(changes, list):
        _fail("schema_changes must be a list")
    fwd = {}
    for ch in changes:
        if (not isinstance(ch, dict) or ch.get("type") != "rename"
                or not isinstance(ch.get("from"), str) or not isinstance(ch.get("to"), str)):
            _fail("unsupported schema change")
        if ch["from"] in fwd:
            _fail("duplicate rename source in schema_changes")
        fwd[ch["from"]] = ch["to"]
    if len(set(fwd.values())) != len(fwd):
        _fail("schema_changes rename targets collide")
    if any(t in fwd for t in fwd.values()):
        _fail("chained renames are not supported")
    inv = {v: k for k, v in fwd.items()}

    return {
        "start": start,
        "end": end,
        "sha": sha.lower(),
        "row_count": rc,
        "columns": cols,
        "fwd": fwd,
        "inv": inv,
    }


def _manifest_type_ok(contract, declared):
    t = declared.strip().lower()
    if t == "string":
        return True
    if contract in INT64_COLS or contract in INT32_COLS:
        return t in _INT_TYPES
    if contract in FLOAT_COLS:
        return t in _FLOAT_TYPES or t in _INT_TYPES
    if contract in DATETIME_COLS:
        return t == "datetime" or t.startswith("timestamp")
    return False


def _check_columns(names, mcols, fwd, inv):
    """Return a map from delivered column name to contract column name."""
    if len(set(names)) != len(names):
        _fail("duplicate column names in data file")
    mnames = [c["name"] for c in mcols]
    if len(set(mnames)) != len(mnames):
        _fail("duplicate column names in manifest")
    expected = {fwd.get(n, n) for n in mnames}
    if expected != set(names):
        _fail("delivered columns do not match manifest after schema_changes")
    canon = {n: inv.get(n, n) for n in names}
    values = set(canon.values())
    if values != CONTRACT_SET or len(values) != len(names):
        _fail("delivered columns do not match the data contract")
    for c in mcols:
        contract = inv.get(c["name"], c["name"])
        if contract not in CONTRACT_SET or not _manifest_type_ok(contract, c["type"]):
            _fail(f"manifest type for {c['name']} is incompatible with the contract")
    return canon


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _arrow_to_pandas_dtype(arrow_type):
    if pa.types.is_integer(arrow_type):
        return pd.Int64Dtype()
    if pa.types.is_floating(arrow_type):
        return pd.Float64Dtype()
    return None


def _coerce_object(s):
    if pd.api.types.is_object_dtype(s.dtype) or isinstance(s.dtype, pd.StringDtype):
        try:
            s = pd.to_numeric(s, errors="raise")
        except (ValueError, TypeError):
            _fail("value cannot be converted to a number")
    return s


def _to_int(s, is32):
    s = _coerce_object(s)
    if pd.api.types.is_bool_dtype(s.dtype):
        _fail("boolean value in integer column")
    if pd.api.types.is_integer_dtype(s.dtype):
        out = s.astype("Int64")
    elif pd.api.types.is_float_dtype(s.dtype):
        vals = s.to_numpy(dtype="float64", na_value=np.nan)
        present = ~np.isnan(vals)
        v = vals[present]
        if (not np.isfinite(v).all()
                or not np.all(v == np.floor(v))
                or np.any(np.abs(v) >= 2.0 ** 63)):
            _fail("non-integer value in integer column")
        ints = np.zeros(len(vals), dtype=np.int64)
        ints[present] = v.astype(np.int64)
        out = pd.Series(pd.arrays.IntegerArray(ints, mask=~present), index=s.index)
    else:
        _fail("unsupported type for integer column")
    if is32:
        nn = out.dropna()
        if len(nn) and (int(nn.min()) < -(2 ** 31) or int(nn.max()) > 2 ** 31 - 1):
            _fail("value outside int32 range")
    return out


def _to_float(s):
    s = _coerce_object(s)
    if pd.api.types.is_bool_dtype(s.dtype) or not (
        pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_float_dtype(s.dtype)
    ):
        _fail("unsupported type for numeric column")
    vals = s.to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(vals).any():
        _fail("infinite value in numeric column")
    return pd.Series(vals, index=s.index)


def _to_local_naive(s):
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(TZ_NAME).dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return s
    if pd.api.types.is_object_dtype(s.dtype) or isinstance(s.dtype, pd.StringDtype):
        try:
            parsed = pd.to_datetime(s, format="ISO8601", errors="raise")
        except (ValueError, TypeError):
            _fail("timestamp cannot be parsed")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            return parsed.dt.tz_convert(TZ_NAME).dt.tz_localize(None)
        if pd.api.types.is_datetime64_any_dtype(parsed.dtype):
            return parsed
    _fail("unsupported type for timestamp column")


def _normalize_column(s, col):
    if col in INT64_COLS:
        return _to_int(s, is32=False)
    if col in INT32_COLS:
        return _to_int(s, is32=True)
    if col in FLOAT_COLS:
        return _to_float(s)
    return _to_local_naive(s)


def _analyze(data_path, manifest_path):
    manifest = _load_manifest(manifest_path)
    spec = _validate_manifest(manifest)

    if not isinstance(data_path, (str, os.PathLike)) or not os.path.isfile(data_path):
        _fail("data file does not exist")
    if _sha256_file(data_path) != spec["sha"]:
        _fail("data file SHA-256 does not match manifest")

    try:
        table = pq.read_table(data_path)
    except Exception:
        _fail("data file is not readable Parquet")

    if table.num_rows != spec["row_count"]:
        _fail("row count does not match manifest")

    names = list(table.column_names)
    canon = _check_columns(names, spec["columns"], spec["fwd"], spec["inv"])

    pdf = table.to_pandas(types_mapper=_arrow_to_pandas_dtype, ignore_metadata=True)
    pdf = pdf.reset_index(drop=True)

    raw = {canon[n]: pdf[n] for n in names}
    data = {c: _normalize_column(raw[c], c) for c in CONTRACT_COLUMNS}
    frame = pd.DataFrame(data, columns=list(CONTRACT_COLUMNS))

    # Required, non-null fields.
    for c in REQUIRED_COLUMNS:
        if frame[c].isna().any():
            _fail(f"required field {c} contains nulls")

    # Every pickup must lie in [start, end) local time.
    pickup = frame["tpep_pickup_datetime"]
    if ((pickup < spec["start"]) | (pickup >= spec["end"])).any():
        _fail("pickup timestamp outside reporting_period")

    # Documented codes and valid ranges.
    if not frame["payment_type"].isin(list(VALID_PAYMENT_TYPES)).all():
        _fail("undocumented payment_type code")
    dist = frame["trip_distance"]
    if not ((dist > 0) & (dist <= 100)).all():
        _fail("trip_distance outside valid range (0, 100]")
    fare = frame["fare_amount"]
    if not ((fare >= 0) & (fare <= 1000)).all():
        _fail("fare_amount outside valid range [0, 1000]")

    # Replays (identical rows) count once; same record_id with differing values is a conflict.
    df = frame.drop_duplicates(ignore_index=True)
    if df["record_id"].duplicated().any():
        _fail("conflicting duplicate record_id")

    n = len(df)
    if n == 0:
        _fail("no trips in reporting period")

    fares = df["fare_amount"].tolist()
    distances = df["trip_distance"].tolist()
    total_fare = math.fsum(fares)
    card_trips = int((df["payment_type"] == 1).sum())

    metrics = {
        "trip_count": int(n),
        "total_fare": total_fare,
        "mean_fare": total_fare / n,
        "mean_trip_distance": math.fsum(distances) / n,
        "card_share": card_trips / n,
    }
    return {"status": "ok", "metrics": metrics, "reason": ""}


def analyze(data_path: str, manifest_path: str) -> dict:
    """Validate a taxi trip snapshot and compute monthly summary metrics.

    Returns {"status": "ok", "metrics": {...}, "reason": ""} when every
    verification step passes, otherwise {"status": "abstain", "metrics": None,
    "reason": "<short explanation>"}.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _analyze(data_path, manifest_path)
    except _Abstain as exc:
        return {"status": "abstain", "metrics": None, "reason": str(exc)}
    except Exception:
        return {
            "status": "abstain",
            "metrics": None,
            "reason": "unexpected error while validating or reading inputs",
        }
