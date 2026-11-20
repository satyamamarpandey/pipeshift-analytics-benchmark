import hashlib
import json
import math
import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_TZ = "America/New_York"
_CHUNK = 1 << 20
_HEX64 = re.compile(r"[0-9a-fA-F]{64}")
_MANIFEST_FIELDS = (
    "dataset", "snapshot_id", "snapshot_version", "reporting_period", "file",
    "file_sha256", "row_count", "columns", "units", "schema_changes", "description",
)
_CONTRACT = {
    "record_id": "int",
    "VendorID": "int",
    "tpep_pickup_datetime": "ts",
    "tpep_dropoff_datetime": "ts",
    "PULocationID": "int",
    "DOLocationID": "int",
    "trip_distance": "float",
    "payment_type": "int",
    "fare_amount": "float",
    "tip_amount": "float",
    "total_amount": "float",
}
_REQUIRED = ("record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount")
_PAYMENT_CODES = [0, 1, 2, 3, 4, 5, 6]


class _Abstain(Exception):
    pass


def _need(cond, reason):
    if not cond:
        raise _Abstain(reason)


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _local_naive(value, label):
    _need(isinstance(value, str), f"{label} is not a string")
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise _Abstain(f"{label} is not a valid timestamp")
    _need(not pd.isna(ts), f"{label} is not a valid timestamp")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_TZ).tz_localize(None)
    return ts


def _load_manifest(manifest_path):
    _need(os.path.isfile(manifest_path), "manifest does not exist")
    with open(manifest_path, "rb") as fh:
        raw = fh.read()
    try:
        m = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise _Abstain("manifest is not valid JSON")
    _need(isinstance(m, dict), "manifest root is not an object")
    missing = [k for k in _MANIFEST_FIELDS if k not in m]
    _need(not missing, "manifest missing fields: " + ", ".join(missing))

    rp = m["reporting_period"]
    _need(isinstance(rp, dict) and all(k in rp for k in ("start", "end", "timezone")),
          "reporting_period malformed")
    _need(rp["timezone"] == _TZ, "reporting_period timezone is not America/New_York")
    start = _local_naive(rp["start"], "reporting_period.start")
    end = _local_naive(rp["end"], "reporting_period.end")
    _need(start < end, "reporting_period is empty or inverted")

    _need(isinstance(m["file"], str), "manifest file is not a string")
    _need(isinstance(m["file_sha256"], str) and _HEX64.fullmatch(m["file_sha256"]) is not None,
          "file_sha256 is not a SHA-256 hex digest")
    rc = m["row_count"]
    _need(isinstance(rc, int) and not isinstance(rc, bool) and rc >= 0, "row_count is invalid")

    cols = m["columns"]
    _need(isinstance(cols, list) and all(
        isinstance(c, dict) and isinstance(c.get("name"), str) and isinstance(c.get("type"), str)
        for c in cols), "columns malformed")
    names = [c["name"] for c in cols]
    _need(len(set(names)) == len(names), "duplicate manifest column names")

    _need(isinstance(m["schema_changes"], list), "schema_changes is not a list")
    return m, start, end


def _sha256_hex(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_schema(manifest, data_path):
    schema = pq.read_schema(data_path)
    delivered = list(schema.names)
    _need(len(set(delivered)) == len(delivered), "delivered file has duplicate column names")

    applied = {c["name"]: c["name"] for c in manifest["columns"]}
    for ch in manifest["schema_changes"]:
        _need(ch.get("type") == "rename" and isinstance(ch.get("from"), str)
              and isinstance(ch.get("to"), str), "unsupported schema change")
        _need(applied.get(ch["from"]) == ch["from"], "rename refers to a column not in manifest")
        applied[ch["from"]] = ch["to"]

    expected = list(applied.values())
    _need(len(set(expected)) == len(expected), "schema changes produce duplicate column names")
    _need(set(expected) == set(delivered),
          "delivered columns do not match manifest columns after schema_changes")

    mapping = {}
    for canon in _CONTRACT:
        _need(canon in applied, f"contract column {canon} missing from manifest")
        mapping[canon] = applied[canon]
    return mapping


def _is_numeric_or_text(t):
    return (pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)
            or pa.types.is_string(t) or pa.types.is_large_string(t))


def _norm_int(arr):
    _need(_is_numeric_or_text(arr.type), "integer column has incompatible type")
    try:
        vals = pc.cast(arr, pa.int64(), safe=True)
    except Exception:
        raise _Abstain("integer column has non-integral or unparseable values")
    nulls = vals.is_null().to_numpy(zero_copy_only=False)
    data = vals.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    return pd.arrays.IntegerArray(data, nulls)


def _norm_float(arr):
    _need(_is_numeric_or_text(arr.type), "float column has incompatible type")
    try:
        vals = pc.cast(arr, pa.float64(), safe=True)
    except Exception:
        raise _Abstain("float column has unparseable values")
    out = vals.to_numpy(zero_copy_only=False).astype(np.float64)
    _need(not np.isinf(out).any(), "float column contains infinite values")
    return out


def _norm_ts(arr):
    t = arr.type
    _need(pa.types.is_timestamp(t) or pa.types.is_string(t) or pa.types.is_large_string(t),
          "timestamp column has incompatible type")
    try:
        s = pd.Series(arr.to_pandas())
        if not pd.api.types.is_datetime64_any_dtype(s.dtype):
            s = pd.to_datetime(s, format="ISO8601")
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            s = s.dt.tz_convert(_TZ).dt.tz_localize(None)
    except Exception:
        raise _Abstain("timestamp column could not be parsed")
    _need(pd.api.types.is_datetime64_any_dtype(s.dtype)
          and not isinstance(s.dtype, pd.DatetimeTZDtype),
          "timestamp column is not a single consistent datetime type")
    return s.to_numpy(dtype="datetime64[ns]")


def _normalize(data_path, mapping):
    table = pq.read_table(data_path, columns=[mapping[c] for c in _CONTRACT])
    cols = {}
    for canon, kind in _CONTRACT.items():
        arr = table.column(mapping[canon]).combine_chunks()
        if kind == "int":
            cols[canon] = _norm_int(arr)
        elif kind == "float":
            cols[canon] = _norm_float(arr)
        else:
            cols[canon] = _norm_ts(arr)
    return pd.DataFrame(cols)


def _validate_and_compute(df, start, end):
    for c in _REQUIRED:
        _need(not df[c].isna().any(), f"required column {c} has nulls")

    d = df["trip_distance"]
    _need(bool(((d > 0) & (d <= 100)).all()), "trip_distance outside (0, 100]")

    f = df["fare_amount"]
    _need(bool(((f >= 0) & (f <= 1000)).all()), "fare_amount outside [0, 1000]")

    pt = df["payment_type"]
    _need(bool(pt.isin(_PAYMENT_CODES).all()), "payment_type has undocumented code")

    p = df["tpep_pickup_datetime"]
    _need(bool(((p >= start) & (p < end)).all()), "pickup timestamp outside reporting period")

    dup = df.duplicated(subset=list(_CONTRACT), keep="first")
    unique = df.loc[~dup]
    _need(not unique["record_id"].duplicated().any(), "conflicting records share a record_id")

    n = int(len(unique))
    _need(n > 0, "no trips in reporting period")

    fares = unique["fare_amount"].to_numpy(dtype=np.float64).tolist()
    dists = unique["trip_distance"].to_numpy(dtype=np.float64).tolist()
    total_fare = math.fsum(fares)
    card_trips = int((unique["payment_type"] == 1).sum())

    return {
        "trip_count": n,
        "total_fare": total_fare,
        "mean_fare": total_fare / n,
        "mean_trip_distance": math.fsum(dists) / n,
        "card_share": card_trips / n,
    }


def _analyze(data_path, manifest_path):
    manifest, start, end = _load_manifest(manifest_path)

    _need(os.path.isfile(data_path), "data file does not exist")
    _need(_sha256_hex(data_path) == manifest["file_sha256"].lower(),
          "file SHA-256 does not match manifest")
    _need(pq.read_metadata(data_path).num_rows == manifest["row_count"],
          "row count does not match manifest")

    mapping = _resolve_schema(manifest, data_path)
    df = _normalize(data_path, mapping)
    metrics = _validate_and_compute(df, start, end)
    return {"status": "ok", "metrics": metrics, "reason": ""}


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except _Abstain as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain(f"unexpected failure: {type(e).__name__}")
