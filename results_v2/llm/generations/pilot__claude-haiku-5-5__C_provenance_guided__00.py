"""Monthly NYC yellow taxi summary with contract verification.

analyze(data_path, manifest_path) verifies the manifest, the data file bytes,
the schema, and the data-contract rules before computing metrics. It abstains
with a reason whenever a check fails. It does not print, prompt, or write files.
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

_TZ = "America/New_York"
_CONTRACT = (
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
_REQUIRED = frozenset(
    {"record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"}
)
_INT_COLS = ("record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type")
_FLOAT_COLS = ("trip_distance", "fare_amount", "tip_amount", "total_amount")
_DT_COLS = ("tpep_pickup_datetime", "tpep_dropoff_datetime")
_PAYMENT_CODES = [0, 1, 2, 3, 4, 5, 6]
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
_SHA_RE = re.compile(r"[0-9a-fA-F]{64}")


class _Abstain(Exception):
    """Raised internally when a verification step fails."""


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _load_manifest(path):
    if not os.path.isfile(path):
        raise _Abstain("manifest file does not exist")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise _Abstain(f"manifest is not valid JSON: {exc}")

    if not isinstance(manifest, dict):
        raise _Abstain("manifest top level is not an object")
    missing = [f for f in _MANIFEST_FIELDS if f not in manifest]
    if missing:
        raise _Abstain(f"manifest missing fields: {', '.join(missing)}")

    rp = manifest["reporting_period"]
    if not isinstance(rp, dict) or not all(
        isinstance(rp.get(k), str) for k in ("start", "end", "timezone")
    ):
        raise _Abstain("manifest reporting_period is malformed")

    sha = manifest["file_sha256"]
    if not isinstance(sha, str) or not _SHA_RE.fullmatch(sha):
        raise _Abstain("manifest file_sha256 is not a SHA-256 hex digest")

    rc = manifest["row_count"]
    if isinstance(rc, bool) or not isinstance(rc, int) or rc < 0:
        raise _Abstain("manifest row_count is not a non-negative integer")

    cols = manifest["columns"]
    if not isinstance(cols, list) or not all(
        isinstance(c, dict) and isinstance(c.get("name"), str) and isinstance(c.get("type"), str)
        for c in cols
    ):
        raise _Abstain("manifest columns are malformed")

    if not isinstance(manifest["schema_changes"], list):
        raise _Abstain("manifest schema_changes is not a list")
    if not isinstance(manifest["file"], str):
        raise _Abstain("manifest file is not a string")
    return manifest


def _manifest_ts(text):
    try:
        ts = pd.Timestamp(text)
    except (ValueError, TypeError) as exc:
        raise _Abstain(f"manifest timestamp is invalid: {exc}")
    if pd.isna(ts):
        raise _Abstain("manifest timestamp is invalid")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_TZ).tz_localize(None)
    return ts


def _reporting_bounds(manifest):
    rp = manifest["reporting_period"]
    if rp["timezone"] != _TZ:
        raise _Abstain(f"reporting_period timezone must be {_TZ}")
    start = _manifest_ts(rp["start"])
    end = _manifest_ts(rp["end"])
    if start >= end:
        raise _Abstain("reporting_period start is not before end")
    return start, end


def _read_bytes(path):
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        raise _Abstain("data file cannot be read")


def _read_table(raw):
    try:
        return pq.read_table(pa.BufferReader(raw))
    except Exception:
        raise _Abstain("data file is not valid Parquet")


def _conform_schema(table, manifest):
    manifest_names = [c["name"] for c in manifest["columns"]]
    if len(set(manifest_names)) != len(manifest_names):
        raise _Abstain("manifest declares duplicate column names")

    fwd, rev = {}, {}
    for change in manifest["schema_changes"]:
        if not isinstance(change, dict) or change.get("type") != "rename":
            raise _Abstain("unsupported schema change in manifest")
        src, dst = change.get("from"), change.get("to")
        if not isinstance(src, str) or not isinstance(dst, str) or src == dst:
            raise _Abstain("malformed rename in schema_changes")
        if src in fwd or dst in rev:
            raise _Abstain("duplicate rename in schema_changes")
        if src not in manifest_names and dst not in manifest_names:
            raise _Abstain("rename in schema_changes refers to an unknown column")
        fwd[src] = dst
        rev[dst] = src

    expected = {fwd.get(n, n) for n in manifest_names}
    delivered = list(table.column_names)
    if len(set(delivered)) != len(delivered):
        raise _Abstain("delivered file has duplicate column names")
    if set(delivered) != expected:
        raise _Abstain("delivered columns do not match manifest after schema_changes")

    canonical = [rev.get(n, n) for n in delivered]
    if len(set(canonical)) != len(canonical) or set(canonical) != set(_CONTRACT):
        raise _Abstain("columns do not match the data contract after normalization")

    df = table.to_pandas()
    df.columns = canonical
    return df.loc[:, list(_CONTRACT)]


def _to_int(s, name, required):
    if pd.api.types.is_bool_dtype(s):
        raise _Abstain(f"{name}: boolean values are not valid integers")
    num = s if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s, errors="coerce")
    if (num.isna() & s.notna()).any():
        raise _Abstain(f"{name}: unparseable integer values")
    if pd.api.types.is_integer_dtype(num):
        out = num.astype("Int64")
    else:
        arr = num.to_numpy(dtype="float64", na_value=np.nan)
        finite = arr[~np.isnan(arr)]
        if not np.isfinite(finite).all() or not (finite == np.floor(finite)).all():
            raise _Abstain(f"{name}: non-integral or non-finite values")
        out = pd.Series(arr, index=s.index).astype("Int64")
    if required and out.isna().any():
        raise _Abstain(f"{name}: required field contains nulls")
    return out


def _to_float(s, name, required):
    if pd.api.types.is_bool_dtype(s):
        raise _Abstain(f"{name}: boolean values are not valid numbers")
    num = s if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s, errors="coerce")
    if (num.isna() & s.notna()).any():
        raise _Abstain(f"{name}: unparseable numeric values")
    arr = num.to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(arr).any():
        raise _Abstain(f"{name}: non-finite values")
    out = pd.Series(arr, index=s.index)
    if required and out.isna().any():
        raise _Abstain(f"{name}: required field contains nulls")
    return out


def _to_naive_ny(s, name, required):
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        out = s.dt.tz_convert(_TZ).dt.tz_localize(None)
    elif pd.api.types.is_datetime64_any_dtype(s):
        out = s
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        raise _Abstain(f"{name}: numeric values are not valid timestamps")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", format="ISO8601")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            parsed = parsed.dt.tz_convert(_TZ).dt.tz_localize(None)
        elif not pd.api.types.is_datetime64_any_dtype(parsed):
            raise _Abstain(f"{name}: timestamps cannot be normalized (mixed offsets)")
        if (parsed.isna() & s.notna()).any():
            raise _Abstain(f"{name}: unparseable timestamp values")
        out = parsed
    if required and out.isna().any():
        raise _Abstain(f"{name}: required field contains nulls")
    return out


def _normalize(df):
    out = pd.DataFrame(index=df.index)
    for c in _INT_COLS:
        out[c] = _to_int(df[c], c, c in _REQUIRED)
    for c in _FLOAT_COLS:
        out[c] = _to_float(df[c], c, c in _REQUIRED)
    for c in _DT_COLS:
        out[c] = _to_naive_ny(df[c], c, c in _REQUIRED)
    return out[list(_CONTRACT)]


def _check_values(df, start, end):
    if not df["payment_type"].isin(_PAYMENT_CODES).all():
        raise _Abstain("payment_type contains undocumented codes")

    d = df["trip_distance"]
    if not ((d > 0) & (d <= 100)).all():
        raise _Abstain("trip_distance outside valid range (0, 100]")

    f = df["fare_amount"]
    if not ((f >= 0) & (f <= 1000)).all():
        raise _Abstain("fare_amount outside valid range [0, 1000]")

    pu = df["tpep_pickup_datetime"]
    if not ((pu >= start) & (pu < end)).all():
        raise _Abstain("pickup timestamps fall outside reporting_period")


def _dedupe(df):
    dup = df["record_id"].duplicated(keep=False)
    if dup.any():
        value_cols = [c for c in df.columns if c != "record_id"]
        nun = df.loc[dup].groupby("record_id")[value_cols].nunique(dropna=False)
        if bool((nun > 1).to_numpy().any()):
            raise _Abstain("conflicting records share the same record_id")
    return df.drop_duplicates(subset="record_id", keep="first").reset_index(drop=True)


def _metrics(df):
    n = len(df)
    if n == 0:
        raise _Abstain("no trips remain after validation")
    fares = [float(v) for v in df["fare_amount"].to_numpy()]
    dists = [float(v) for v in df["trip_distance"].to_numpy()]
    total = math.fsum(fares)
    card_count = int(df["payment_type"].eq(1).sum())
    return {
        "trip_count": n,
        "total_fare": total,
        "mean_fare": total / n,
        "mean_trip_distance": math.fsum(dists) / n,
        "card_share": card_count / n,
    }


def _run(data_path, manifest_path):
    manifest = _load_manifest(manifest_path)
    start, end = _reporting_bounds(manifest)

    raw = _read_bytes(data_path)
    if hashlib.sha256(raw).hexdigest() != manifest["file_sha256"].lower():
        raise _Abstain("SHA-256 of data file does not match file_sha256")

    table = _read_table(raw)
    if table.num_rows != manifest["row_count"]:
        raise _Abstain("row count does not match manifest row_count")

    df = _conform_schema(table, manifest)
    df = _normalize(df)
    _check_values(df, start, end)
    deduped = _dedupe(df)
    return {"status": "ok", "metrics": _metrics(deduped), "reason": ""}


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _run(data_path, manifest_path)
    except _Abstain as exc:
        return _abstain(str(exc))
    except Exception as exc:
        return _abstain(f"unexpected failure: {type(exc).__name__}: {exc}")
