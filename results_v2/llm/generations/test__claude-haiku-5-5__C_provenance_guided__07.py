import hashlib
import json
import math
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

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
_CONTRACT_SET = set(_CONTRACT_COLUMNS)
_REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
_INT_COLUMNS = {"record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"}
_FLOAT_COLUMNS = {"trip_distance", "fare_amount", "tip_amount", "total_amount"}
_TS_COLUMNS = {"tpep_pickup_datetime", "tpep_dropoff_datetime"}
_VALID_PAYMENT_CODES = [0, 1, 2, 3, 4, 5, 6]

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

_OFFSET_RE = re.compile(r"(?:[Zz]|[+-]\d{2}:?\d{2})$")
_SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(manifest_path):
    if not isinstance(manifest_path, str) or not os.path.isfile(manifest_path):
        raise _Abstain("manifest not found")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        raise _Abstain("manifest is not valid JSON")
    if not isinstance(m, dict):
        raise _Abstain("manifest is not a JSON object")
    missing = [k for k in _MANIFEST_FIELDS if k not in m]
    if missing:
        raise _Abstain("manifest missing fields: " + ", ".join(missing))

    for k in ("dataset", "snapshot_id", "file", "file_sha256", "description"):
        if not isinstance(m[k], str):
            raise _Abstain(f"manifest field {k} must be a string")
    if not (isinstance(m["snapshot_version"], str) or _is_int(m["snapshot_version"])):
        raise _Abstain("manifest snapshot_version has invalid type")
    if not _SHA_RE.match(m["file_sha256"]):
        raise _Abstain("manifest file_sha256 is not a SHA-256 hex digest")
    if not _is_int(m["row_count"]) or m["row_count"] < 0:
        raise _Abstain("manifest row_count must be a non-negative integer")

    rp = m["reporting_period"]
    if not isinstance(rp, dict) or not all(
        isinstance(rp.get(k), str) for k in ("start", "end", "timezone")
    ):
        raise _Abstain("manifest reporting_period is malformed")

    cols = m["columns"]
    if not isinstance(cols, list) or not cols:
        raise _Abstain("manifest columns is malformed")
    for c in cols:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or not isinstance(c.get("type"), str):
            raise _Abstain("manifest columns entry is malformed")

    if not isinstance(m["units"], dict):
        raise _Abstain("manifest units is malformed")
    if not isinstance(m["schema_changes"], list):
        raise _Abstain("manifest schema_changes is malformed")
    for ch in m["schema_changes"]:
        if not isinstance(ch, dict):
            raise _Abstain("manifest schema_changes entry is malformed")
    return m


def _parse_period(rp):
    if rp["timezone"] != _TZ:
        raise _Abstain("reporting_period timezone must be America/New_York")
    tz = ZoneInfo(_TZ)

    def parse(s):
        try:
            d = datetime.fromisoformat(s)
        except ValueError:
            raise _Abstain("invalid reporting_period timestamp")
        if d.tzinfo is not None:
            d = d.astimezone(tz).replace(tzinfo=None)
        return d

    start = parse(rp["start"])
    end = parse(rp["end"])
    if start >= end:
        raise _Abstain("reporting_period start is not before end")
    return start, end


def _rename_map(schema_changes):
    rename = {}
    for ch in schema_changes:
        if ch.get("type") != "rename" or not isinstance(ch.get("from"), str) or not isinstance(ch.get("to"), str):
            raise _Abstain("unsupported schema change declaration")
        if ch["to"] in rename and rename[ch["to"]] != ch["from"]:
            raise _Abstain("conflicting schema change declarations")
        rename[ch["to"]] = ch["from"]
    return rename


def _canon(name, rename):
    return rename.get(name, name)


def _to_number(s, name):
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        raise _Abstain(f"{name}: wrong physical type")
    null = s.isna()
    if not pd.api.types.is_numeric_dtype(s):
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    num = pd.to_numeric(s, errors="coerce")
    if bool((num.isna() & ~null).any()):
        raise _Abstain(f"{name}: non-numeric values present")
    return num


def _to_int(s, name, nullable):
    num = _to_number(s, name)
    if pd.api.types.is_integer_dtype(num):
        out = num.astype("Int64")
    else:
        arr = num.to_numpy(dtype="float64", na_value=np.nan)
        notnull = ~np.isnan(arr)
        vals = arr[notnull]
        if not np.all(np.isfinite(vals)) or not np.all(np.mod(vals, 1.0) == 0):
            raise _Abstain(f"{name}: non-integral values present")
        out = pd.Series(arr, index=num.index).astype("Int64")
    if not nullable and bool(out.isna().any()):
        raise _Abstain(f"{name}: required values are null")
    return out


def _to_float(s, name, nullable):
    num = _to_number(s, name)
    arr = num.to_numpy(dtype="float64", na_value=np.nan)
    notnull = ~np.isnan(arr)
    if not nullable and not notnull.all():
        raise _Abstain(f"{name}: required values are null")
    if not np.all(np.isfinite(arr[notnull])):
        raise _Abstain(f"{name}: non-finite values present")
    return pd.Series(arr, index=num.index)


def _to_local_naive(s, name, nullable):
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is not None:
            out = s.dt.tz_convert(_TZ).dt.tz_localize(None)
        else:
            out = s
    else:
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            raise _Abstain(f"{name}: wrong physical type")
        strs = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        null = strs.isna()
        nonnull = strs[~null].astype(str)
        has_offset = nonnull.map(lambda v: _OFFSET_RE.search(v) is not None)
        if len(nonnull) > 0 and bool(has_offset.all()):
            aware = pd.to_datetime(strs, format="ISO8601", errors="coerce", utc=True)
            if bool((aware.isna() & ~null).any()):
                raise _Abstain(f"{name}: unparseable timestamps")
            out = aware.dt.tz_convert(_TZ).dt.tz_localize(None)
        elif len(nonnull) > 0 and bool((~has_offset).all()):
            naive = pd.to_datetime(strs, format="ISO8601", errors="coerce")
            if bool((naive.isna() & ~null).any()):
                raise _Abstain(f"{name}: unparseable timestamps")
            out = naive
        elif len(nonnull) == 0:
            out = pd.to_datetime(strs, format="ISO8601", errors="coerce")
        else:
            raise _Abstain(f"{name}: mixed timezone-aware and naive timestamps")
    if not nullable and bool(out.isna().any()):
        raise _Abstain(f"{name}: required values are null")
    return out


def _check_units(units):
    d_units = str(units.get("trip_distance", "")).strip().lower()
    f_units = str(units.get("fare_amount", "")).strip().lower()
    if not ("mile" in d_units or d_units == "mi"):
        raise _Abstain("manifest units do not declare trip_distance in miles")
    if not ("usd" in f_units or "$" in f_units or "dollar" in f_units):
        raise _Abstain("manifest units do not declare fare_amount in USD")


def _analyze(data_path, manifest_path):
    # Step 1: manifest
    manifest = _load_manifest(manifest_path)

    # Step 2: file hash
    if not isinstance(data_path, str) or not os.path.isfile(data_path):
        raise _Abstain("data file not found")
    if _sha256_file(data_path) != manifest["file_sha256"].lower():
        raise _Abstain("file SHA-256 does not match manifest")

    # Step 3: row count
    table = pq.read_table(data_path)
    if table.num_rows != manifest["row_count"]:
        raise _Abstain("row count does not match manifest")

    # Step 4: schema agreement after applying schema_changes
    rename = _rename_map(manifest["schema_changes"])
    delivered = list(table.column_names)
    if len(set(delivered)) != len(delivered):
        raise _Abstain("duplicate column names in data file")
    canon_delivered = [_canon(c, rename) for c in delivered]
    if len(set(canon_delivered)) != len(canon_delivered):
        raise _Abstain("ambiguous column names after schema changes")
    if set(canon_delivered) != _CONTRACT_SET:
        raise _Abstain("delivered columns do not match the data contract")
    manifest_names = [_canon(c["name"], rename) for c in manifest["columns"]]
    if len(set(manifest_names)) != len(manifest_names) or set(manifest_names) != _CONTRACT_SET:
        raise _Abstain("manifest columns do not match the data contract")

    # Reporting period and units
    start, end = _parse_period(manifest["reporting_period"])
    _check_units(manifest["units"])

    # Normalize schema
    df = table.to_pandas()
    df.columns = canon_delivered
    df = df.loc[:, _CONTRACT_COLUMNS]

    out = pd.DataFrame(index=df.index)
    for c in _CONTRACT_COLUMNS:
        nullable = c not in _REQUIRED
        s = df[c]
        if c in _TS_COLUMNS:
            out[c] = _to_local_naive(s, c, nullable)
        elif c in _INT_COLUMNS:
            out[c] = _to_int(s, c, nullable)
        else:
            out[c] = _to_float(s, c, nullable)

    # Required fields
    for c in _REQUIRED:
        if bool(out[c].isna().any()):
            raise _Abstain(f"required field {c} is null")

    # Step 5: pickup within reporting period (start inclusive, end exclusive)
    pickup = out["tpep_pickup_datetime"]
    in_period = (pickup >= pd.Timestamp(start)) & (pickup < pd.Timestamp(end))
    if not bool(in_period.all()):
        raise _Abstain("pickup timestamps outside reporting period")

    # Step 6: documented codes and valid ranges
    if not bool(out["payment_type"].isin(_VALID_PAYMENT_CODES).all()):
        raise _Abstain("undocumented payment_type code")
    dist = out["trip_distance"]
    if not bool(((dist > 0) & (dist <= 100)).all()):
        raise _Abstain("trip_distance outside valid range (0, 100]")
    fare = out["fare_amount"]
    if not bool(((fare >= 0) & (fare <= 1000)).all()):
        raise _Abstain("fare_amount outside valid range [0, 1000]")

    # Replays (identical rows) count once; same record_id with differing values is a conflict
    dedup = out.drop_duplicates(subset=_CONTRACT_COLUMNS, keep="first")
    if bool(dedup["record_id"].duplicated().any()):
        raise _Abstain("conflicting duplicate record_id values")

    n = len(dedup)
    if n == 0:
        raise _Abstain("no trips in reporting period")

    fares = dedup["fare_amount"].astype("float64").tolist()
    distances = dedup["trip_distance"].astype("float64").tolist()
    total_fare = math.fsum(fares)
    mean_fare = total_fare / n
    mean_trip_distance = math.fsum(distances) / n
    card_trips = int((dedup["payment_type"] == 1).sum())
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
    try:
        return _analyze(data_path, manifest_path)
    except _Abstain as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain(f"could not process inputs ({type(e).__name__})")
