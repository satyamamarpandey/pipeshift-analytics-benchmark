import hashlib
import json
import math
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_TZ = "America/New_York"
_REQUIRED_MANIFEST = [
    "dataset", "snapshot_id", "snapshot_version", "reporting_period", "file",
    "file_sha256", "row_count", "columns", "units", "schema_changes", "description",
]
_CONTRACT = [
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]
_REQUIRED_NONNULL = [
    "record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount",
]
_INT_COLS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"]
_FLOAT_COLS = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
_TS_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
_TZ_SUFFIX = re.compile(r"(Z|[+-]\d{2}(:?\d{2})?)$", re.IGNORECASE)


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_local_naive_scalar(value, tz):
    try:
        ts = pd.Timestamp(value)
    except Exception:
        raise _Abstain(f"unparseable reporting_period bound: {value!r}")
    if pd.isna(ts):
        raise _Abstain("reporting_period bound is null")
    if ts.tzinfo is None:
        try:
            ts = ts.tz_localize(tz, ambiguous="raise", nonexistent="raise")
        except Exception:
            raise _Abstain("reporting_period bound cannot be localized")
    return ts.tz_convert(_TZ).tz_localize(None)


def _parse_ts(series, name):
    s = series
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(_TZ).dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return pd.to_datetime(s)
    nonnull = s.dropna()
    if len(nonnull) and not all(isinstance(v, (str, pd.Timestamp)) or hasattr(v, "year")
                                for v in nonnull):
        raise _Abstain(f"column {name} has non-datetime values")
    strs = nonnull.map(lambda v: v.strip() if isinstance(v, str) else v)
    if (strs.map(lambda v: isinstance(v, str) and v == "")).any():
        raise _Abstain(f"column {name} has empty timestamp strings")

    def has_tz(v):
        if isinstance(v, str):
            body = v[10:] if len(v) > 10 else ""
            return bool(_TZ_SUFFIX.search(body))
        return getattr(v, "tzinfo", None) is not None

    flags = strs.map(has_tz) if len(strs) else pd.Series([], dtype=bool)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if len(strs) == 0:
        return out
    try:
        if flags.all():
            parsed = pd.to_datetime(strs, utc=True, format="mixed", errors="coerce")
            parsed = parsed.dt.tz_convert(_TZ).dt.tz_localize(None)
        elif not flags.any():
            parsed = pd.to_datetime(strs, format="mixed", errors="coerce")
        else:
            raise _Abstain(f"column {name} mixes timezone-aware and naive timestamps")
    except _Abstain:
        raise
    except Exception:
        raise _Abstain(f"column {name} has unparseable timestamps")
    if parsed.isna().any():
        raise _Abstain(f"column {name} has unparseable timestamps")
    out.loc[parsed.index] = parsed.astype("datetime64[ns]").values
    return out


def _parse_num(series, name, integer):
    s = series
    if pd.api.types.is_bool_dtype(s.dtype):
        raise _Abstain(f"column {name} has boolean values")
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        s2 = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        s2 = s2.map(lambda v: None if isinstance(v, str) and v == "" else v)
        if s2.map(lambda v: isinstance(v, bool)).any():
            raise _Abstain(f"column {name} has boolean values")
        num = pd.to_numeric(s2, errors="coerce")
        if (num.isna() & s2.notna()).any():
            raise _Abstain(f"column {name} has non-numeric values")
    else:
        num = pd.to_numeric(s, errors="coerce")
        if (num.isna() & s.notna()).any():
            raise _Abstain(f"column {name} has non-numeric values")
    num = num.astype("float64") if not integer else num
    vals = pd.to_numeric(num, errors="coerce").astype("float64")
    if np.isinf(vals.dropna()).any():
        raise _Abstain(f"column {name} has infinite values")
    if integer:
        nn = vals.dropna()
        if (nn != np.floor(nn)).any():
            raise _Abstain(f"column {name} has non-integer values")
        if len(nn) and (nn.abs() > 2 ** 53).any():
            # fall back to exact integer handling
            try:
                return pd.array(num, dtype="Int64")
            except Exception:
                raise _Abstain(f"column {name} has invalid integers")
        return pd.Series(vals, index=s.index).round().astype("Int64")
    return vals


def _unit_factor(units, col, kind):
    if not isinstance(units, dict):
        return 1.0
    u = units.get(col)
    if u is None:
        return 1.0
    if not isinstance(u, str):
        raise _Abstain(f"unrecognized unit for {col}")
    u = u.strip().lower()
    if kind == "distance":
        if u in ("mi", "mile", "miles"):
            return 1.0
        if u in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
            return 1.0 / 1.609344
        raise _Abstain(f"unrecognized unit for {col}: {u}")
    if u in ("usd", "dollar", "dollars", "us dollars", "$"):
        return 1.0
    if u in ("cents", "cent", "usd cents"):
        return 0.01
    raise _Abstain(f"unrecognized unit for {col}: {u}")


def _run(data_path, manifest_path):
    # 1. manifest
    if not os.path.isfile(manifest_path):
        raise _Abstain("manifest missing")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        raise _Abstain("manifest is not valid JSON")
    if not isinstance(manifest, dict):
        raise _Abstain("manifest is not a JSON object")
    missing = [k for k in _REQUIRED_MANIFEST if k not in manifest]
    if missing:
        raise _Abstain(f"manifest missing fields: {missing}")
    rp = manifest["reporting_period"]
    if not isinstance(rp, dict) or any(k not in rp for k in ("start", "end", "timezone")):
        raise _Abstain("manifest reporting_period incomplete")
    cols = manifest["columns"]
    if not isinstance(cols, list) or not cols or not all(
            isinstance(c, dict) and isinstance(c.get("name"), str) for c in cols):
        raise _Abstain("manifest columns invalid")
    changes = manifest["schema_changes"]
    if changes is None:
        changes = []
    if not isinstance(changes, list):
        raise _Abstain("manifest schema_changes invalid")
    rc = manifest["row_count"]
    if isinstance(rc, bool) or not isinstance(rc, int):
        raise _Abstain("manifest row_count invalid")
    sha = manifest["file_sha256"]
    if not isinstance(sha, str):
        raise _Abstain("manifest file_sha256 invalid")

    # 2. checksum
    if not os.path.isfile(data_path):
        raise _Abstain("data file missing")
    if _sha256(data_path) != sha.strip().lower():
        raise _Abstain("file checksum mismatch")

    # 3. row count
    try:
        pf = pq.ParquetFile(data_path)
        nrows = pf.metadata.num_rows
        table = pf.read()
    except Exception:
        raise _Abstain("data file is not readable Parquet")
    if nrows != rc or table.num_rows != rc:
        raise _Abstain(f"row count mismatch: file {table.num_rows}, manifest {rc}")

    # 4. columns
    renames = {}
    for ch in changes:
        if not isinstance(ch, dict) or "type" not in ch:
            raise _Abstain("invalid schema_changes entry")
        if ch["type"] == "rename":
            fr, to = ch.get("from"), ch.get("to")
            if not isinstance(fr, str) or not isinstance(to, str):
                raise _Abstain("invalid rename entry")
            if to in renames and renames[to] != fr:
                raise _Abstain("conflicting renames")
            renames[to] = fr
        else:
            raise _Abstain(f"unsupported schema change: {ch['type']}")
    if len(set(renames.values())) != len(renames):
        raise _Abstain("conflicting renames")

    delivered = list(table.column_names)
    if len(set(delivered)) != len(delivered):
        raise _Abstain("duplicate delivered column names")
    for to, fr in renames.items():
        if to not in delivered:
            raise _Abstain(f"renamed column {to} not delivered")
        if fr in delivered:
            raise _Abstain(f"column {fr} delivered alongside its rename {to}")
    norm_delivered = [renames.get(c, c) for c in delivered]
    if len(set(norm_delivered)) != len(norm_delivered):
        raise _Abstain("column name collision after renames")
    man_names = [renames.get(c["name"], c["name"]) for c in cols]
    if len(set(man_names)) != len(man_names):
        raise _Abstain("duplicate manifest column names")
    if set(man_names) != set(norm_delivered):
        raise _Abstain("delivered columns disagree with manifest")
    absent = [c for c in _CONTRACT if c not in norm_delivered]
    if absent:
        raise _Abstain(f"contract columns missing: {absent}")

    df = table.to_pandas()
    df.columns = norm_delivered

    # normalize types
    for c in _TS_COLS:
        df[c] = _parse_ts(df[c], c)
    for c in _INT_COLS:
        df[c] = _parse_num(df[c], c, integer=True)
    for c in _FLOAT_COLS:
        df[c] = _parse_num(df[c], c, integer=False)

    # units
    units = manifest["units"]
    df["trip_distance"] = df["trip_distance"] * _unit_factor(units, "trip_distance", "distance")
    for c in ("fare_amount", "tip_amount", "total_amount"):
        df[c] = df[c] * _unit_factor(units, c, "money")

    # required non-null
    for c in _REQUIRED_NONNULL:
        if df[c].isna().any():
            raise _Abstain(f"required column {c} has nulls")

    # 5. reporting period
    tz = rp["timezone"]
    if not isinstance(tz, str):
        raise _Abstain("invalid reporting_period timezone")
    try:
        start = _to_local_naive_scalar(rp["start"], tz)
        end = _to_local_naive_scalar(rp["end"], tz)
    except _Abstain:
        raise
    except Exception:
        raise _Abstain("invalid reporting_period")
    if not start < end:
        raise _Abstain("empty reporting_period")
    pu = df["tpep_pickup_datetime"]
    if ((pu < start) | (pu >= end)).any():
        raise _Abstain("pickup timestamps outside reporting_period")

    # 6. contract rules
    if not df["payment_type"].isin([0, 1, 2, 3, 4, 5, 6]).all():
        raise _Abstain("undocumented payment_type codes")
    td = df["trip_distance"]
    if not ((td > 0) & (td <= 100)).all():
        raise _Abstain("trip_distance out of valid range")
    fa = df["fare_amount"]
    if not ((fa >= 0) & (fa <= 1000)).all():
        raise _Abstain("fare_amount out of valid range")

    # replays and conflicts
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    if df["record_id"].duplicated().any():
        raise _Abstain("conflicting records share a record_id")

    n = len(df)
    if n == 0:
        raise _Abstain("no trips")
    total_fare = float(df["fare_amount"].sum())
    return {
        "status": "ok",
        "metrics": {
            "trip_count": int(n),
            "total_fare": total_fare,
            "mean_fare": float(df["fare_amount"].mean()),
            "mean_trip_distance": float(df["trip_distance"].mean()),
            "card_share": float((df["payment_type"] == 1).sum()) / n,
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        result = _run(data_path, manifest_path)
        m = result["metrics"]
        for k, v in m.items():
            if isinstance(v, float) and not math.isfinite(v):
                return _abstain(f"non-finite metric {k}")
        return result
    except _Abstain as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain(f"unexpected error: {type(e).__name__}")
