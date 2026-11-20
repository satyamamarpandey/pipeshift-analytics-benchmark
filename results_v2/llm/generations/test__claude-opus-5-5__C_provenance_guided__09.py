import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NY_TZ = "America/New_York"
CONTRACT_COLS = [
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]
REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
INT_COLS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"]
FLOAT_COLS = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
DT_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
MONEY_COLS = ["fare_amount", "tip_amount", "total_amount"]
VALID_PAYMENT = {0, 1, 2, 3, 4, 5, 6}
MANIFEST_FIELDS = [
    "dataset", "snapshot_id", "snapshot_version", "reporting_period", "file",
    "file_sha256", "row_count", "columns", "units", "schema_changes", "description",
]

DIST_UNITS = {
    "mile": 1.0, "miles": 1.0, "mi": 1.0,
    "km": 1.0 / 1.609344, "kilometer": 1.0 / 1.609344, "kilometers": 1.0 / 1.609344,
    "kilometre": 1.0 / 1.609344, "kilometres": 1.0 / 1.609344,
    "m": 1.0 / 1609.344, "meter": 1.0 / 1609.344, "meters": 1.0 / 1609.344,
    "metre": 1.0 / 1609.344, "metres": 1.0 / 1609.344,
}
MONEY_UNITS = {
    "usd": 1.0, "$": 1.0, "dollar": 1.0, "dollars": 1.0, "us dollars": 1.0,
    "cents": 0.01, "cent": 0.01, "usd cents": 0.01, "usd_cents": 0.01,
}


class Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _to_ny_naive_ts(v):
    if isinstance(v, (bool, np.bool_)):
        raise Abstain("invalid datetime value")
    if isinstance(v, (int, float, np.integer, np.floating)):
        fv = float(v)
        if not math.isfinite(fv):
            raise Abstain("invalid datetime value")
        a = abs(fv)
        unit = "ns" if a > 1e17 else "us" if a > 1e14 else "ms" if a > 1e11 else "s"
        ts = pd.Timestamp(int(v), unit=unit)
    else:
        try:
            ts = pd.Timestamp(v.strip() if isinstance(v, str) else v)
        except Exception:
            raise Abstain("unparseable datetime value")
    if ts is pd.NaT or pd.isna(ts):
        raise Abstain("unparseable datetime value")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    return ts


def _norm_datetime(s):
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(NY_TZ).dt.tz_localize(None).astype("datetime64[ns]")
    if pd.api.types.is_datetime64_dtype(s.dtype):
        return s.astype("datetime64[ns]")
    out = []
    for v in s.tolist():
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            out.append(pd.NaT)
        elif isinstance(v, str) and v.strip() == "":
            out.append(pd.NaT)
        else:
            out.append(_to_ny_naive_ts(v))
    return pd.Series(pd.to_datetime(pd.Series(out, dtype="object"), errors="raise"),
                     index=s.index).astype("datetime64[ns]")


def _norm_numeric(s, name):
    if pd.api.types.is_bool_dtype(s.dtype):
        raise Abstain(f"invalid type for {name}")
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        vals = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        vals = vals.map(lambda v: None if isinstance(v, str) and v == "" else v)
        if vals.map(lambda v: isinstance(v, (bool, np.bool_))).any():
            raise Abstain(f"invalid type for {name}")
        num = pd.to_numeric(vals, errors="coerce")
        if (num.isna() & vals.notna()).any():
            raise Abstain(f"non-numeric values in {name}")
    else:
        num = pd.to_numeric(s, errors="coerce")
        if (num.isna() & s.notna()).any():
            raise Abstain(f"non-numeric values in {name}")
    num = num.astype("float64")
    if np.isinf(num.to_numpy()).any():
        raise Abstain(f"infinite values in {name}")
    return num


def _parse_period_bound(v):
    if not isinstance(v, str) or not v.strip():
        raise Abstain("invalid reporting_period bound")
    return _to_ny_naive_ts(v)


def _run(data_path, manifest_path):
    # 1. Manifest
    if not os.path.isfile(manifest_path):
        raise Abstain("manifest missing")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        raise Abstain("manifest is not valid JSON")
    if not isinstance(manifest, dict):
        raise Abstain("manifest is not a JSON object")
    missing = [k for k in MANIFEST_FIELDS if k not in manifest]
    if missing:
        raise Abstain(f"manifest missing fields: {missing}")
    rp = manifest["reporting_period"]
    if not isinstance(rp, dict) or not all(k in rp for k in ("start", "end", "timezone")):
        raise Abstain("invalid reporting_period")
    if rp["timezone"] != NY_TZ:
        raise Abstain("unexpected reporting_period timezone")
    start = _parse_period_bound(rp["start"])
    end = _parse_period_bound(rp["end"])
    if not start < end:
        raise Abstain("empty reporting_period")
    sha = manifest["file_sha256"]
    if not isinstance(sha, str) or len(sha.strip()) != 64:
        raise Abstain("invalid file_sha256")
    rc = manifest["row_count"]
    if isinstance(rc, bool) or not isinstance(rc, int) or rc < 0:
        raise Abstain("invalid row_count")
    cols_m = manifest["columns"]
    if not isinstance(cols_m, list) or not cols_m:
        raise Abstain("invalid manifest columns")
    m_names = []
    for c in cols_m:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or "type" not in c:
            raise Abstain("invalid manifest column entry")
        m_names.append(c["name"])
    if len(set(m_names)) != len(m_names):
        raise Abstain("duplicate manifest columns")
    changes = manifest["schema_changes"]
    if changes is None:
        changes = []
    if not isinstance(changes, list):
        raise Abstain("invalid schema_changes")
    renames = []
    for ch in changes:
        if not isinstance(ch, dict):
            raise Abstain("invalid schema_changes entry")
        if ch.get("type") != "rename":
            raise Abstain(f"unsupported schema change: {ch.get('type')}")
        fr, to = ch.get("from"), ch.get("to")
        if not isinstance(fr, str) or not isinstance(to, str) or not fr or not to:
            raise Abstain("invalid rename entry")
        renames.append((fr, to))

    # 2. Checksum
    if not os.path.isfile(data_path):
        raise Abstain("data file missing")
    h = hashlib.sha256()
    with open(data_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest().lower() != sha.strip().lower():
        raise Abstain("checksum mismatch")

    # 3. Row count
    try:
        table = pq.read_table(data_path)
    except Exception:
        raise Abstain("cannot read parquet file")
    if table.num_rows != rc:
        raise Abstain("row count mismatch")

    # 4. Columns
    names = list(table.column_names)
    if len(set(names)) != len(names):
        raise Abstain("duplicate delivered columns")

    def to_original(name_list):
        cur = list(name_list)
        for fr, to in reversed(renames):
            if to in cur:
                if fr in cur:
                    raise Abstain(f"both {fr} and {to} present")
                cur[cur.index(to)] = fr
        return cur

    orig_names = to_original(names)
    orig_m = to_original(m_names)
    if len(set(orig_names)) != len(orig_names) or len(set(orig_m)) != len(orig_m):
        raise Abstain("column name collision after renames")
    if set(orig_names) != set(orig_m):
        raise Abstain("delivered columns do not match manifest")
    miss_c = [c for c in CONTRACT_COLS if c not in orig_names]
    if miss_c:
        raise Abstain(f"missing contract columns: {miss_c}")

    try:
        df = table.to_pandas(timestamp_as_object=False)
    except Exception:
        try:
            df = table.to_pandas(timestamp_as_object=True)
        except Exception:
            raise Abstain("cannot convert data")
    df.columns = orig_names

    # Normalize types
    for c in DT_COLS:
        df[c] = _norm_datetime(df[c])
    for c in INT_COLS + FLOAT_COLS:
        df[c] = _norm_numeric(df[c], c)
    for c in INT_COLS:
        v = df[c].dropna()
        if (v != np.floor(v)).any():
            raise Abstain(f"non-integer values in {c}")

    # Units
    units = manifest["units"]
    if isinstance(units, dict):
        for k, u in units.items():
            key = to_original([k])[0]
            if key == "trip_distance" or key in MONEY_COLS:
                if not isinstance(u, str):
                    raise Abstain(f"invalid unit for {key}")
                ul = u.strip().lower()
                table_u = DIST_UNITS if key == "trip_distance" else MONEY_UNITS
                if ul not in table_u:
                    raise Abstain(f"unknown unit {u} for {key}")
                df[key] = df[key] * table_u[ul]
    elif units is not None and not isinstance(units, (str, list)):
        raise Abstain("invalid units")

    # 6a. Required non-null
    for c in REQUIRED:
        if df[c].isna().any():
            raise Abstain(f"null values in required column {c}")

    # 5. Reporting period
    pu = df["tpep_pickup_datetime"]
    if ((pu < start) | (pu >= end)).any():
        raise Abstain("pickup outside reporting_period")

    # 6b. Codes and ranges
    if not df["payment_type"].isin(VALID_PAYMENT).all():
        raise Abstain("undocumented payment_type code")
    td = df["trip_distance"]
    if not ((td > 0) & (td <= 100)).all():
        raise Abstain("trip_distance out of range")
    fa = df["fare_amount"]
    if not ((fa >= 0) & (fa <= 1000)).all():
        raise Abstain("fare_amount out of range")

    # 6c. Replays and conflicts
    df = df.drop_duplicates(keep="first")
    if df["record_id"].duplicated().any():
        raise Abstain("conflicting records with same record_id")

    n = len(df)
    if n == 0:
        raise Abstain("no trips")
    fare = df["fare_amount"].to_numpy(dtype="float64")
    dist = df["trip_distance"].to_numpy(dtype="float64")
    total_fare = float(fare.sum())
    return {
        "status": "ok",
        "metrics": {
            "trip_count": int(n),
            "total_fare": total_fare,
            "mean_fare": total_fare / n,
            "mean_trip_distance": float(dist.sum()) / n,
            "card_share": float((df["payment_type"] == 1).sum()) / n,
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _run(data_path, manifest_path)
    except Abstain as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain(f"unexpected error: {type(e).__name__}")
