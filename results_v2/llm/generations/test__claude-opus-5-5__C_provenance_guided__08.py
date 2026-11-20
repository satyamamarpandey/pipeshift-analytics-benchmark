import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NY_TZ = "America/New_York"

CONTRACT_COLUMNS = [
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]
REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
INT_COLS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"]
FLOAT_COLS = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
TS_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
MONEY_COLS = ["fare_amount", "tip_amount", "total_amount"]
VALID_PAYMENT = {0, 1, 2, 3, 4, 5, 6}
MANIFEST_FIELDS = [
    "dataset", "snapshot_id", "snapshot_version", "reporting_period", "file",
    "file_sha256", "row_count", "columns", "units", "schema_changes", "description",
]


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _load_manifest(manifest_path):
    if not os.path.isfile(manifest_path):
        raise _Abstain("manifest missing")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        raise _Abstain("manifest is not valid JSON")
    if not isinstance(m, dict):
        raise _Abstain("manifest is not a JSON object")
    for k in MANIFEST_FIELDS:
        if k not in m:
            raise _Abstain(f"manifest missing field: {k}")
    rp = m["reporting_period"]
    if not isinstance(rp, dict) or not all(k in rp for k in ("start", "end", "timezone")):
        raise _Abstain("manifest reporting_period invalid")
    if not isinstance(m["file_sha256"], str) or len(m["file_sha256"].strip()) != 64:
        raise _Abstain("manifest file_sha256 invalid")
    rc = m["row_count"]
    if isinstance(rc, bool) or not isinstance(rc, int) or rc < 0:
        raise _Abstain("manifest row_count invalid")
    cols = m["columns"]
    if not isinstance(cols, list) or not cols:
        raise _Abstain("manifest columns invalid")
    for c in cols:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or "type" not in c:
            raise _Abstain("manifest columns invalid")
    sc = m["schema_changes"]
    if sc is None:
        sc = []
    if not isinstance(sc, list):
        raise _Abstain("manifest schema_changes invalid")
    for ch in sc:
        if not isinstance(ch, dict):
            raise _Abstain("manifest schema_changes invalid")
        if ch.get("type") == "rename":
            if not isinstance(ch.get("from"), str) or not isinstance(ch.get("to"), str):
                raise _Abstain("manifest rename entry invalid")
    m["schema_changes"] = sc
    return m


def _apply_renames(name, renames):
    for frm, to in renames:
        if name == frm:
            name = to
    return name


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_naive_local(ts, tz=NY_TZ):
    if ts.tzinfo is not None:
        return ts.tz_convert(tz).tz_localize(None)
    return ts


def _parse_ts(series, name):
    s = series
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(NY_TZ).dt.tz_localize(None).astype("datetime64[ns]")
    if pd.api.types.is_datetime64_dtype(s.dtype):
        return s.astype("datetime64[ns]")
    if pd.api.types.is_numeric_dtype(s.dtype) and not s.isna().all():
        raise _Abstain(f"{name} has non-datetime numeric values")
    cache = {}
    out = []
    for v in s.tolist():
        if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT:
            out.append(pd.NaT)
            continue
        key = (type(v).__name__, str(v))
        if key in cache:
            out.append(cache[key])
            continue
        try:
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8")
            if isinstance(v, str):
                txt = v.strip()
                if not txt:
                    raise ValueError
                ts = pd.Timestamp(txt)
            elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                raise ValueError
            else:
                ts = pd.Timestamp(v)
            if ts is pd.NaT:
                raise ValueError
            ts = _to_naive_local(ts)
        except Exception:
            raise _Abstain(f"unparseable timestamp in {name}")
        cache[key] = ts
        out.append(ts)
    return pd.Series(pd.to_datetime(pd.Series(out, dtype=object), errors="raise"),
                     index=series.index).astype("datetime64[ns]")


def _parse_num(series, name):
    s = series
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    if pd.api.types.is_bool_dtype(s.dtype):
        raise _Abstain(f"{name} has boolean type")
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.astype("float64")
    vals = []
    for v in s.tolist():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            vals.append(np.nan)
            continue
        if isinstance(v, bool):
            raise _Abstain(f"{name} has boolean values")
        try:
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8")
            if isinstance(v, str):
                t = v.strip()
                if t == "" or t.lower() in ("nan", "null", "none", "na", "nat"):
                    vals.append(np.nan)
                    continue
                f = float(t)
            else:
                f = float(v)
        except Exception:
            raise _Abstain(f"non-numeric value in {name}")
        if math.isinf(f):
            raise _Abstain(f"infinite value in {name}")
        vals.append(f)
    return pd.Series(vals, index=series.index, dtype="float64")


def _unit_factors(units):
    dist_f, money_f = 1.0, 1.0
    if units is None:
        return dist_f, money_f
    if not isinstance(units, dict):
        if isinstance(units, str):
            return dist_f, money_f
        raise _Abstain("manifest units invalid")
    for k, v in units.items():
        key = str(k).strip().lower()
        if isinstance(v, dict):
            v = v.get("unit", v.get("units"))
        if v is None:
            continue
        u = str(v).strip().lower()
        if key in ("trip_distance", "distance"):
            if "km" in u or "kilomet" in u:
                dist_f = 1.0 / 1.609344
            elif "mile" in u or u in ("mi", "mi."):
                dist_f = 1.0
            else:
                raise _Abstain(f"unsupported distance unit: {v}")
        elif key in ("fare_amount", "currency", "money", "amount", "fare", "fares"):
            if "cent" in u:
                money_f = 0.01
            elif "usd" in u or "dollar" in u or u == "$":
                money_f = 1.0
            else:
                raise _Abstain(f"unsupported currency unit: {v}")
    return dist_f, money_f


def _run(data_path, manifest_path):
    m = _load_manifest(manifest_path)

    if not os.path.isfile(data_path):
        raise _Abstain("data file missing")
    if _sha256(data_path).lower() != m["file_sha256"].strip().lower():
        raise _Abstain("checksum mismatch")

    try:
        pf = pq.ParquetFile(data_path)
        meta_rows = pf.metadata.num_rows
        table = pf.read()
    except Exception:
        raise _Abstain("data file unreadable as parquet")
    if meta_rows != m["row_count"] or table.num_rows != m["row_count"]:
        raise _Abstain("row count mismatch")

    delivered = list(table.column_names)
    if len(set(delivered)) != len(delivered):
        raise _Abstain("duplicate column names in data")

    renames = [(c["from"], c["to"]) for c in m["schema_changes"]
               if isinstance(c, dict) and c.get("type") == "rename"]
    manifest_names = [c["name"] for c in m["columns"]]
    expected = {_apply_renames(n, renames) for n in manifest_names}
    if expected != set(delivered):
        raise _Abstain("delivered columns disagree with manifest")

    df = table.to_pandas()

    rename_back = {}
    for c in CONTRACT_COLUMNS:
        f = _apply_renames(c, renames)
        if f != c and f in df.columns and c not in df.columns:
            rename_back[f] = c
    df = df.rename(columns=rename_back)

    for c in CONTRACT_COLUMNS:
        if c not in df.columns:
            raise _Abstain(f"missing contract column: {c}")

    dist_f, money_f = _unit_factors(m.get("units"))

    for c in TS_COLS:
        df[c] = _parse_ts(df[c], c)
    for c in INT_COLS + FLOAT_COLS:
        df[c] = _parse_num(df[c], c)

    for c in REQUIRED:
        if df[c].isna().any():
            raise _Abstain(f"null values in required column {c}")

    for c in INT_COLS:
        v = df[c].dropna()
        if not np.all(np.isfinite(v)) or not np.all(v == np.floor(v)):
            raise _Abstain(f"non-integer values in {c}")

    df["trip_distance"] = df["trip_distance"] * dist_f
    for c in MONEY_COLS:
        df[c] = df[c] * money_f

    if not df["payment_type"].isin(list(VALID_PAYMENT)).all():
        raise _Abstain("undocumented payment_type code")
    td = df["trip_distance"]
    if not ((td > 0) & (td <= 100)).all():
        raise _Abstain("trip_distance out of valid range")
    fa = df["fare_amount"]
    if not ((fa >= 0) & (fa <= 1000)).all():
        raise _Abstain("fare_amount out of valid range")

    rp = m["reporting_period"]
    tzname = str(rp["timezone"]).strip()
    try:
        start = pd.Timestamp(str(rp["start"]))
        end = pd.Timestamp(str(rp["end"]))
        if start is pd.NaT or end is pd.NaT:
            raise ValueError
        start = _to_naive_local(start, tzname)
        end = _to_naive_local(end, tzname)
    except Exception:
        raise _Abstain("reporting_period unparseable")
    if not start < end:
        raise _Abstain("reporting_period empty or inverted")
    pu = df["tpep_pickup_datetime"]
    if tzname in (NY_TZ, "US/Eastern"):
        inside = (pu >= start) & (pu < end)
    else:
        try:
            pu_u = pu.dt.tz_localize(NY_TZ, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC")
            s_u = start.tz_localize(tzname).tz_convert("UTC")
            e_u = end.tz_localize(tzname).tz_convert("UTC")
        except Exception:
            raise _Abstain("reporting_period timezone invalid")
        inside = (pu_u >= s_u) & (pu_u < e_u)
    if not inside.all():
        raise _Abstain("pickup timestamps outside reporting_period")

    df = df.drop_duplicates(keep="first")
    if df["record_id"].duplicated().any():
        raise _Abstain("conflicting records share a record_id")

    n = int(len(df))
    if n == 0:
        raise _Abstain("no trips")
    total_fare = float(df["fare_amount"].sum())
    return {
        "status": "ok",
        "metrics": {
            "trip_count": n,
            "total_fare": total_fare,
            "mean_fare": total_fare / n,
            "mean_trip_distance": float(df["trip_distance"].sum()) / n,
            "card_share": float((df["payment_type"] == 1).sum()) / n,
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _run(data_path, manifest_path)
    except _Abstain as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain(f"unexpected error: {type(e).__name__}")
