import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CONTRACT_COLUMNS = [
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]
REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
NUMERIC_COLS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "trip_distance",
                "payment_type", "fare_amount", "tip_amount", "total_amount"]
INTEGER_COLS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"]
DATETIME_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
MANIFEST_FIELDS = ["dataset", "snapshot_id", "snapshot_version", "reporting_period", "file",
                   "file_sha256", "row_count", "columns", "units", "schema_changes",
                   "description"]
VALID_PAYMENT = {0, 1, 2, 3, 4, 5, 6}
LOCAL_TZ = "America/New_York"


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _parse_bound(value, period_tz):
    if not isinstance(value, str) or not value.strip():
        raise _Abstain("invalid reporting_period bound")
    try:
        ts = pd.Timestamp(value.strip())
    except Exception:
        raise _Abstain("unparseable reporting_period bound")
    if ts is pd.NaT or pd.isna(ts):
        raise _Abstain("unparseable reporting_period bound")
    if ts.tzinfo is None:
        if period_tz and period_tz != LOCAL_TZ:
            try:
                ts = ts.tz_localize(period_tz).tz_convert(LOCAL_TZ).tz_localize(None)
            except Exception:
                raise _Abstain("invalid reporting_period timezone")
    else:
        ts = ts.tz_convert(LOCAL_TZ).tz_localize(None)
    return ts


def _to_local_naive(s, strict):
    dtype = s.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    if pd.api.types.is_datetime64_dtype(dtype):
        return pd.to_datetime(s)
    if pd.api.types.is_bool_dtype(dtype):
        raise _Abstain("boolean timestamp column")
    if pd.api.types.is_numeric_dtype(dtype):
        nn = s.dropna()
        if len(nn) == 0:
            return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        mx = float(np.nanmax(np.abs(nn.astype("float64"))))
        if mx < 1e11:
            unit = "s"
        elif mx < 1e14:
            unit = "ms"
        elif mx < 1e17:
            unit = "us"
        else:
            unit = "ns"
        out = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
        return out.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

    cache = {}

    def conv(v):
        if v is None:
            return pd.NaT
        try:
            if pd.isna(v):
                return pd.NaT
        except (TypeError, ValueError):
            pass
        key = (type(v).__name__, str(v))
        if key in cache:
            return cache[key]
        try:
            if isinstance(v, str):
                txt = v.strip()
                if txt == "":
                    r = pd.NaT
                else:
                    r = pd.Timestamp(txt)
            else:
                r = pd.Timestamp(v)
            if r is not pd.NaT and not pd.isna(r) and r.tzinfo is not None:
                r = r.tz_convert(LOCAL_TZ).tz_localize(None)
        except Exception:
            if strict:
                raise _Abstain("unparseable timestamp value")
            r = pd.NaT
        cache[key] = r
        return r

    vals = [conv(v) for v in s.tolist()]
    return pd.Series(pd.to_datetime(vals), index=s.index).astype("datetime64[ns]")


def _to_numeric(s, name):
    if pd.api.types.is_bool_dtype(s.dtype):
        raise _Abstain(f"boolean values in {name}")
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.astype("float64")
    orig_null = s.isna()
    work = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    work = work.map(lambda v: None if isinstance(v, str) and v == "" else v)
    if work.map(lambda v: isinstance(v, bool)).any():
        raise _Abstain(f"boolean values in {name}")
    out = pd.to_numeric(work, errors="coerce").astype("float64")
    bad = out.isna() & ~orig_null & work.notna()
    if bad.any():
        raise _Abstain(f"non-numeric values in {name}")
    return out


def _unit_factor(units, names, kind):
    if not isinstance(units, dict):
        return 1.0
    val = None
    for n in names:
        if n in units:
            val = units[n]
            break
    if val is None:
        return 1.0
    if isinstance(val, dict):
        val = val.get("unit", val.get("units"))
    if not isinstance(val, str):
        raise _Abstain(f"unrecognized unit specification for {names[0]}")
    u = val.strip().lower()
    if kind == "distance":
        if u in {"mi", "mile", "miles"}:
            return 1.0
        if u in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
            return 1.0 / 1.609344
        if u in {"m", "meter", "meters", "metre", "metres"}:
            return 1.0 / 1609.344
        raise _Abstain(f"unknown distance unit '{val}'")
    else:
        if u in {"usd", "$", "dollar", "dollars", "us dollars", "us dollar", "usd dollars"}:
            return 1.0
        if u in {"cent", "cents", "usd cents", "us cents", "usd_cents"}:
            return 0.01
        raise _Abstain(f"unknown currency unit '{val}'")


def _run(data_path, manifest_path):
    # 1. Manifest
    if not os.path.isfile(manifest_path):
        raise _Abstain("manifest missing")
    try:
        with open(manifest_path, "rb") as fh:
            manifest = json.loads(fh.read().decode("utf-8"))
    except Exception:
        raise _Abstain("manifest is not valid JSON")
    if not isinstance(manifest, dict):
        raise _Abstain("manifest is not a JSON object")
    missing = [f for f in MANIFEST_FIELDS if f not in manifest]
    if missing:
        raise _Abstain(f"manifest missing fields: {missing}")
    rp = manifest["reporting_period"]
    if not isinstance(rp, dict) or not all(k in rp for k in ("start", "end", "timezone")):
        raise _Abstain("invalid reporting_period")
    period_tz = rp["timezone"]
    if not isinstance(period_tz, str) or not period_tz.strip():
        raise _Abstain("invalid reporting_period timezone")
    period_tz = period_tz.strip()
    try:
        pd.Timestamp("2000-01-01").tz_localize(period_tz)
    except Exception:
        raise _Abstain("invalid reporting_period timezone")
    start = _parse_bound(rp["start"], period_tz)
    end = _parse_bound(rp["end"], period_tz)
    if not start < end:
        raise _Abstain("empty reporting_period")
    sha = manifest["file_sha256"]
    if not isinstance(sha, str) or len(sha.strip()) != 64:
        raise _Abstain("invalid file_sha256")
    row_count = manifest["row_count"]
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise _Abstain("invalid row_count")
    cols = manifest["columns"]
    if not isinstance(cols, list) or not cols:
        raise _Abstain("invalid columns list")
    man_names = []
    for c in cols:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or "type" not in c:
            raise _Abstain("invalid column entry in manifest")
        man_names.append(c["name"])
    if len(set(man_names)) != len(man_names):
        raise _Abstain("duplicate column names in manifest")
    changes = manifest["schema_changes"]
    if changes is None:
        changes = []
    if not isinstance(changes, list):
        raise _Abstain("invalid schema_changes")
    renames = []
    for ch in changes:
        if not isinstance(ch, dict):
            raise _Abstain("invalid schema_changes entry")
        if ch.get("type") == "rename":
            f, t = ch.get("from"), ch.get("to")
            if not isinstance(f, str) or not isinstance(t, str) or not f or not t:
                raise _Abstain("invalid rename entry")
            renames.append((f, t))

    # 2. Checksum
    if not os.path.isfile(data_path):
        raise _Abstain("data file missing")
    h = hashlib.sha256()
    with open(data_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest().lower() != sha.strip().lower():
        raise _Abstain("file checksum mismatch")

    # 3. Row count
    try:
        table = pq.read_table(data_path)
    except Exception:
        raise _Abstain("data file is not readable Parquet")
    if table.num_rows != row_count:
        raise _Abstain("row count mismatch")

    # 4. Columns
    delivered = list(table.column_names)
    if len(set(delivered)) != len(delivered):
        raise _Abstain("duplicate delivered column names")
    mapped = list(man_names)
    for f, t in renames:
        mapped = [t if n == f else n for n in mapped]
    if set(delivered) != set(mapped) and set(delivered) != set(man_names):
        raise _Abstain("delivered columns disagree with manifest")
    # normalize delivered names back to contract names
    name_map = {n: n for n in delivered}
    for f, t in reversed(renames):
        for k, v in list(name_map.items()):
            if v == t:
                name_map[k] = f
    targets = list(name_map.values())
    if len(set(targets)) != len(targets):
        raise _Abstain("column name collision after applying renames")
    missing_c = [c for c in CONTRACT_COLUMNS if c not in targets]
    if missing_c:
        raise _Abstain(f"contract columns missing: {missing_c}")

    try:
        df = table.to_pandas()
    except Exception:
        try:
            df = table.to_pandas(timestamp_as_object=True)
        except Exception:
            raise _Abstain("cannot convert data")
    df = df.rename(columns=name_map)
    df = df[CONTRACT_COLUMNS].copy()
    if len(df) == 0:
        raise _Abstain("no rows")

    # type normalization
    units = manifest["units"]
    for c in NUMERIC_COLS:
        df[c] = _to_numeric(df[c], c)
    df["tpep_pickup_datetime"] = _to_local_naive(df["tpep_pickup_datetime"], strict=True)
    df["tpep_dropoff_datetime"] = _to_local_naive(df["tpep_dropoff_datetime"], strict=False)

    inv = {f: t for f, t in renames}
    dist_f = _unit_factor(units, ["trip_distance", inv.get("trip_distance", "trip_distance")],
                          "distance")
    for c in ["fare_amount", "tip_amount", "total_amount"]:
        fct = _unit_factor(units, [c, inv.get(c, c)], "money")
        if fct != 1.0:
            df[c] = df[c] * fct
    if dist_f != 1.0:
        df["trip_distance"] = df["trip_distance"] * dist_f

    # required non-null
    for c in REQUIRED:
        if df[c].isna().any():
            raise _Abstain(f"null values in required column {c}")

    # 5. period
    pu = df["tpep_pickup_datetime"]
    if ((pu < start) | (pu >= end)).any():
        raise _Abstain("pickup timestamps outside reporting_period")

    # 6. contract rules
    for c in INTEGER_COLS:
        v = df[c].dropna()
        if (~np.isfinite(v)).any() or (v != np.floor(v)).any():
            raise _Abstain(f"non-integer values in {c}")
    if not df["payment_type"].isin(list(VALID_PAYMENT)).all():
        raise _Abstain("undocumented payment_type codes")
    td = df["trip_distance"]
    if (~np.isfinite(td)).any() or ((td <= 0) | (td > 100)).any():
        raise _Abstain("trip_distance outside valid range")
    fa = df["fare_amount"]
    if (~np.isfinite(fa)).any() or ((fa < 0) | (fa > 1000)).any():
        raise _Abstain("fare_amount outside valid range")

    # replays and conflicts
    df = df.drop_duplicates(keep="first")
    if df["record_id"].duplicated().any():
        raise _Abstain("conflicting records share record_id")

    n = int(len(df))
    if n == 0:
        raise _Abstain("no trips")
    total_fare = float(df["fare_amount"].sum())
    mean_fare = float(df["fare_amount"].mean())
    mean_dist = float(df["trip_distance"].mean())
    card = int((df["payment_type"] == 1).sum())
    vals = [total_fare, mean_fare, mean_dist]
    if any(not math.isfinite(v) for v in vals):
        raise _Abstain("non-finite metric")
    return {
        "status": "ok",
        "metrics": {
            "trip_count": n,
            "total_fare": total_fare,
            "mean_fare": mean_fare,
            "mean_trip_distance": mean_dist,
            "card_share": card / n,
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _run(data_path, manifest_path)
    except _Abstain as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain(f"unexpected error during validation: {type(e).__name__}")
