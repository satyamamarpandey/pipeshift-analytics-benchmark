import hashlib
import json
import math
import os
from decimal import Decimal

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


class Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _norm_unit(u):
    return str(u).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _distance_factor(unit):
    u = _norm_unit(unit)
    if u in ("mi", "mile", "miles", "statutemile", "statutemiles"):
        return 1.0
    if u in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
        return 1.0 / 1.609344
    if u in ("m", "meter", "meters", "metre", "metres"):
        return 1.0 / 1609.344
    raise Abstain(f"unsupported distance unit: {unit}")


def _money_factor(unit):
    u = _norm_unit(unit)
    if u in ("usd", "dollar", "dollars", "us$", "$", "usdollar", "usdollars"):
        return 1.0
    if u in ("cents", "cent", "usdcents", "usdcent", "centsusd", "uscents"):
        return 0.01
    raise Abstain(f"unsupported currency/unit: {unit}")


def _parse_period_bound(v):
    if v is None:
        raise Abstain("reporting_period bound missing")
    try:
        ts = pd.Timestamp(v)
    except Exception:
        raise Abstain("unparseable reporting_period bound")
    if ts is pd.NaT:
        raise Abstain("unparseable reporting_period bound")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    return ts


def _get_period(manifest):
    rp = manifest.get("reporting_period")
    if rp is None:
        raise Abstain("manifest lacks reporting_period")
    if isinstance(rp, dict):
        start = None
        end = None
        for k in ("start", "from", "begin", "start_date", "start_inclusive"):
            if k in rp:
                start = rp[k]
                break
        for k in ("end", "to", "stop", "end_date", "end_exclusive"):
            if k in rp:
                end = rp[k]
                break
        if start is None or end is None:
            raise Abstain("reporting_period incomplete")
        s, e = _parse_period_bound(start), _parse_period_bound(end)
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        s, e = _parse_period_bound(rp[0]), _parse_period_bound(rp[1])
    elif isinstance(rp, str):
        try:
            p = pd.Period(rp, freq="M") if len(rp.strip()) <= 7 else None
        except Exception:
            p = None
        if p is None:
            raise Abstain("unrecognised reporting_period format")
        s = p.start_time
        e = (p + 1).start_time
    else:
        raise Abstain("unrecognised reporting_period format")
    if not s < e:
        raise Abstain("empty reporting_period")
    return s, e


def _find_sha(manifest):
    cands = [manifest]
    for k in ("data", "file", "files", "checksum", "checksums", "integrity"):
        v = manifest.get(k)
        if isinstance(v, dict):
            cands.append(v)
    for d in cands:
        for k in ("sha256", "sha256sum", "data_sha256", "file_sha256"):
            v = d.get(k)
            if isinstance(v, str) and len(v.strip()) == 64:
                return v.strip().lower()
    return None


def _find_row_count(manifest):
    cands = [manifest]
    for k in ("data", "file", "stats", "statistics"):
        v = manifest.get(k)
        if isinstance(v, dict):
            cands.append(v)
    for d in cands:
        for k in ("row_count", "num_rows", "record_count", "expected_rows",
                  "expected_row_count", "n_rows", "rows"):
            v = d.get(k)
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                return int(v)
    return None


def _to_float_series(s, col):
    if pd.api.types.is_bool_dtype(s.dtype):
        raise Abstain(f"column {col} has boolean type")
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.astype("float64")
    vals = s.astype(object).tolist()
    out = np.empty(len(vals), dtype="float64")
    bad = 0
    for i, v in enumerate(vals):
        if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA or v is pd.NaT:
            out[i] = np.nan
            continue
        if isinstance(v, bool):
            bad += 1
            out[i] = np.nan
            continue
        try:
            if isinstance(v, (int, float, np.integer, np.floating, Decimal)):
                out[i] = float(v)
            elif isinstance(v, (str, bytes)):
                t = v.decode() if isinstance(v, bytes) else v
                t = t.strip().replace(",", "").replace("$", "")
                if t == "" or t.lower() in ("nan", "null", "none", "na", "n/a"):
                    out[i] = np.nan
                else:
                    out[i] = float(t)
            else:
                out[i] = float(v)
        except Exception:
            bad += 1
            out[i] = np.nan
    if bad:
        raise Abstain(f"column {col} has {bad} non-numeric values")
    return pd.Series(out, index=s.index)


def _to_local_dt(s, col, src_tz):
    def fix_naive(ser):
        if src_tz and src_tz != NY_TZ:
            ser = ser.dt.tz_localize(src_tz, ambiguous="NaT", nonexistent="NaT")
            return ser.dt.tz_convert(NY_TZ).dt.tz_localize(None)
        return ser

    dt = s.dtype
    if isinstance(dt, pd.DatetimeTZDtype):
        return s.dt.tz_convert(NY_TZ).dt.tz_localize(None).astype("datetime64[ns]")
    if pd.api.types.is_datetime64_dtype(dt):
        return fix_naive(s.astype("datetime64[ns]"))
    if pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt):
        f = s.astype("float64")
        nn = f.dropna()
        if len(nn) == 0:
            return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        mx = nn.abs().max()
        unit = "s" if mx < 1e11 else "ms" if mx < 1e14 else "us" if mx < 1e17 else "ns"
        r = pd.to_datetime(f, unit=unit, utc=True, errors="coerce")
        return r.dt.tz_convert(NY_TZ).dt.tz_localize(None)

    vals = s.astype(object).tolist()
    out = []
    naive_mask = []
    bad = 0
    for v in vals:
        if v is None or v is pd.NaT or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
            out.append(pd.NaT)
            naive_mask.append(False)
            continue
        try:
            if isinstance(v, bytes):
                v = v.decode()
            if isinstance(v, str):
                t = v.strip()
                if t == "" or t.lower() in ("nan", "nat", "null", "none"):
                    out.append(pd.NaT)
                    naive_mask.append(False)
                    continue
                ts = pd.Timestamp(t)
            elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                x = float(v)
                ax = abs(x)
                unit = "s" if ax < 1e11 else "ms" if ax < 1e14 else "us" if ax < 1e17 else "ns"
                ts = pd.Timestamp(x, unit=unit, tz="UTC")
            else:
                ts = pd.Timestamp(v)
            if ts is pd.NaT:
                out.append(pd.NaT)
                naive_mask.append(False)
                continue
            if ts.tzinfo is not None:
                ts = ts.tz_convert(NY_TZ).tz_localize(None)
                naive_mask.append(False)
            else:
                naive_mask.append(True)
            out.append(ts)
        except Exception:
            bad += 1
            out.append(pd.NaT)
            naive_mask.append(False)
    if bad:
        raise Abstain(f"column {col} has {bad} unparseable timestamps")
    r = pd.Series(pd.to_datetime(out), index=s.index).astype("datetime64[ns]")
    if src_tz and src_tz != NY_TZ:
        m = pd.Series(naive_mask, index=s.index)
        if m.any():
            conv = fix_naive(r[m])
            r = r.copy()
            r[m] = conv
    return r


def _analyze(data_path, manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception:
        raise Abstain("cannot read manifest")
    if not isinstance(manifest, dict):
        raise Abstain("manifest is not an object")

    sha = _find_sha(manifest)
    if sha is not None:
        h = hashlib.sha256()
        with open(data_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest().lower() != sha:
            raise Abstain("data checksum does not match manifest")

    start, end = _get_period(manifest)

    try:
        table = pq.read_table(data_path)
    except Exception:
        raise Abstain("cannot read parquet data")
    try:
        df = table.to_pandas()
    except Exception:
        try:
            df = pd.DataFrame({n: table.column(n).to_pylist() for n in table.column_names})
        except Exception:
            raise Abstain("cannot convert data")
    raw_rows = len(df)

    # schema changes
    changes = manifest.get("schema_changes") or []
    if not isinstance(changes, list):
        raise Abstain("schema_changes is not a list")
    unit_overrides = {}
    renames = []
    for ch in changes:
        if not isinstance(ch, dict):
            raise Abstain("malformed schema change")
        typ = str(ch.get("type", "")).strip().lower()
        if typ == "rename":
            f, t = ch.get("from"), ch.get("to")
            if not isinstance(f, str) or not isinstance(t, str):
                raise Abstain("malformed rename")
            renames.append((f, t))
        elif typ in ("unit", "unit_change", "units", "change_unit", "unit_conversion"):
            col = ch.get("column") or ch.get("field") or ch.get("name")
            to_u = ch.get("to") or ch.get("unit") or ch.get("new_unit")
            if not col or not to_u:
                raise Abstain("malformed unit change")
            unit_overrides[col] = to_u
        elif typ in ("add", "add_column", "reorder", "type_change", "retype", "cast",
                     "change_type", "timezone", "tz_change"):
            continue
        else:
            raise Abstain(f"unsupported schema change type: {typ}")

    cols = list(df.columns)
    if len(set(cols)) != len(cols):
        raise Abstain("duplicate column names in data")
    for f, t in reversed(renames):
        if t in df.columns and f != t:
            if f in df.columns:
                raise Abstain(f"rename target conflict: both {f} and {t} present")
            df = df.rename(columns={t: f})
    # resolve renamed unit overrides
    for f, t in renames:
        if t in unit_overrides and f not in unit_overrides:
            unit_overrides[f] = unit_overrides.pop(t)

    # case-insensitive fallback
    lower_map = {}
    for c in df.columns:
        lower_map.setdefault(str(c).lower(), []).append(c)
    for c in CONTRACT_COLS:
        if c not in df.columns:
            m = lower_map.get(c.lower(), [])
            if len(m) == 1 and m[0] not in CONTRACT_COLS:
                df = df.rename(columns={m[0]: c})

    for c in REQUIRED:
        if c not in df.columns:
            raise Abstain(f"required column missing: {c}")

    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(object)

    # source timezone for naive timestamps
    src_tz = None
    for k in ("timezone", "time_zone", "tz", "timestamp_timezone"):
        v = manifest.get(k)
        if isinstance(v, str) and v.strip():
            src_tz = v.strip()
            break
    if src_tz:
        try:
            pd.Timestamp("2020-01-01").tz_localize(src_tz)
        except Exception:
            raise Abstain(f"unknown timezone {src_tz}")

    present = [c for c in CONTRACT_COLS if c in df.columns]
    work = pd.DataFrame(index=df.index)
    for c in present:
        if c in DT_COLS:
            work[c] = _to_local_dt(df[c], c, src_tz)
        else:
            work[c] = _to_float_series(df[c], c)

    # units
    units = manifest.get("units")
    units = dict(units) if isinstance(units, dict) else {}
    units.update(unit_overrides)
    if "trip_distance" in units:
        work["trip_distance"] = work["trip_distance"] * _distance_factor(units["trip_distance"])
    for k in ("distance_unit", "distance_units"):
        if k in manifest and "trip_distance" not in units:
            work["trip_distance"] = work["trip_distance"] * _distance_factor(manifest[k])
    cur = manifest.get("currency")
    global_money = _money_factor(cur) if cur is not None else 1.0
    for c in MONEY_COLS:
        if c in work.columns:
            fac = _money_factor(units[c]) if c in units else global_money
            if fac != 1.0:
                work[c] = work[c] * fac

    # required non-null
    for c in REQUIRED:
        n = int(work[c].isna().sum())
        if n:
            raise Abstain(f"{n} null values in required column {c}")

    for c in INT_COLS:
        if c in work.columns:
            v = work[c].dropna()
            if len(v) and not np.all(np.isfinite(v.values)) or (len(v) and not np.all(v.values == np.round(v.values))):
                raise Abstain(f"non-integer values in {c}")

    # dedup replays / detect conflicts
    work = work.drop_duplicates(subset=present, keep="first")
    if work["record_id"].duplicated().any():
        n = int(work["record_id"].duplicated().sum())
        raise Abstain(f"{n} conflicting records share a record_id")
    n = len(work)

    rc = _find_row_count(manifest)
    if rc is not None and rc != raw_rows and rc != n:
        raise Abstain(f"row count {raw_rows} does not match manifest {rc}")

    if n == 0:
        raise Abstain("no trips in data")

    pt = work["payment_type"].astype("int64")
    if not pt.isin(list(VALID_PAYMENT)).all():
        raise Abstain("invalid payment_type codes")

    td = work["trip_distance"]
    if not ((td > 0) & (td <= 100)).all():
        raise Abstain(f"{int((~((td > 0) & (td <= 100))).sum())} trip_distance values out of range")
    fa = work["fare_amount"]
    if not ((fa >= 0) & (fa <= 1000)).all():
        raise Abstain(f"{int((~((fa >= 0) & (fa <= 1000))).sum())} fare_amount values out of range")

    pu = work["tpep_pickup_datetime"]
    inside = (pu >= start) & (pu < end)
    if not inside.all():
        raise Abstain(f"{int((~inside).sum())} pickups outside reporting_period")

    total_fare = float(fa.sum())
    return {
        "status": "ok",
        "metrics": {
            "trip_count": int(n),
            "total_fare": total_fare,
            "mean_fare": float(fa.mean()),
            "mean_trip_distance": float(td.mean()),
            "card_share": float((pt == 1).sum()) / float(n),
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        if not os.path.exists(data_path):
            return _abstain("data file not found")
        if not os.path.exists(manifest_path):
            return _abstain("manifest file not found")
        return _analyze(data_path, manifest_path)
    except Abstain as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain(f"unexpected error: {type(e).__name__}")
