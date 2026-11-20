import json
import re

import numpy as np
import pandas as pd

TZ = "America/New_York"
CONTRACT = [
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]
REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
DATETIME_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
INT_COLS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"]
FLOAT_COLS = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
VALID_PAYMENT = {0, 1, 2, 3, 4, 5, 6}
_TZ_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$", re.IGNORECASE)


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _to_local_naive_ts(value):
    ts = pd.Timestamp(value)
    if ts is pd.NaT or pd.isna(ts):
        raise _Abstain("unparseable reporting_period boundary")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(TZ).tz_localize(None)
    return ts


def _parse_period(manifest):
    rp = manifest.get("reporting_period")
    if rp is None:
        raise _Abstain("manifest has no reporting_period")
    start = end = None
    if isinstance(rp, dict):
        for k in ("start", "from", "begin", "start_date", "start_inclusive"):
            if rp.get(k) is not None:
                start = rp[k]
                break
        for k in ("end", "to", "until", "end_date", "end_exclusive"):
            if rp.get(k) is not None:
                end = rp[k]
                break
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        start, end = rp
    elif isinstance(rp, str):
        s = rp.strip()
        if "/" in s:
            start, end = [p.strip() for p in s.split("/", 1)]
        elif re.fullmatch(r"\d{4}-\d{2}", s):
            p = pd.Period(s, freq="M")
            return p.start_time, (p + 1).start_time
        else:
            raise _Abstain("unrecognised reporting_period format")
    if start is None or end is None:
        raise _Abstain("incomplete reporting_period")
    start_ts, end_ts = _to_local_naive_ts(start), _to_local_naive_ts(end)
    if not start_ts < end_ts:
        raise _Abstain("invalid reporting_period")
    return start_ts, end_ts


def _apply_renames(df, manifest):
    changes = manifest.get("schema_changes") or []
    if not isinstance(changes, list):
        raise _Abstain("schema_changes is not a list")
    mapping = {}
    for ch in changes:
        if not isinstance(ch, dict):
            raise _Abstain("malformed schema_changes entry")
        if str(ch.get("type", "")).lower() == "rename":
            src, dst = ch.get("from"), ch.get("to")
            if not src or not dst:
                raise _Abstain("malformed rename entry")
            mapping[dst] = src
    if mapping:
        cols = list(df.columns)
        new_cols = []
        for c in cols:
            if c in mapping:
                new_cols.append(mapping[c])
            else:
                new_cols.append(c)
        # A renamed-from name still present alongside its new name is ambiguous
        if len(set(new_cols)) != len(new_cols):
            raise _Abstain("duplicate columns after applying renames")
        df.columns = new_cols
    if len(set(df.columns)) != len(df.columns):
        raise _Abstain("duplicate column names")
    return df


def _normalize_obj(s):
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    return s


def _to_datetime(s, name):
    s = _normalize_obj(s)
    notnull = s.notna()
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        out = s.dt.tz_convert(TZ).dt.tz_localize(None)
    elif pd.api.types.is_datetime64_dtype(s.dtype):
        out = pd.to_datetime(s)
    elif pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
        vals = pd.to_numeric(s, errors="coerce")
        mx = vals.abs().max()
        if pd.isna(mx):
            unit = "s"
        elif mx > 1e17:
            unit = "ns"
        elif mx > 1e14:
            unit = "us"
        elif mx > 1e11:
            unit = "ms"
        else:
            unit = "s"
        out = pd.to_datetime(vals, unit=unit, utc=True, errors="coerce").dt.tz_convert(TZ).dt.tz_localize(None)
    else:
        def conv(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return pd.NaT
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8", "replace")
            try:
                if isinstance(v, str):
                    v2 = v.strip()
                    if v2 == "":
                        return pd.NaT
                    ts = pd.Timestamp(v2)
                else:
                    ts = pd.Timestamp(v)
            except Exception:
                return None  # marker for unparseable
            if ts is pd.NaT or pd.isna(ts):
                return pd.NaT
            if ts.tzinfo is not None:
                ts = ts.tz_convert(TZ).tz_localize(None)
            return ts
        converted = [conv(v) for v in s.tolist()]
        if any(c is None for c in converted):
            raise _Abstain(f"unparseable datetime values in {name}")
        out = pd.Series(pd.to_datetime(pd.Series(converted, dtype=object), errors="coerce"), index=s.index)
        if isinstance(out.dtype, pd.DatetimeTZDtype):
            out = out.dt.tz_convert(TZ).dt.tz_localize(None)
    bad = notnull & out.isna()
    if name in REQUIRED and bad.any():
        raise _Abstain(f"unparseable datetime values in {name}")
    if bad.any():
        # tolerated for optional columns only when the original was blank-ish
        orig = s[bad]
        if not all(isinstance(v, str) and v.strip() == "" for v in orig.tolist()):
            raise _Abstain(f"unparseable datetime values in {name}")
    return out.astype("datetime64[ns]")


def _to_numeric(s, name):
    s = _normalize_obj(s)
    if pd.api.types.is_bool_dtype(s.dtype):
        raise _Abstain(f"boolean values in numeric column {name}")
    if pd.api.types.is_numeric_dtype(s.dtype):
        out = pd.to_numeric(s, errors="coerce").astype("float64")
    else:
        def clean(v):
            if v is None:
                return np.nan
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8", "replace")
            if isinstance(v, bool):
                return "bad"
            if isinstance(v, str):
                t = v.strip().replace(",", "").replace("$", "")
                return np.nan if t == "" else t
            return v
        cleaned = pd.Series([clean(v) for v in s.tolist()], index=s.index, dtype=object)
        out = pd.to_numeric(cleaned, errors="coerce").astype("float64")
        bad = cleaned.notna() & out.isna()
        if bad.any():
            raise _Abstain(f"unparseable numeric values in {name}")
    if np.isinf(out.to_numpy(dtype="float64", na_value=np.nan)).any():
        raise _Abstain(f"infinite values in {name}")
    return out


def _unit_factor(manifest, col):
    units = manifest.get("units")
    if not isinstance(units, dict) or col not in units:
        return 1.0
    u = str(units[col]).strip().lower()
    if col == "trip_distance":
        if u in ("mi", "mile", "miles"):
            return 1.0
        if u in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
            return 1.0 / 1.609344
        raise _Abstain(f"unsupported unit for {col}: {u}")
    if col in ("fare_amount", "tip_amount", "total_amount"):
        if u in ("usd", "dollar", "dollars", "$"):
            return 1.0
        if u in ("cents", "cent", "usd_cents"):
            return 0.01
        raise _Abstain(f"unsupported unit for {col}: {u}")
    return 1.0


def _run(data_path, manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception:
        raise _Abstain("manifest unreadable")
    if not isinstance(manifest, dict):
        raise _Abstain("manifest is not a JSON object")

    cur = manifest.get("currency")
    if cur is not None and str(cur).strip().upper() not in ("USD", ""):
        raise _Abstain("non-USD currency declared in manifest")

    start, end = _parse_period(manifest)

    try:
        df = pd.read_parquet(data_path)
    except Exception:
        raise _Abstain("data file unreadable")

    df = _apply_renames(df, manifest)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise _Abstain("missing required columns: " + ", ".join(missing))

    raw_rows = len(df)
    present = [c for c in CONTRACT if c in df.columns]
    data = pd.DataFrame(index=df.index)
    for c in present:
        if c in DATETIME_COLS:
            data[c] = _to_datetime(df[c], c)
        else:
            data[c] = _to_numeric(df[c], c)

    for c in REQUIRED:
        if data[c].isna().any():
            raise _Abstain(f"null values in required column {c}")

    for c in INT_COLS:
        if c in data.columns:
            v = data[c].dropna()
            if (v != np.floor(v)).any():
                raise _Abstain(f"non-integer values in {c}")

    for c in FLOAT_COLS:
        if c in data.columns:
            f = _unit_factor(manifest, c)
            if f != 1.0:
                data[c] = data[c] * f

    # Replays vs conflicts
    data = data.reset_index(drop=True)
    data = data.drop_duplicates(subset=present, keep="first")
    if data["record_id"].duplicated().any():
        raise _Abstain("conflicting records share a record_id")

    if len(data) == 0:
        raise _Abstain("no trips in data")

    for key in ("row_count", "record_count", "num_rows", "expected_row_count"):
        if key in manifest and manifest[key] is not None:
            try:
                expected = int(manifest[key])
            except Exception:
                raise _Abstain("invalid row count in manifest")
            if expected not in (raw_rows, len(data)):
                raise _Abstain("row count does not match manifest")

    pt = data["payment_type"]
    if not pt.isin(list(VALID_PAYMENT)).all():
        raise _Abstain("invalid payment_type codes")

    dist = data["trip_distance"]
    if not ((dist > 0) & (dist <= 100)).all():
        raise _Abstain("trip_distance out of valid range")

    fare = data["fare_amount"]
    if not ((fare >= 0) & (fare <= 1000)).all():
        raise _Abstain("fare_amount out of valid range")

    pu = data["tpep_pickup_datetime"]
    if not ((pu >= start) & (pu < end)).all():
        raise _Abstain("pickup times outside reporting_period")

    n = int(len(data))
    total_fare = float(fare.sum())
    return {
        "status": "ok",
        "metrics": {
            "trip_count": n,
            "total_fare": total_fare,
            "mean_fare": total_fare / n,
            "mean_trip_distance": float(dist.sum()) / n,
            "card_share": float((pt == 1).sum()) / n,
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _run(data_path, manifest_path)
    except _Abstain as e:
        return _abstain(str(e))
    except Exception as e:  # never return untrustworthy numbers
        return _abstain(f"unexpected error: {type(e).__name__}")
