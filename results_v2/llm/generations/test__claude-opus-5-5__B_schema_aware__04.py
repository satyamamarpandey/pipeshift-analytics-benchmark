"""Monthly summary of NYC yellow taxi trips, validated against the data contract."""

import hashlib
import json
import math
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NY_TZ = "America/New_York"

CONTRACT_COLUMNS = [
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]
REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance",
            "payment_type", "fare_amount"]
DATETIME_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
INT_COLS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"]
FLOAT_COLS = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
VALID_PAYMENT = {0, 1, 2, 3, 4, 5, 6}

IGNORABLE_CHANGE_TYPES = {
    "add", "add_column", "added", "column_added", "reorder", "reorder_columns",
    "type_change", "cast", "retype", "change_type", "dtype_change",
    "timezone", "tz_aware", "timezone_change",
}


class Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


# --------------------------------------------------------------------------- #
# Manifest handling
# --------------------------------------------------------------------------- #
def _to_local_naive_scalar(value):
    ts = pd.Timestamp(value)
    if ts is pd.NaT or pd.isna(ts):
        raise Abstain("unparseable reporting_period bound")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    return ts


def _parse_period(manifest):
    rp = manifest.get("reporting_period")
    if rp is None:
        for k in ("period", "reporting_month", "month"):
            if k in manifest:
                rp = manifest[k]
                break
    if rp is None and "period_start" in manifest and "period_end" in manifest:
        rp = {"start": manifest["period_start"], "end": manifest["period_end"]}
    if rp is None:
        raise Abstain("manifest has no reporting_period")

    start = end = None
    if isinstance(rp, dict):
        for k in ("start", "from", "begin", "start_inclusive", "period_start"):
            if k in rp and rp[k] is not None:
                start = rp[k]
                break
        for k in ("end", "to", "until", "end_exclusive", "period_end"):
            if k in rp and rp[k] is not None:
                end = rp[k]
                break
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        start, end = rp
    elif isinstance(rp, str):
        m = re.fullmatch(r"\s*(\d{4})-(\d{1,2})\s*", rp)
        if m:
            s = pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
            return s, s + pd.DateOffset(months=1)
        raise Abstain("unrecognised reporting_period format")
    if start is None or end is None:
        raise Abstain("reporting_period missing start or end")

    start_ts = _to_local_naive_scalar(start)
    if isinstance(end, str) and re.fullmatch(r"\s*\d{4}-\d{1,2}\s*", end):
        end_ts = _to_local_naive_scalar(end) + pd.DateOffset(months=1) \
            if False else _to_local_naive_scalar(end)
    else:
        end_ts = _to_local_naive_scalar(end)
    if not start_ts < end_ts:
        raise Abstain("reporting_period start is not before end")
    return start_ts, end_ts


def _verify_checksum(manifest, data_path):
    for key, algo in (("sha256", "sha256"), ("md5", "md5"), ("checksum", None)):
        if key not in manifest or manifest[key] in (None, ""):
            continue
        expected = manifest[key]
        if isinstance(expected, dict):
            algo = expected.get("algorithm", expected.get("algo", algo))
            expected = expected.get("value", expected.get("digest"))
        if not isinstance(expected, str):
            continue
        expected = expected.strip().lower()
        if ":" in expected and algo is None:
            algo, expected = expected.split(":", 1)
        if algo is None:
            algo = {32: "md5", 40: "sha1", 64: "sha256"}.get(len(expected))
        if algo is None:
            continue
        algo = str(algo).lower().replace("-", "")
        try:
            h = hashlib.new(algo)
        except ValueError:
            continue
        with open(data_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest().lower() != expected:
            raise Abstain("data file checksum does not match manifest")


def _apply_schema_changes(df, manifest):
    changes = manifest.get("schema_changes") or []
    if isinstance(changes, dict):
        changes = [changes]
    if not isinstance(changes, list):
        raise Abstain("schema_changes is not a list")

    renames = []
    for ch in changes:
        if not isinstance(ch, dict):
            raise Abstain("malformed schema_changes entry")
        ctype = str(ch.get("type", "")).strip().lower()
        if ctype == "rename":
            src, dst = ch.get("from"), ch.get("to")
            if not isinstance(src, str) or not isinstance(dst, str):
                raise Abstain("malformed rename in schema_changes")
            renames.append((src, dst))
        elif ctype in IGNORABLE_CHANGE_TYPES:
            continue
        else:
            raise Abstain(f"unsupported schema change type: {ctype!r}")

    # Undo renames in reverse order (new name -> original name).
    for src, dst in reversed(renames):
        if dst in df.columns:
            if src in df.columns and src != dst:
                raise Abstain(f"both {src!r} and renamed {dst!r} present")
            df = df.rename(columns={dst: src})
    return df


def _unit_factors(manifest):
    """Return multiplicative factors for distance and fare, from optional unit metadata."""
    units = manifest.get("units") or {}
    if not isinstance(units, dict):
        units = {}
    dist_factor = 1.0
    fare_factor = 1.0

    du = units.get("trip_distance", manifest.get("distance_unit"))
    if du is not None:
        du = str(du).strip().lower()
        if du in ("mi", "mile", "miles"):
            dist_factor = 1.0
        elif du in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
            dist_factor = 1.0 / 1.609344
        else:
            raise Abstain(f"unsupported trip_distance unit: {du}")

    cur = manifest.get("currency")
    if cur is not None and str(cur).strip().upper() not in ("USD", "US$", "$"):
        raise Abstain(f"unsupported currency: {cur}")
    fu = units.get("fare_amount", manifest.get("fare_unit"))
    if fu is not None:
        fu = str(fu).strip().lower()
        if fu in ("usd", "dollar", "dollars", "$"):
            fare_factor = 1.0
        elif fu in ("cent", "cents", "usd_cents"):
            fare_factor = 0.01
        else:
            raise Abstain(f"unsupported fare_amount unit: {fu}")
    return dist_factor, fare_factor


# --------------------------------------------------------------------------- #
# Column coercion
# --------------------------------------------------------------------------- #
def _decategorize(s):
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.astype(object)
    return s


def _coerce_numeric(s, name):
    s = _decategorize(s)
    if pd.api.types.is_bool_dtype(s):
        raise Abstain(f"column {name} has boolean type")
    if pd.api.types.is_numeric_dtype(s):
        out = s.astype("float64")
    else:
        obj = s.astype(object)
        obj = obj.map(lambda v: v.strip() if isinstance(v, str) else v)
        obj = obj.map(lambda v: None if isinstance(v, str) and v == "" else v)
        out = pd.to_numeric(obj, errors="coerce").astype("float64")
        bad = out.isna() & obj.notna()
        if bad.any():
            raise Abstain(f"non-numeric values in {name}")
    if np.isinf(out.to_numpy(dtype="float64", na_value=np.nan)).any():
        raise Abstain(f"infinite values in {name}")
    return out


def _epoch_to_local(s):
    vals = s.astype("float64")
    nonnull = vals.dropna().abs()
    if nonnull.empty:
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    mx = nonnull.max()
    unit = "ns" if mx > 1e17 else "us" if mx > 1e14 else "ms" if mx > 1e11 else "s"
    ts = pd.to_datetime(vals, unit=unit, utc=True)
    return ts.dt.tz_convert(NY_TZ).dt.tz_localize(None)


def _coerce_datetime(s, name):
    s = _decategorize(s)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(NY_TZ).dt.tz_localize(None).astype("datetime64[ns]")
    if pd.api.types.is_datetime64_dtype(s):
        return s.astype("datetime64[ns]")
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return _epoch_to_local(s)

    obj = s.astype(object)
    uniques = pd.unique(obj[obj.notna()])
    mapping = {}
    for v in uniques:
        if isinstance(v, str):
            txt = v.strip()
            if txt == "":
                mapping[v] = pd.NaT
                continue
            try:
                ts = pd.Timestamp(txt)
            except Exception:
                raise Abstain(f"unparseable datetime in {name}")
        else:
            try:
                ts = pd.Timestamp(v)
            except Exception:
                raise Abstain(f"unparseable datetime in {name}")
        if ts is pd.NaT or pd.isna(ts):
            raise Abstain(f"unparseable datetime in {name}")
        if ts.tzinfo is not None:
            ts = ts.tz_convert(NY_TZ).tz_localize(None)
        mapping[v] = ts
    out = obj.map(lambda v: mapping.get(v, pd.NaT) if v is not None and not
                  (isinstance(v, float) and math.isnan(v)) else pd.NaT)
    return pd.to_datetime(out).astype("datetime64[ns]")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def _analyze(data_path, manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception:
        raise Abstain("manifest unreadable")
    if not isinstance(manifest, dict):
        raise Abstain("manifest is not a JSON object")

    period_start, period_end = _parse_period(manifest)
    _verify_checksum(manifest, data_path)
    dist_factor, fare_factor = _unit_factors(manifest)

    try:
        table = pq.read_table(data_path)
        df = table.to_pandas()
    except Exception:
        raise Abstain("data file unreadable")

    if df.columns.duplicated().any():
        raise Abstain("duplicate column names in data")
    df = _apply_schema_changes(df, manifest)
    if df.columns.duplicated().any():
        raise Abstain("duplicate column names after applying renames")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise Abstain(f"required columns missing: {missing}")

    raw_rows = len(df)
    expected_rows = None
    for k in ("row_count", "expected_row_count", "num_rows", "record_count"):
        if isinstance(manifest.get(k), (int, float)) and not isinstance(manifest.get(k), bool):
            expected_rows = int(manifest[k])
            break

    present = [c for c in CONTRACT_COLUMNS if c in df.columns]
    work = pd.DataFrame(index=df.index)
    for c in present:
        if c in DATETIME_COLS:
            work[c] = _coerce_datetime(df[c], c)
        else:
            work[c] = _coerce_numeric(df[c], c)

    # Required non-null
    for c in REQUIRED:
        if work[c].isna().any():
            raise Abstain(f"null values in required column {c}")

    # Integer-valued columns must be integral
    for c in INT_COLS:
        if c in work.columns:
            v = work[c].dropna().to_numpy(dtype="float64")
            if v.size and not np.all(np.floor(v) == v):
                raise Abstain(f"non-integer values in {c}")

    # Units
    if dist_factor != 1.0:
        work["trip_distance"] = work["trip_distance"] * dist_factor
    if fare_factor != 1.0:
        for c in ("fare_amount", "tip_amount", "total_amount"):
            if c in work.columns:
                work[c] = work[c] * fare_factor

    if raw_rows == 0:
        raise Abstain("no records in data")

    # Replays and conflicts
    deduped = work.drop_duplicates(subset=present, keep="first")
    if deduped["record_id"].duplicated().any():
        raise Abstain("conflicting records share a record_id")

    if expected_rows is not None and expected_rows not in (raw_rows, len(deduped)):
        raise Abstain("row count does not match manifest")

    # Domain validation
    td = deduped["trip_distance"]
    if not ((td > 0) & (td <= 100)).all():
        raise Abstain("trip_distance outside valid range (0, 100]")
    fa = deduped["fare_amount"]
    if not ((fa >= 0) & (fa <= 1000)).all():
        raise Abstain("fare_amount outside valid range [0, 1000]")
    if not deduped["payment_type"].isin(VALID_PAYMENT).all():
        raise Abstain("unknown payment_type codes")
    pu = deduped["tpep_pickup_datetime"]
    if not ((pu >= period_start) & (pu < period_end)).all():
        raise Abstain("pickup times outside reporting_period")

    n = len(deduped)
    if n == 0:
        raise Abstain("no valid trips")

    fares = deduped["fare_amount"].to_numpy(dtype="float64")
    dists = deduped["trip_distance"].to_numpy(dtype="float64")
    pay = deduped["payment_type"].to_numpy(dtype="float64")

    total_fare = float(fares.sum())
    metrics = {
        "trip_count": int(n),
        "total_fare": total_fare,
        "mean_fare": float(total_fare / n),
        "mean_trip_distance": float(dists.sum() / n),
        "card_share": float((pay == 1).sum() / n),
    }
    for k, v in metrics.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise Abstain(f"non-finite metric {k}")
    return {"status": "ok", "metrics": metrics, "reason": ""}


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except Abstain as exc:
        return _abstain(str(exc))
    except Exception as exc:  # any unexpected failure -> untrustworthy
        return _abstain(f"unexpected error: {type(exc).__name__}")
