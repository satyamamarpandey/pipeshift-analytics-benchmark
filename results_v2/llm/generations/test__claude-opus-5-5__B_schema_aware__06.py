"""Monthly summary of NYC TLC yellow taxi trips, validated against a data contract."""

import json
import math
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NY_TZ = "America/New_York"

CONTRACT_COLUMNS = [
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
REQUIRED = [
    "record_id",
    "tpep_pickup_datetime",
    "trip_distance",
    "payment_type",
    "fare_amount",
]
INT_COLS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"]
FLOAT_COLS = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
DT_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
VALID_PAYMENT = {0, 1, 2, 3, 4, 5, 6}
BENIGN_CHANGE_TYPES = {
    "add", "add_column", "added", "column_added", "reorder", "reorder_columns",
    "type_change", "change_type", "cast", "retype", "timezone", "tz_aware",
}


class Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


# --------------------------------------------------------------------------- #
# Manifest handling
# --------------------------------------------------------------------------- #
def _to_ny_naive_scalar(value):
    ts = pd.Timestamp(value)
    if ts is pd.NaT or pd.isna(ts):
        raise Abstain("unparseable reporting_period bound")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    return ts


def _parse_period(manifest):
    rp = manifest.get("reporting_period")
    if rp is None:
        for k in ("period", "reportingPeriod", "reporting_month"):
            if k in manifest:
                rp = manifest[k]
                break
    if rp is None:
        raise Abstain("manifest lacks reporting_period")

    if isinstance(rp, dict):
        start = None
        end = None
        for k in ("start", "begin", "from", "start_inclusive", "start_date"):
            if rp.get(k) is not None:
                start = rp[k]
                break
        for k in ("end", "stop", "to", "end_exclusive", "end_date"):
            if rp.get(k) is not None:
                end = rp[k]
                break
        if start is None or end is None:
            raise Abstain("reporting_period missing start or end")
        try:
            s = _to_ny_naive_scalar(start)
            e = _to_ny_naive_scalar(end)
        except Abstain:
            raise
        except Exception:
            raise Abstain("unparseable reporting_period")
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        try:
            s = _to_ny_naive_scalar(rp[0])
            e = _to_ny_naive_scalar(rp[1])
        except Abstain:
            raise
        except Exception:
            raise Abstain("unparseable reporting_period")
    elif isinstance(rp, str):
        m = re.fullmatch(r"\s*(\d{4})-(\d{1,2})\s*", rp)
        if not m:
            raise Abstain("unrecognized reporting_period format")
        y, mo = int(m.group(1)), int(m.group(2))
        if not 1 <= mo <= 12:
            raise Abstain("invalid reporting_period month")
        s = pd.Timestamp(year=y, month=mo, day=1)
        e = s + pd.offsets.MonthBegin(1)
    else:
        raise Abstain("unrecognized reporting_period format")

    if not s < e:
        raise Abstain("empty or inverted reporting_period")
    return s, e


def _apply_schema_changes(df, manifest):
    changes = manifest.get("schema_changes") or []
    if not isinstance(changes, list):
        raise Abstain("schema_changes is not a list")
    for ch in changes:
        if not isinstance(ch, dict):
            raise Abstain("malformed schema_changes entry")
        ctype = str(ch.get("type", "")).strip().lower()
        if ctype == "rename":
            src = ch.get("from")
            dst = ch.get("to")
            if not isinstance(src, str) or not isinstance(dst, str):
                raise Abstain("malformed rename entry")
            if src == dst:
                continue
            if dst in df.columns:
                if src in df.columns:
                    raise Abstain(f"ambiguous rename: both {src!r} and {dst!r} present")
                df = df.rename(columns={dst: src})
            elif src in df.columns:
                # already in original naming
                pass
            # if neither present, missing-column check will catch required ones
        elif ctype in BENIGN_CHANGE_TYPES:
            continue
        else:
            raise Abstain(f"unsupported schema change type {ctype!r}")
    return df


def _check_units(manifest):
    """Abstain on declared units/currency incompatible with the contract."""
    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from walk(v, f"{path}.{k}".lower())
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v, path)
        else:
            yield path, obj

    for path, val in walk(manifest):
        if not isinstance(val, str):
            continue
        v = val.strip().lower()
        if path.endswith("currency") and v not in ("usd", "us$", "$", "us dollar", "dollars"):
            raise Abstain(f"unsupported currency {val!r}")
        if ("distance_unit" in path or path.endswith("distance.unit")) and v not in (
            "mi", "mile", "miles"
        ):
            raise Abstain(f"unsupported distance unit {val!r}")


# --------------------------------------------------------------------------- #
# Column coercion
# --------------------------------------------------------------------------- #
def _null_mask(s):
    return s.isna()


def _coerce_numeric(s, name):
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    if pd.api.types.is_bool_dtype(s.dtype):
        raise Abstain(f"column {name} has boolean type")
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.astype("float64")
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        raise Abstain(f"column {name} has datetime type")
    orig_null = s.isna()

    def norm(v):
        if v is None:
            return None
        if isinstance(v, (bool, np.bool_)):
            return np.nan  # will be flagged
        if isinstance(v, bytes):
            try:
                v = v.decode("utf-8")
            except Exception:
                return np.nan
        if isinstance(v, str):
            t = v.strip()
            if t == "" or t.lower() in ("nan", "null", "none", "na", "n/a"):
                return None
            return t
        try:
            return float(v)
        except Exception:
            return np.nan

    obj = s.astype(object).map(norm)
    newly_null = obj.isna() & ~orig_null
    # strings that normalize to "null-like" are considered nulls
    bad_marker = s.astype(object).map(
        lambda v: isinstance(v, (bool, np.bool_))
    )
    out = pd.to_numeric(obj, errors="coerce").astype("float64")
    unparsable = out.isna() & obj.notna()
    if unparsable.any() or bad_marker.any():
        raise Abstain(f"column {name} contains non-numeric values")
    _ = newly_null
    return out


def _coerce_int(s, name):
    f = _coerce_numeric(s, name)
    nonnull = f.dropna()
    if len(nonnull) and (
        not np.all(np.isfinite(nonnull.values))
        or not np.all(nonnull.values == np.floor(nonnull.values))
    ):
        raise Abstain(f"column {name} contains non-integer values")
    return f


def _elementwise_datetime(s, name):
    def conv(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return pd.NaT
        if isinstance(v, str):
            t = v.strip()
            if t == "" or t.lower() in ("nan", "nat", "null", "none"):
                return pd.NaT
            v = t
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            raise Abstain(f"column {name} has numeric timestamps of unknown epoch/zone")
        try:
            ts = pd.Timestamp(v)
        except Exception:
            raise Abstain(f"column {name} contains unparseable timestamps")
        if ts is pd.NaT:
            return pd.NaT
        if ts.tzinfo is not None:
            ts = ts.tz_convert(NY_TZ).tz_localize(None)
        return ts

    vals = s.astype(object)
    cache = {}
    out = []
    for v in vals:
        try:
            key = v if not isinstance(v, float) else ("f", v)
            hash(key)
        except TypeError:
            key = None
        if key is not None and key in cache:
            out.append(cache[key])
            continue
        r = conv(v)
        if key is not None:
            cache[key] = r
        out.append(r)
    return pd.to_datetime(pd.Series(out, index=s.index, dtype=object))


def _coerce_datetime(s, name):
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(NY_TZ).dt.tz_localize(None).astype("datetime64[ns]")
    if pd.api.types.is_datetime64_dtype(s.dtype):
        return s.astype("datetime64[ns]")
    if pd.api.types.is_bool_dtype(s.dtype) or pd.api.types.is_numeric_dtype(s.dtype):
        if s.isna().all():
            return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        raise Abstain(f"column {name} has numeric timestamps of unknown epoch/zone")
    out = _elementwise_datetime(s, name)
    if isinstance(out.dtype, pd.DatetimeTZDtype):
        out = out.dt.tz_convert(NY_TZ).dt.tz_localize(None)
    return out.astype("datetime64[ns]")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _run(data_path, manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception:
        raise Abstain("cannot read manifest")
    if not isinstance(manifest, dict):
        raise Abstain("manifest is not a JSON object")

    start, end = _parse_period(manifest)
    _check_units(manifest)

    try:
        table = pq.read_table(data_path)
        df = table.to_pandas()
    except Exception:
        raise Abstain("cannot read parquet data")

    if df.columns.duplicated().any():
        raise Abstain("duplicate column names in data")

    df = _apply_schema_changes(df, manifest)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise Abstain(f"missing required columns: {missing}")

    raw_rows = len(df)
    present = [c for c in CONTRACT_COLUMNS if c in df.columns]
    work = pd.DataFrame(index=df.index)
    for c in present:
        if c in INT_COLS:
            work[c] = _coerce_int(df[c], c)
        elif c in FLOAT_COLS:
            work[c] = _coerce_numeric(df[c], c)
        else:
            work[c] = _coerce_datetime(df[c], c)

    # Required non-null
    for c in REQUIRED:
        if work[c].isna().any():
            raise Abstain(f"null values in required column {c}")

    # Domain rules
    td = work["trip_distance"]
    if not ((td > 0) & (td <= 100)).all():
        raise Abstain("trip_distance outside valid range (0, 100]")
    fa = work["fare_amount"]
    if not ((fa >= 0) & (fa <= 1000)).all():
        raise Abstain("fare_amount outside valid range [0, 1000]")
    if not work["payment_type"].isin(list(VALID_PAYMENT)).all():
        raise Abstain("payment_type contains undefined codes")
    for c in ("tip_amount", "total_amount"):
        if c in work.columns:
            v = work[c].dropna()
            if len(v) and not np.all(np.isfinite(v.values)):
                raise Abstain(f"non-finite values in {c}")

    # Replays vs conflicts
    dedup = work.drop_duplicates(subset=present, keep="first")
    if dedup["record_id"].duplicated().any():
        raise Abstain("conflicting records share a record_id")

    # Reporting period
    pu = dedup["tpep_pickup_datetime"]
    if not ((pu >= start) & (pu < end)).all():
        raise Abstain("trips outside the manifest reporting_period")

    # Optional manifest row-count check
    for k in ("row_count", "record_count", "num_rows", "n_rows"):
        if k in manifest and manifest[k] is not None:
            try:
                expected = int(manifest[k])
            except Exception:
                continue
            if expected not in (raw_rows, len(dedup)):
                raise Abstain("row count disagrees with manifest")

    n = int(len(dedup))
    if n == 0:
        raise Abstain("no trips in data")

    fares = dedup["fare_amount"].to_numpy(dtype="float64")
    dists = dedup["trip_distance"].to_numpy(dtype="float64")
    total_fare = float(math.fsum(fares))
    metrics = {
        "trip_count": n,
        "total_fare": total_fare,
        "mean_fare": total_fare / n,
        "mean_trip_distance": float(math.fsum(dists)) / n,
        "card_share": float((dedup["payment_type"] == 1).sum()) / n,
    }
    for k, v in metrics.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise Abstain(f"non-finite metric {k}")
    return metrics


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        metrics = _run(data_path, manifest_path)
    except Abstain as exc:
        return _abstain(str(exc))
    except Exception as exc:  # any unexpected failure -> untrustworthy
        return _abstain(f"unexpected error: {type(exc).__name__}")
    return {"status": "ok", "metrics": metrics, "reason": ""}
