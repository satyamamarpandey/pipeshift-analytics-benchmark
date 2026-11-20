"""Monthly summary of NYC yellow taxi trips from a Parquet snapshot plus a JSON manifest.

The module checks the input against the data contract. It returns metrics only when
the result can be trusted. Otherwise it abstains and gives a short reason.
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime

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
REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
INT_COLUMNS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"]
FLOAT_COLUMNS = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
DATETIME_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
VALID_PAYMENT_CODES = {0, 1, 2, 3, 4, 5, 6}


class ContractError(Exception):
    """Raised when the data cannot produce a trustworthy result."""


def _abstain(reason: str) -> dict:
    return {"status": "abstain", "metrics": None, "reason": reason}


# --------------------------------------------------------------------------- manifest
def _parse_bound(value, src_tz):
    if value is None:
        raise ContractError("reporting_period bound missing")
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:
        raise ContractError(f"unparseable reporting_period bound {value!r}") from exc
    if ts is pd.NaT:
        raise ContractError("reporting_period bound is NaT")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    return ts


def _reporting_period(manifest: dict):
    rp = manifest.get("reporting_period")
    if rp is None:
        for key in ("period", "reportingPeriod", "reporting_month", "month"):
            if key in manifest:
                rp = manifest[key]
                break
    if rp is None:
        raise ContractError("manifest has no reporting_period")
    if isinstance(rp, dict):
        start = next((rp[k] for k in ("start", "from", "begin", "start_inclusive") if k in rp), None)
        end = next((rp[k] for k in ("end", "to", "until", "end_exclusive") if k in rp), None)
        if start is None or end is None:
            raise ContractError("reporting_period lacks start/end")
        start, end = _parse_bound(start, None), _parse_bound(end, None)
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        start, end = _parse_bound(rp[0], None), _parse_bound(rp[1], None)
    elif isinstance(rp, str):
        text = rp.strip()
        try:
            start = pd.Timestamp(datetime.strptime(text, "%Y-%m"))
        except ValueError:
            raise ContractError(f"unrecognised reporting_period {rp!r}")
        end = start + pd.offsets.MonthBegin(1)
    else:
        raise ContractError("unrecognised reporting_period format")
    if not start < end:
        raise ContractError("reporting_period start is not before end")
    return start, end


def _source_timezone(manifest: dict):
    for key in ("timezone", "source_timezone", "timestamp_timezone", "tz"):
        val = manifest.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _unit_factors(manifest: dict):
    """Return the multipliers that convert fare_amount to USD and trip_distance to miles."""
    fare_factor = 1.0
    dist_factor = 1.0
    units = manifest.get("units")
    if units is None:
        return fare_factor, dist_factor
    if not isinstance(units, dict):
        raise ContractError("manifest units not understood")
    fare_unit = units.get("fare_amount")
    if fare_unit is not None:
        u = str(fare_unit).strip().lower()
        if u in ("usd", "dollar", "dollars", "$"):
            fare_factor = 1.0
        elif u in ("cents", "usd_cents", "cent", "usd cents"):
            fare_factor = 0.01
        else:
            raise ContractError(f"unsupported fare_amount unit {fare_unit!r}")
    dist_unit = units.get("trip_distance")
    if dist_unit is not None:
        u = str(dist_unit).strip().lower()
        if u in ("mi", "mile", "miles"):
            dist_factor = 1.0
        elif u in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
            dist_factor = 1.0 / 1.609344
        elif u in ("m", "meter", "meters", "metre", "metres"):
            dist_factor = 1.0 / 1609.344
        else:
            raise ContractError(f"unsupported trip_distance unit {dist_unit!r}")
    return fare_factor, dist_factor


def _expected_row_count(manifest: dict):
    for key in ("row_count", "record_count", "num_rows", "rows"):
        val = manifest.get(key)
        if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
            return int(val)
    return None


# --------------------------------------------------------------------------- coercion
def _to_numeric(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        raise ContractError(f"column {name} has boolean type")
    if pd.api.types.is_numeric_dtype(series):
        out = series.astype("float64")
    else:
        work = series.map(lambda v: v.strip() if isinstance(v, str) else v)
        work = work.map(lambda v: None if isinstance(v, str) and v == "" else v)
        out = pd.to_numeric(work, errors="coerce").astype("float64")
        bad = out.isna() & work.notna()
        if bad.any():
            raise ContractError(f"column {name} has non-numeric values")
    if np.isinf(out.to_numpy(dtype="float64", na_value=np.nan)).any():
        raise ContractError(f"column {name} has infinite values")
    return out


def _epoch_to_local(values: pd.Series) -> pd.Series:
    arr = values.astype("float64")
    mag = arr.abs().max()
    if pd.isna(mag):
        unit = "s"
    elif mag > 1e17:
        unit = "ns"
    elif mag > 1e14:
        unit = "us"
    elif mag > 1e11:
        unit = "ms"
    else:
        unit = "s"
    ts = pd.to_datetime(arr, unit=unit, utc=True)
    return ts.dt.tz_convert(NY_TZ).dt.tz_localize(None)


def _localize_naive(s: pd.Series, src_tz):
    if src_tz is None or src_tz in (NY_TZ,):
        return s
    try:
        return s.dt.tz_localize(src_tz).dt.tz_convert(NY_TZ).dt.tz_localize(None)
    except Exception as exc:
        raise ContractError(f"cannot apply source timezone {src_tz}") from exc


def _parse_scalar_datetime(v, src_tz):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return pd.NaT
    if isinstance(v, str):
        t = v.strip()
        if not t:
            return pd.NaT
        ts = pd.Timestamp(t)
    elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return None  # signal numeric
    else:
        ts = pd.Timestamp(v)
    if ts is pd.NaT:
        return pd.NaT
    if ts.tzinfo is not None:
        return ts.tz_convert(NY_TZ).tz_localize(None)
    if src_tz and src_tz != NY_TZ:
        return ts.tz_localize(src_tz).tz_convert(NY_TZ).tz_localize(None)
    return ts


def _to_local_datetime(series: pd.Series, name: str, src_tz) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(NY_TZ).dt.tz_localize(None).astype("datetime64[ns]")
    if pd.api.types.is_datetime64_dtype(series):
        return _localize_naive(series.astype("datetime64[ns]"), src_tz)
    if pd.api.types.is_bool_dtype(series):
        raise ContractError(f"column {name} has boolean type")
    if pd.api.types.is_numeric_dtype(series):
        return _epoch_to_local(series)

    # Object or string values. Try the fast vectorised path first.
    notna = series.notna() & ~series.map(lambda v: isinstance(v, str) and v.strip() == "")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            all_str = series[notna].map(lambda v: isinstance(v, str)).all()
            if all_str:
                parsed = pd.to_datetime(series.where(notna), format="ISO8601", errors="raise")
                if isinstance(parsed.dtype, pd.DatetimeTZDtype):
                    return parsed.dt.tz_convert(NY_TZ).dt.tz_localize(None).astype("datetime64[ns]")
                if pd.api.types.is_datetime64_dtype(parsed):
                    has_offset = series[notna].map(
                        lambda v: v.strip().endswith("Z") or ("+" in v[10:]) or ("-" in v[10:])
                    ).any()
                    if not has_offset:
                        return _localize_naive(parsed.astype("datetime64[ns]"), src_tz)
        except Exception:
            pass

    # Slow path: parse each value on its own.
    out = []
    for v in series.tolist():
        try:
            r = _parse_scalar_datetime(v, src_tz)
        except Exception as exc:
            raise ContractError(f"column {name} has unparseable timestamps") from exc
        if r is None:
            raise ContractError(f"column {name} mixes numeric and text timestamps")
        out.append(r)
    result = pd.Series(pd.to_datetime(out), index=series.index).astype("datetime64[ns]")
    if (result.isna() & notna).any():
        raise ContractError(f"column {name} has unparseable timestamps")
    return result


# --------------------------------------------------------------------------- main
def _load_frame(data_path: str) -> pd.DataFrame:
    table = pq.read_table(data_path)
    try:
        return table.to_pandas()
    except Exception:
        return table.to_pandas(timestamp_as_object=True)


def _apply_renames(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    changes = manifest.get("schema_changes") or []
    if not isinstance(changes, list):
        raise ContractError("schema_changes is not a list")
    mapping = {}
    for ch in changes:
        if not isinstance(ch, dict):
            raise ContractError("schema_changes entry not understood")
        if str(ch.get("type", "")).lower() != "rename":
            continue
        src, dst = ch.get("from"), ch.get("to")
        if not isinstance(src, str) or not isinstance(dst, str):
            raise ContractError("rename entry lacks from/to")
        if dst in df.columns:
            if src in df.columns and src != dst:
                raise ContractError(f"both {src} and renamed {dst} present")
            mapping[dst] = src
    if mapping:
        df = df.rename(columns=mapping)
    if df.columns.duplicated().any():
        raise ContractError("duplicate column names after renames")
    return df


def _analyze(data_path: str, manifest_path: str) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ContractError("manifest is not a JSON object")

    start, end = _reporting_period(manifest)
    src_tz = _source_timezone(manifest)
    fare_factor, dist_factor = _unit_factors(manifest)

    df = _load_frame(data_path)
    raw_rows = len(df)
    df = _apply_renames(df, manifest)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ContractError(f"required columns missing: {', '.join(missing)}")

    present = [c for c in CONTRACT_COLUMNS if c in df.columns]
    df = df[present].copy()

    for c in DATETIME_COLUMNS:
        if c in df.columns:
            df[c] = _to_local_datetime(df[c], c, src_tz)
    for c in INT_COLUMNS + FLOAT_COLUMNS:
        if c in df.columns:
            df[c] = _to_numeric(df[c], c)

    for c in INT_COLUMNS:
        if c in df.columns:
            vals = df[c].dropna()
            if len(vals) and not np.all(np.floor(vals.to_numpy()) == vals.to_numpy()):
                raise ContractError(f"column {c} has non-integer values")

    nulls = [c for c in REQUIRED if df[c].isna().any()]
    if nulls:
        raise ContractError(f"nulls in required columns: {', '.join(nulls)}")

    if fare_factor != 1.0:
        for c in ("fare_amount", "tip_amount", "total_amount"):
            if c in df.columns:
                df[c] = df[c] * fare_factor
    if dist_factor != 1.0:
        df["trip_distance"] = df["trip_distance"] * dist_factor

    # Replays: identical rows count once. Conflicts: same id with different values.
    df = df.drop_duplicates(keep="first")
    if df["record_id"].duplicated().any():
        raise ContractError("conflicting records share a record_id")

    expected = _expected_row_count(manifest)
    if expected is not None and expected not in (raw_rows, len(df)):
        raise ContractError(
            f"manifest row count {expected} does not match data ({raw_rows} rows, {len(df)} unique)"
        )

    if len(df) == 0:
        raise ContractError("no trips in data")

    dist = df["trip_distance"].to_numpy(dtype="float64")
    if not np.all((dist > 0) & (dist <= 100)):
        raise ContractError("trip_distance outside (0, 100]")
    fare = df["fare_amount"].to_numpy(dtype="float64")
    if not np.all((fare >= 0) & (fare <= 1000)):
        raise ContractError("fare_amount outside [0, 1000]")
    pay = df["payment_type"].to_numpy(dtype="float64")
    if not np.all(np.isin(pay, list(VALID_PAYMENT_CODES))):
        raise ContractError("unknown payment_type codes")
    pickup = df["tpep_pickup_datetime"]
    if not ((pickup >= start) & (pickup < end)).all():
        raise ContractError("pickup times outside reporting_period")

    n = int(len(df))
    total_fare = float(math.fsum(fare.tolist()))
    metrics = {
        "trip_count": n,
        "total_fare": total_fare,
        "mean_fare": total_fare / n,
        "mean_trip_distance": float(math.fsum(dist.tolist())) / n,
        "card_share": float(np.count_nonzero(pay == 1)) / n,
    }
    return {"status": "ok", "metrics": metrics, "reason": ""}


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _analyze(data_path, manifest_path)
    except ContractError as exc:
        return _abstain(str(exc))
    except Exception as exc:  # any unexpected failure means the result cannot be trusted
        return _abstain(f"could not process input: {type(exc).__name__}: {exc}")
