import json
import math
import re
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_TZ = "America/New_York"
_CONTRACT = (
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
)
_REQUIRED = ("record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount")
_INT_COLS = frozenset({"record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"})
_TS_COLS = frozenset({"tpep_pickup_datetime", "tpep_dropoff_datetime"})
_VALID_PAYMENT = [0, 1, 2, 3, 4, 5, 6]
_TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _arrow_dtype(arrow_type):
    if arrow_type == pa.int64():
        return pd.Int64Dtype()
    if arrow_type == pa.int32():
        return pd.Int32Dtype()
    return None


def _scalar_ts(value, label):
    if not isinstance(value, str):
        raise _Abstain(f"manifest {label} is not a string timestamp")
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise _Abstain(f"manifest {label} is not a valid timestamp") from exc
    if pd.isna(ts):
        raise _Abstain(f"manifest {label} is not a valid timestamp")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_TZ).tz_localize(None)
    return ts


def _load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise _Abstain("manifest is not a JSON object")
    period = manifest.get("reporting_period")
    if not isinstance(period, dict) or "start" not in period or "end" not in period:
        raise _Abstain("manifest lacks reporting_period start/end")
    start = _scalar_ts(period["start"], "reporting_period.start")
    end = _scalar_ts(period["end"], "reporting_period.end")
    if start >= end:
        raise _Abstain("manifest reporting_period is empty or inverted")
    changes = manifest.get("schema_changes", [])
    if changes is None:
        changes = []
    if not isinstance(changes, list):
        raise _Abstain("manifest schema_changes is not a list")
    return start, end, changes


def _apply_renames(df, changes):
    # Each declaration: file column named "to" holds contract column "from".
    mapping = {}
    for change in changes:
        if not isinstance(change, dict) or change.get("type") != "rename":
            raise _Abstain("unsupported or malformed schema change")
        frm, to = change.get("from"), change.get("to")
        if not isinstance(frm, str) or not isinstance(to, str) or frm not in _CONTRACT:
            raise _Abstain("invalid rename declaration")
        if to in mapping or frm in mapping.values():
            raise _Abstain("ambiguous rename declarations")
        mapping[to] = frm
    present = set(df.columns)
    rename_map = {}
    for to, frm in mapping.items():
        if to in present and to != frm:
            if frm in present:
                raise _Abstain(f"both '{to}' and '{frm}' present for a declared rename")
            rename_map[to] = frm
    return df.rename(columns=rename_map)


def _numeric(s, name, integral):
    s = s.reset_index(drop=True)
    dtype = s.dtype
    if pd.api.types.is_bool_dtype(dtype):
        raise _Abstain(f"{name}: boolean values are not allowed")
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        cleaned = []
        for v in s.astype(object):
            if isinstance(v, str):
                txt = v.strip()
                cleaned.append(None if txt == "" else txt)
            elif v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and math.isnan(v)):
                cleaned.append(None)
            elif isinstance(v, (bool, np.bool_)):
                raise _Abstain(f"{name}: boolean values are not allowed")
            elif isinstance(v, (int, float, np.integer, np.floating)):
                cleaned.append(v)
            else:
                raise _Abstain(f"{name}: unsupported value type")
        try:
            values = pd.to_numeric(pd.Series(cleaned, dtype=object), errors="raise")
        except (TypeError, ValueError) as exc:
            raise _Abstain(f"{name}: non-numeric values") from exc
        if not pd.api.types.is_numeric_dtype(values.dtype):
            values = values.astype("float64")
    elif pd.api.types.is_numeric_dtype(dtype):
        values = s
    else:
        raise _Abstain(f"{name}: unsupported column type {dtype}")

    if pd.api.types.is_complex_dtype(values.dtype):
        raise _Abstain(f"{name}: complex values are not allowed")

    if integral:
        if pd.api.types.is_integer_dtype(values.dtype):
            return values.astype("Int64")
        f = values.to_numpy(dtype="float64", na_value=np.nan)
        present = f[~np.isnan(f)]
        if (not np.isfinite(present).all()
                or not np.array_equal(present, np.floor(present))
                or (np.abs(present) >= 2.0 ** 63).any()):
            raise _Abstain(f"{name}: non-integer values")
        return pd.Series(f, dtype="float64").astype("Int64")

    f = values.to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(f).any():
        raise _Abstain(f"{name}: infinite values")
    return pd.Series(f, dtype="float64")


def _timestamps(s, name):
    s = s.reset_index(drop=True)
    dtype = s.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(_TZ).dt.tz_localize(None).astype("datetime64[ns]")
    if pd.api.types.is_datetime64_dtype(dtype):
        return s.astype("datetime64[ns]")
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        texts = []
        for v in s.astype(object):
            if isinstance(v, str):
                txt = v.strip()
                texts.append(None if txt == "" else txt)
            elif v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and math.isnan(v)):
                texts.append(None)
            else:
                raise _Abstain(f"{name}: unsupported timestamp value type")
        aware_idx = [i for i, t in enumerate(texts) if t is not None and _TZ_SUFFIX.search(t)]
        aware_set = set(aware_idx)
        naive_idx = [i for i, t in enumerate(texts) if t is not None and i not in aware_set]
        out = np.full(len(texts), np.datetime64("NaT"), dtype="datetime64[ns]")
        try:
            if aware_idx:
                parsed = pd.to_datetime(
                    pd.Series([texts[i] for i in aware_idx], dtype=object),
                    format="ISO8601", utc=True,
                )
                converted = parsed.dt.tz_convert(_TZ).dt.tz_localize(None)
                out[aware_idx] = converted.to_numpy(dtype="datetime64[ns]")
            if naive_idx:
                parsed = pd.to_datetime(
                    pd.Series([texts[i] for i in naive_idx], dtype=object),
                    format="ISO8601",
                )
                out[naive_idx] = parsed.to_numpy(dtype="datetime64[ns]")
        except (ValueError, TypeError, OverflowError) as exc:
            raise _Abstain(f"{name}: unparseable timestamps") from exc
        return pd.Series(out, dtype="datetime64[ns]")
    raise _Abstain(f"{name}: unsupported timestamp type {dtype}")


def _canonicalize(df):
    columns = {}
    for c in _CONTRACT:
        s = df[c]
        if c in _TS_COLS:
            columns[c] = _timestamps(s, c)
        else:
            columns[c] = _numeric(s, c, integral=(c in _INT_COLS))
    return pd.DataFrame(columns)


def _validate(t):
    for c in _REQUIRED:
        if bool(t[c].isna().any()):
            raise _Abstain(f"required column '{c}' contains nulls")
    dist = t["trip_distance"].to_numpy(dtype="float64")
    if not bool(((dist > 0) & (dist <= 100)).all()):
        raise _Abstain("trip_distance outside 0 < x <= 100")
    fare = t["fare_amount"].to_numpy(dtype="float64")
    if not bool(((fare >= 0) & (fare <= 1000)).all()):
        raise _Abstain("fare_amount outside 0 <= x <= 1000")
    if not bool(t["payment_type"].isin(_VALID_PAYMENT).all()):
        raise _Abstain("payment_type contains codes outside 0..6")


def _analyze(data_path, manifest_path):
    start, end, changes = _load_manifest(manifest_path)

    table = pq.read_table(data_path)
    df = table.to_pandas(types_mapper=_arrow_dtype)
    df = _apply_renames(df, changes)

    if df.columns.duplicated().any():
        raise _Abstain("duplicate column names in data")
    missing = [c for c in _CONTRACT if c not in df.columns]
    if missing:
        raise _Abstain(f"missing contract columns: {', '.join(missing)}")

    t = _canonicalize(df)
    _validate(t)

    t = t.drop_duplicates(subset=list(_CONTRACT), keep="first")
    if bool(t["record_id"].duplicated().any()):
        raise _Abstain("conflicting records share a record_id")

    pickup = t["tpep_pickup_datetime"]
    sel = t[(pickup >= start) & (pickup < end)]
    n = int(len(sel))
    if n == 0:
        raise _Abstain("no trips within the reporting period")

    fares = sel["fare_amount"].to_numpy(dtype="float64").tolist()
    dists = sel["trip_distance"].to_numpy(dtype="float64").tolist()
    total_fare = math.fsum(fares)
    card_trips = int((sel["payment_type"] == 1).sum())

    metrics = {
        "trip_count": n,
        "total_fare": float(total_fare),
        "mean_fare": float(total_fare / n),
        "mean_trip_distance": float(math.fsum(dists) / n),
        "card_share": float(card_trips / n),
    }
    return {"status": "ok", "metrics": metrics, "reason": ""}


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _analyze(data_path, manifest_path)
    except _Abstain as exc:
        return _abstain(str(exc))
    except Exception as exc:
        return _abstain(f"could not produce a trustworthy result ({type(exc).__name__})")
