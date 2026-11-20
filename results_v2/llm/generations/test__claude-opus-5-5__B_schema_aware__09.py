"""Monthly summary of NYC yellow taxi trips with contract validation."""
from __future__ import annotations

import json
import math
import re
from decimal import Decimal

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

_TZ_SUFFIX = re.compile(r"(Z|[+-]\d{2}:?\d{2}|UTC|GMT)\s*$", re.IGNORECASE)


class Abstain(Exception):
    pass


def _abstain(reason: str) -> dict:
    return {"status": "abstain", "metrics": None, "reason": reason}


# --------------------------------------------------------------------------- #
# Manifest helpers
# --------------------------------------------------------------------------- #
def _to_local_naive_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts is pd.NaT or pd.isna(ts):
        raise Abstain("invalid reporting_period bound")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    return ts


def _reporting_period(manifest: dict):
    rp = manifest.get("reporting_period")
    if rp is None:
        raise Abstain("manifest missing reporting_period")
    start = end = None
    if isinstance(rp, dict):
        for k in ("start", "start_inclusive", "from", "begin", "start_date"):
            if k in rp:
                start = rp[k]
                break
        for k in ("end", "end_exclusive", "to", "until", "end_date", "stop"):
            if k in rp:
                end = rp[k]
                break
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        start, end = rp
    elif isinstance(rp, str) and re.fullmatch(r"\d{4}-\d{2}", rp.strip()):
        start = pd.Timestamp(rp.strip() + "-01")
        end = start + pd.offsets.MonthBegin(1)
    if start is None or end is None:
        raise Abstain("cannot interpret reporting_period")
    s, e = _to_local_naive_ts(start), _to_local_naive_ts(end)
    if not s < e:
        raise Abstain("empty or inverted reporting_period")
    return s, e


def _apply_renames(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    changes = manifest.get("schema_changes") or []
    if not isinstance(changes, list):
        raise Abstain("schema_changes is not a list")
    renames = []
    for ch in changes:
        if not isinstance(ch, dict):
            raise Abstain("malformed schema change entry")
        typ = str(ch.get("type", "")).lower()
        if typ == "rename":
            src, dst = ch.get("from"), ch.get("to")
            if not isinstance(src, str) or not isinstance(dst, str):
                raise Abstain("malformed rename entry")
            renames.append((src, dst))
        elif typ in ("add", "add_column", "reorder", "type_change", "retype", "cast"):
            continue
        else:
            raise Abstain(f"unsupported schema change type: {typ!r}")
    # apply in reverse so chains A->B, B->C resolve back to A
    for src, dst in reversed(renames):
        if dst in df.columns:
            if src in df.columns and src != dst:
                raise Abstain(f"both {src!r} and renamed {dst!r} present")
            df = df.rename(columns={dst: src})
        elif src in df.columns:
            continue  # already under original name
    return df


def _unit_factors(manifest: dict) -> dict:
    """Return multiplicative factors to convert into contract units."""
    factors = {"trip_distance": 1.0, "fare_amount": 1.0,
               "tip_amount": 1.0, "total_amount": 1.0}
    units = manifest.get("units")
    money_keys = ("fare_amount", "tip_amount", "total_amount")

    def money_factor(u: str) -> float:
        u = u.strip().lower()
        if u in ("usd", "dollar", "dollars", "$", "us dollars"):
            return 1.0
        if u in ("cents", "cent", "usd_cents", "usd cents"):
            return 0.01
        raise Abstain(f"unsupported currency unit {u!r}")

    def dist_factor(u: str) -> float:
        u = u.strip().lower()
        if u in ("mile", "miles", "mi"):
            return 1.0
        if u in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
            return 1.0 / 1.609344
        if u in ("m", "meter", "meters", "metre", "metres"):
            return 1.0 / 1609.344
        raise Abstain(f"unsupported distance unit {u!r}")

    if isinstance(units, dict):
        for col, u in units.items():
            if not isinstance(u, str):
                continue
            if col in money_keys:
                factors[col] = money_factor(u)
            elif col == "trip_distance":
                factors[col] = dist_factor(u)
            elif col in ("currency", "money", "amounts"):
                f = money_factor(u)
                for k in money_keys:
                    factors[k] = f
            elif col in ("distance",):
                factors["trip_distance"] = dist_factor(u)
    cur = manifest.get("currency")
    if isinstance(cur, str):
        f = money_factor(cur)
        for k in money_keys:
            factors[k] = f
    du = manifest.get("distance_unit")
    if isinstance(du, str):
        factors["trip_distance"] = dist_factor(du)
    return factors


# --------------------------------------------------------------------------- #
# Column coercion
# --------------------------------------------------------------------------- #
def _to_numeric(s: pd.Series, name: str) -> pd.Series:
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    if s.dtype == object:
        def conv(v):
            if v is None:
                return np.nan
            if isinstance(v, float) and math.isnan(v):
                return np.nan
            if isinstance(v, bool):
                return np.nan  # booleans are not a valid encoding
            if isinstance(v, Decimal):
                return float(v)
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8", "replace")
            if isinstance(v, str):
                t = v.strip().replace(",", "")
                if t == "" or t.lower() in ("nan", "null", "none", "na", "n/a"):
                    return np.nan
                try:
                    return float(t)
                except ValueError:
                    return "__BAD__"
            try:
                return float(v)
            except (TypeError, ValueError):
                return "__BAD__"
        conv_vals = s.map(conv)
        if (conv_vals.astype(object) == "__BAD__").any():
            raise Abstain(f"unparseable values in {name}")
        out = pd.to_numeric(conv_vals, errors="coerce").astype("float64")
    elif pd.api.types.is_bool_dtype(s.dtype):
        raise Abstain(f"boolean values in numeric column {name}")
    elif pd.api.types.is_numeric_dtype(s.dtype):
        out = pd.to_numeric(s, errors="coerce").astype("float64")
    else:
        raise Abstain(f"incompatible type for {name}: {s.dtype}")
    if np.isinf(out.to_numpy(dtype="float64", na_value=np.nan)).any():
        raise Abstain(f"infinite values in {name}")
    return out


def _to_int(s: pd.Series, name: str) -> pd.Series:
    f = _to_numeric(s, name)
    nn = f.dropna()
    if len(nn) and not np.all(np.equal(np.floor(nn.to_numpy()), nn.to_numpy())):
        raise Abstain(f"non-integral values in {name}")
    return f.astype("Int64") if len(f) == 0 or f.notna().all() is False else f.round().astype("Int64")


def _epoch_to_local(num: pd.Series) -> pd.Series:
    vals = num.dropna().abs()
    if len(vals) == 0:
        return pd.Series(pd.NaT, index=num.index, dtype="datetime64[ns]")
    m = vals.median()
    unit = "s" if m < 1e11 else "ms" if m < 1e14 else "us" if m < 1e17 else "ns"
    ts = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
    return ts.dt.tz_convert(NY_TZ).dt.tz_localize(None)


def _to_local_datetime(s: pd.Series, name: str) -> pd.Series:
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(NY_TZ).dt.tz_localize(None).astype("datetime64[ns]")
    if pd.api.types.is_datetime64_dtype(s.dtype):
        return s.astype("datetime64[ns]")
    if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
        return _epoch_to_local(s.astype("float64"))
    if s.dtype != object:
        raise Abstain(f"incompatible type for {name}: {s.dtype}")

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    nonnull = s[s.notna()]
    if len(nonnull) == 0:
        return out

    kinds = nonnull.map(lambda v: "str" if isinstance(v, (str, bytes)) else
                        "ts" if isinstance(v, (pd.Timestamp, np.datetime64)) or
                        hasattr(v, "year") else
                        "num" if isinstance(v, (int, float, Decimal, np.number))
                        and not isinstance(v, bool) else "bad")
    if (kinds == "bad").any():
        raise Abstain(f"unparseable values in {name}")

    # Python datetime / Timestamp objects (possibly tz-aware, possibly mixed)
    tsv = nonnull[kinds == "ts"]
    for idx, v in tsv.items():
        t = pd.Timestamp(v)
        if t.tzinfo is not None:
            t = t.tz_convert(NY_TZ).tz_localize(None)
        out.at[idx] = t

    numv = nonnull[kinds == "num"]
    if len(numv):
        out.loc[numv.index] = _epoch_to_local(numv.astype("float64"))

    strv = nonnull[kinds == "str"].map(
        lambda v: (v.decode("utf-8", "replace") if isinstance(v, bytes) else v).strip())
    strv = strv[~strv.str.lower().isin(["", "nan", "nat", "null", "none"])]
    if len(strv):
        aware_mask = strv.str.contains(_TZ_SUFFIX)
        # tz-aware strings
        aw = strv[aware_mask]
        if len(aw):
            parsed = pd.to_datetime(aw, utc=True, errors="coerce", format="mixed")
            if parsed.isna().any():
                raise Abstain(f"unparseable timestamps in {name}")
            out.loc[aw.index] = parsed.dt.tz_convert(NY_TZ).dt.tz_localize(None)
        nv = strv[~aware_mask]
        if len(nv):
            if nv.str.fullmatch(r"-?\d+(\.\d+)?").all():
                out.loc[nv.index] = _epoch_to_local(nv.astype("float64"))
            else:
                parsed = pd.to_datetime(nv, errors="coerce", format="mixed")
                if parsed.isna().any():
                    raise Abstain(f"unparseable timestamps in {name}")
                if isinstance(parsed.dtype, pd.DatetimeTZDtype):
                    parsed = parsed.dt.tz_convert(NY_TZ).dt.tz_localize(None)
                out.loc[nv.index] = parsed
    return out


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def _analyze(data_path: str, manifest_path: str) -> dict:
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        raise Abstain(f"cannot read manifest: {type(exc).__name__}")
    if not isinstance(manifest, dict):
        raise Abstain("manifest is not a JSON object")

    start, end = _reporting_period(manifest)
    factors = _unit_factors(manifest)

    try:
        table = pq.read_table(data_path)
    except Exception as exc:  # noqa: BLE001
        raise Abstain(f"cannot read parquet: {type(exc).__name__}")
    try:
        df = table.to_pandas(timestamp_as_object=False)
    except Exception:  # out-of-range timestamps etc.
        df = table.to_pandas(timestamp_as_object=True)

    if df.columns.duplicated().any():
        raise Abstain("duplicate column names in data")
    df = _apply_renames(df, manifest)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise Abstain(f"missing required columns: {missing}")

    present = [c for c in CONTRACT_COLUMNS if c in df.columns]
    norm = pd.DataFrame(index=df.index)
    for c in present:
        if c in DATETIME_COLS:
            try:
                norm[c] = _to_local_datetime(df[c], c)
            except Abstain:
                if c in REQUIRED:
                    raise
                norm[c] = df[c].astype(str)
        elif c in INT_COLS:
            try:
                f = _to_numeric(df[c], c)
                nn = f.dropna().to_numpy()
                if len(nn) and not np.all(np.floor(nn) == nn):
                    raise Abstain(f"non-integral values in {c}")
                norm[c] = f
            except Abstain:
                if c in REQUIRED:
                    raise
                norm[c] = df[c].astype(str)
        else:
            try:
                norm[c] = _to_numeric(df[c], c) * factors.get(c, 1.0)
            except Abstain:
                if c in REQUIRED:
                    raise
                norm[c] = df[c].astype(str)

    # Required not-null
    for c in REQUIRED:
        if norm[c].isna().any():
            raise Abstain(f"null values in required column {c}")

    if len(norm) == 0:
        raise Abstain("no trips in data")

    # Replays vs conflicts
    norm = norm.drop_duplicates(keep="first")
    if norm["record_id"].duplicated().any():
        raise Abstain("conflicting records share a record_id")

    # Domain checks
    pt = norm["payment_type"]
    if not pt.isin(list(VALID_PAYMENT)).all():
        raise Abstain("invalid payment_type codes")
    dist = norm["trip_distance"]
    if not ((dist > 0) & (dist <= 100)).all():
        raise Abstain("trip_distance outside valid range (0, 100]")
    fare = norm["fare_amount"]
    if not ((fare >= 0) & (fare <= 1000)).all():
        raise Abstain("fare_amount outside valid range [0, 1000]")

    pu = norm["tpep_pickup_datetime"]
    if not ((pu >= start) & (pu < end)).all():
        raise Abstain("pickup times outside reporting_period")

    # Optional manifest row-count consistency check
    for key in ("row_count", "record_count", "num_rows", "n_rows", "rows"):
        if key in manifest and isinstance(manifest[key], (int, float)) \
                and not isinstance(manifest[key], bool):
            expected = int(manifest[key])
            if expected not in (len(df), len(norm)):
                raise Abstain(f"row count mismatch with manifest {key}")
            break

    n = int(len(norm))
    if n == 0:
        raise Abstain("no trips after validation")

    fare_vals = norm["fare_amount"].to_numpy(dtype="float64")
    dist_vals = norm["trip_distance"].to_numpy(dtype="float64")
    total_fare = float(fare_vals.sum())
    metrics = {
        "trip_count": n,
        "total_fare": total_fare,
        "mean_fare": float(total_fare / n),
        "mean_trip_distance": float(dist_vals.sum() / n),
        "card_share": float((norm["payment_type"].to_numpy(dtype="float64") == 1).sum() / n),
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
    except Exception as exc:  # noqa: BLE001
        return _abstain(f"unexpected error: {type(exc).__name__}")
