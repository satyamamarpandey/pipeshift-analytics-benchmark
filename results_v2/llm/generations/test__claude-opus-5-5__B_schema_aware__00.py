import json
import math
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NY_TZ = "America/New_York"

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


class ContractError(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": str(reason)}


def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in ("", "nan", "none", "null", "nat", "na", "<na>"):
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _to_ny_naive_scalar(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    return ts


def _parse_datetime(series, name):
    s = series
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(NY_TZ).dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    missing = s.map(_is_missing)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        nonnull = vals[~missing]
        if nonnull.empty:
            return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        mx = float(np.nanmax(np.abs(nonnull.to_numpy(dtype="float64"))))
        if mx > 1e17:
            unit = "ns"
        elif mx > 1e14:
            unit = "us"
        elif mx > 1e11:
            unit = "ms"
        else:
            unit = "s"
        out = pd.to_datetime(vals, unit=unit, errors="coerce", utc=True)
        out = out.dt.tz_convert(NY_TZ).dt.tz_localize(None)
        if (out.isna() & ~missing).any():
            raise ContractError(f"unparseable values in {name}")
        return out
    # object / string
    result = [pd.NaT] * len(s)
    vals = s.tolist()
    for i, v in enumerate(vals):
        if missing.iloc[i]:
            continue
        try:
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8")
            if isinstance(v, str):
                v = v.strip()
            ts = pd.Timestamp(v)
            if pd.isna(ts):
                raise ValueError
            result[i] = _to_ny_naive_scalar(ts)
        except Exception:
            raise ContractError(f"unparseable values in {name}")
    return pd.Series(pd.to_datetime(pd.Series(result, dtype="object")), index=s.index).astype("datetime64[ns]")


def _parse_numeric(series, name):
    s = series
    if pd.api.types.is_bool_dtype(s):
        raise ContractError(f"boolean values in numeric column {name}")
    missing = s.map(_is_missing)
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce").astype("float64")
    else:
        def conv(v):
            if _is_missing(v):
                return np.nan
            if isinstance(v, bool):
                raise ContractError(f"boolean values in numeric column {name}")
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8")
            if isinstance(v, str):
                t = v.strip().replace(",", "")
                if t.startswith("$"):
                    t = t[1:]
                try:
                    return float(t)
                except ValueError:
                    raise ContractError(f"unparseable values in {name}")
            try:
                return float(v)
            except Exception:
                raise ContractError(f"unparseable values in {name}")
        out = pd.Series([conv(v) for v in s.tolist()], index=s.index, dtype="float64")
    if (out.isna() & ~missing).any():
        raise ContractError(f"unparseable values in {name}")
    if np.isinf(out.to_numpy(dtype="float64", na_value=np.nan)).any():
        raise ContractError(f"infinite values in {name}")
    return out


def _parse_period_bound(v):
    if v is None:
        raise ContractError("reporting_period bound missing")
    ts = pd.Timestamp(v)
    if pd.isna(ts):
        raise ContractError("invalid reporting_period bound")
    return _to_ny_naive_scalar(ts)


def _get_period(manifest):
    rp = manifest.get("reporting_period")
    if rp is None:
        for k in ("period", "reportingPeriod"):
            if k in manifest:
                rp = manifest[k]
                break
    if rp is None:
        raise ContractError("manifest has no reporting_period")
    if isinstance(rp, dict):
        start = rp.get("start", rp.get("from", rp.get("begin")))
        end = rp.get("end", rp.get("to", rp.get("until")))
        if start is None or end is None:
            raise ContractError("reporting_period missing start/end")
        start, end = _parse_period_bound(start), _parse_period_bound(end)
    elif isinstance(rp, (list, tuple)) and len(rp) == 2:
        start, end = _parse_period_bound(rp[0]), _parse_period_bound(rp[1])
    elif isinstance(rp, str):
        m = re.fullmatch(r"\s*(\d{4})-(\d{2})\s*", rp)
        if not m:
            raise ContractError("unrecognized reporting_period format")
        start = pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
        end = start + pd.offsets.MonthBegin(1)
    else:
        raise ContractError("unrecognized reporting_period format")
    if not end > start:
        raise ContractError("reporting_period end not after start")
    return start, end


def _apply_renames(df, manifest):
    changes = manifest.get("schema_changes") or []
    if not isinstance(changes, list):
        raise ContractError("schema_changes is not a list")
    for ch in changes:
        if not isinstance(ch, dict):
            raise ContractError("malformed schema_changes entry")
        if str(ch.get("type", "")).lower() != "rename":
            continue
        src, dst = ch.get("from"), ch.get("to")
        if not isinstance(src, str) or not isinstance(dst, str):
            raise ContractError("malformed rename entry")
        if dst in df.columns:
            if src in df.columns and src != dst:
                raise ContractError(f"both {src} and renamed {dst} present")
            df = df.rename(columns={dst: src})
    return df


def _unit_factor(manifest, col):
    units = manifest.get("units")
    unit = None
    if isinstance(units, dict):
        unit = units.get(col)
    if unit is None:
        unit = manifest.get(f"{col}_unit")
    if unit is None:
        cur = manifest.get("currency")
        if col in ("fare_amount", "tip_amount", "total_amount") and cur is not None:
            if str(cur).strip().upper() not in ("USD",):
                raise ContractError(f"unsupported currency {cur}")
        return 1.0
    u = str(unit).strip().lower()
    if col in ("fare_amount", "tip_amount", "total_amount"):
        if u in ("usd", "dollar", "dollars", "$"):
            return 1.0
        if u in ("cents", "cent", "usd_cents"):
            return 0.01
        raise ContractError(f"unsupported unit {unit} for {col}")
    if col == "trip_distance":
        if u in ("mi", "mile", "miles"):
            return 1.0
        if u in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
            return 1.0 / 1.609344
        raise ContractError(f"unsupported unit {unit} for {col}")
    return 1.0


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return _abstain(f"cannot read manifest: {type(e).__name__}")
        if not isinstance(manifest, dict):
            return _abstain("manifest is not a JSON object")

        try:
            table = pq.read_table(data_path)
            df = table.to_pandas()
        except Exception as e:
            return _abstain(f"cannot read data: {type(e).__name__}")

        if df.columns.duplicated().any():
            return _abstain("duplicate column names in data")

        df = _apply_renames(df, manifest)
        if df.columns.duplicated().any():
            return _abstain("duplicate column names after renames")

        missing_cols = [c for c in REQUIRED if c not in df.columns]
        if missing_cols:
            return _abstain(f"missing required columns: {missing_cols}")

        start, end = _get_period(manifest)
        raw_rows = len(df)

        present = [c for c in CONTRACT if c in df.columns]
        norm = pd.DataFrame(index=df.index)
        for c in present:
            if c in DATETIME_COLS:
                norm[c] = _parse_datetime(df[c], c)
            else:
                vals = _parse_numeric(df[c], c)
                if c in INT_COLS:
                    nn = vals.dropna()
                    if (nn != np.floor(nn)).any():
                        raise ContractError(f"non-integer values in {c}")
                norm[c] = vals

        for c in REQUIRED:
            if norm[c].isna().any():
                return _abstain(f"null values in required column {c}")

        for c in FLOAT_COLS:
            if c in norm.columns:
                f = _unit_factor(manifest, c)
                if f != 1.0:
                    norm[c] = norm[c] * f

        # Replays vs conflicts
        norm = norm.drop_duplicates(subset=present, keep="first")
        if norm["record_id"].duplicated().any():
            return _abstain("conflicting records share a record_id")

        # Optional manifest row count check
        expected = None
        for k in ("row_count", "record_count", "expected_row_count", "num_rows", "rows"):
            if k in manifest and isinstance(manifest[k], (int, float)) and not isinstance(manifest[k], bool):
                expected = int(manifest[k])
                break
        if expected is not None and expected not in (raw_rows, len(norm)):
            return _abstain("row count does not match manifest")

        if len(norm) == 0:
            return _abstain("no trips in data")

        dist = norm["trip_distance"]
        if ((dist <= 0) | (dist > 100)).any():
            return _abstain("trip_distance outside valid range")
        fare = norm["fare_amount"]
        if ((fare < 0) | (fare > 1000)).any():
            return _abstain("fare_amount outside valid range")
        pt = norm["payment_type"]
        if not pt.isin(list(VALID_PAYMENT)).all():
            return _abstain("unknown payment_type codes")
        pu = norm["tpep_pickup_datetime"]
        if ((pu < start) | (pu >= end)).any():
            return _abstain("pickup times outside reporting_period")

        n = int(len(norm))
        total_fare = float(fare.sum())
        metrics = {
            "trip_count": n,
            "total_fare": total_fare,
            "mean_fare": total_fare / n,
            "mean_trip_distance": float(dist.sum()) / n,
            "card_share": float((pt == 1).sum()) / n,
        }
        for k, v in metrics.items():
            if isinstance(v, float) and not math.isfinite(v):
                return _abstain(f"non-finite metric {k}")
        return {"status": "ok", "metrics": metrics, "reason": ""}
    except ContractError as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain(f"unexpected error: {type(e).__name__}")
