"""Monthly summary of NYC yellow taxi trips with strict contract verification."""

import hashlib
import json
import math
import os
from datetime import date, datetime

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

DATA_TZ = "America/New_York"

CONTRACT_COLUMNS = [
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]
REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance",
            "payment_type", "fare_amount"]
MANIFEST_FIELDS = ["dataset", "snapshot_id", "snapshot_version", "reporting_period",
                   "file", "file_sha256", "row_count", "columns", "units",
                   "schema_changes", "description"]
MONEY_COLS = ["fare_amount", "tip_amount", "total_amount"]
INT_COLS = ["record_id", "VendorID", "PULocationID", "DOLocationID", "payment_type"]
VALID_PAYMENT = {0, 1, 2, 3, 4, 5, 6}


class Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


# --------------------------------------------------------------------------- helpers
def _is_null(v):
    if v is None:
        return True
    if v is pd.NaT:
        return True
    try:
        if isinstance(v, float) and math.isnan(v):
            return True
    except Exception:
        pass
    try:
        r = pd.isna(v)
        if isinstance(r, (bool, np.bool_)):
            return bool(r)
    except Exception:
        pass
    return False


def _to_numeric(s, col):
    """Coerce a series to float; abstain if any non-null value is unparseable."""
    if pd.api.types.is_bool_dtype(s):
        raise Abstain(f"column {col} has boolean type")
    if pd.api.types.is_numeric_dtype(s):
        return pd.Series(s.astype("float64"), index=s.index)
    vals = []
    for v in s.tolist():
        if _is_null(v):
            vals.append(np.nan)
            continue
        if isinstance(v, (bool, np.bool_)):
            raise Abstain(f"column {col} has boolean value")
        if isinstance(v, (int, float, np.integer, np.floating)):
            vals.append(float(v))
            continue
        if hasattr(v, "__float__") and not isinstance(v, (str, bytes)):
            try:
                vals.append(float(v))
                continue
            except Exception:
                raise Abstain(f"column {col} has unparseable value")
        if isinstance(v, bytes):
            try:
                v = v.decode("utf-8")
            except Exception:
                raise Abstain(f"column {col} has unparseable value")
        if isinstance(v, str):
            t = v.strip()
            if t == "" or t.lower() in ("nan", "null", "none", "na", "nat"):
                vals.append(np.nan)
                continue
            try:
                vals.append(float(t))
                continue
            except Exception:
                raise Abstain(f"column {col} has non-numeric value {v!r}")
        raise Abstain(f"column {col} has unsupported value type")
    return pd.Series(np.array(vals, dtype="float64"), index=s.index)


def _to_integer_valued(s, col):
    f = _to_numeric(s, col)
    nonnull = f.dropna()
    if len(nonnull) and not np.all(np.isfinite(nonnull.values)):
        raise Abstain(f"column {col} has non-finite values")
    if len(nonnull) and not np.all(nonnull.values == np.round(nonnull.values)):
        raise Abstain(f"column {col} has non-integer values")
    return f


def _check_tz(name):
    if ZoneInfo is None:
        return
    try:
        ZoneInfo(str(name))
    except Exception:
        raise Abstain(f"unknown timezone {name!r}")


def _ts_to_local_naive(ts, assume_tz=DATA_TZ):
    """Convert a pandas Timestamp to naive local New York time."""
    if ts is pd.NaT or ts is None:
        return pd.NaT
    if ts.tzinfo is not None:
        return ts.tz_convert(DATA_TZ).tz_localize(None)
    if assume_tz != DATA_TZ:
        return ts.tz_localize(assume_tz, ambiguous="raise",
                              nonexistent="raise").tz_convert(DATA_TZ).tz_localize(None)
    return ts


def _epoch_unit(maxabs):
    if maxabs < 1e11:
        return "s"
    if maxabs < 1e14:
        return "ms"
    if maxabs < 1e17:
        return "us"
    return "ns"


def _to_datetime_local(s, col):
    """Return series of naive datetime64[ns] in New York local time."""
    try:
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            out = s.dt.tz_convert(DATA_TZ).dt.tz_localize(None)
            return out.astype("datetime64[ns]")
        if pd.api.types.is_datetime64_dtype(s):
            return s.astype("datetime64[ns]")
        if pd.api.types.is_bool_dtype(s):
            raise Abstain(f"column {col} has boolean type")
        if pd.api.types.is_numeric_dtype(s):
            f = s.astype("float64")
            nn = f.dropna()
            if len(nn) == 0:
                return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
            if not np.all(np.isfinite(nn.values)):
                raise Abstain(f"column {col} has non-finite timestamps")
            unit = _epoch_unit(float(np.max(np.abs(nn.values))))
            out = pd.to_datetime(f, unit=unit, utc=True)
            return out.dt.tz_convert(DATA_TZ).dt.tz_localize(None).astype("datetime64[ns]")
    except Abstain:
        raise
    except Exception:
        raise Abstain(f"column {col} could not be parsed as timestamps")

    vals = []
    for v in s.tolist():
        if _is_null(v):
            vals.append(pd.NaT)
            continue
        try:
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            if isinstance(v, str):
                t = v.strip()
                if t == "" or t.lower() in ("nan", "nat", "null", "none"):
                    vals.append(pd.NaT)
                    continue
                ts = pd.Timestamp(t)
            elif isinstance(v, (pd.Timestamp, datetime, date, np.datetime64)):
                ts = pd.Timestamp(v)
            else:
                raise ValueError("unsupported")
            if ts is pd.NaT:
                vals.append(pd.NaT)
                continue
            vals.append(_ts_to_local_naive(ts))
        except Exception:
            raise Abstain(f"column {col} has unparseable timestamp {v!r}")
    try:
        return pd.Series(pd.to_datetime(pd.Series(vals, dtype="object")).values,
                         index=s.index).astype("datetime64[ns]")
    except Exception:
        raise Abstain(f"column {col} could not be parsed as timestamps")


def _parse_period_bound(v, tz):
    if not isinstance(v, str) or not v.strip():
        raise Abstain("reporting_period bound is not a non-empty string")
    try:
        ts = pd.Timestamp(v.strip())
    except Exception:
        raise Abstain(f"unparseable reporting_period bound {v!r}")
    if ts is pd.NaT:
        raise Abstain("reporting_period bound is NaT")
    try:
        return _ts_to_local_naive(ts, assume_tz=tz)
    except Exception:
        raise Abstain("reporting_period bound could not be localized")


def _unit_factor(unit, kind, col):
    if unit is None:
        return 1.0
    if isinstance(unit, dict):
        unit = unit.get("unit", unit.get("units"))
        if unit is None:
            return 1.0
    if not isinstance(unit, str):
        raise Abstain(f"unrecognized unit for {col}")
    u = unit.strip().lower().replace(".", "")
    if kind == "distance":
        if u in ("mi", "mile", "miles", "statute miles"):
            return 1.0
        if u in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
            return 1.0 / 1.609344
        if u in ("m", "meter", "meters", "metre", "metres"):
            return 1.0 / 1609.344
        raise Abstain(f"unrecognized distance unit {unit!r} for {col}")
    if u in ("usd", "dollar", "dollars", "us dollars", "us dollar", "$", "usd ($)"):
        return 1.0
    if u in ("cents", "cent", "usd cents", "us cents", "usd_cents", "¢"):
        return 0.01
    raise Abstain(f"unrecognized currency unit {unit!r} for {col}")


# --------------------------------------------------------------------------- main
def _run(data_path, manifest_path):
    # 1. Manifest
    if not isinstance(manifest_path, str) or not os.path.isfile(manifest_path):
        raise Abstain("manifest missing")
    try:
        with open(manifest_path, "r", encoding="utf-8-sig") as fh:
            manifest = json.load(fh)
    except Exception:
        raise Abstain("manifest is not valid JSON")
    if not isinstance(manifest, dict):
        raise Abstain("manifest is not a JSON object")
    missing = [f for f in MANIFEST_FIELDS if f not in manifest]
    if missing:
        raise Abstain(f"manifest missing fields: {missing}")

    rp = manifest["reporting_period"]
    if not isinstance(rp, dict) or not all(k in rp for k in ("start", "end", "timezone")):
        raise Abstain("reporting_period malformed")
    tz = rp["timezone"]
    if not isinstance(tz, str) or not tz.strip():
        raise Abstain("reporting_period timezone invalid")
    tz = tz.strip()
    _check_tz(tz)

    sha = manifest["file_sha256"]
    if not isinstance(sha, str) or len(sha.strip()) != 64:
        raise Abstain("file_sha256 malformed")
    try:
        int(sha.strip(), 16)
    except Exception:
        raise Abstain("file_sha256 malformed")

    row_count = manifest["row_count"]
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise Abstain("row_count malformed")

    columns = manifest["columns"]
    if not isinstance(columns, list) or not columns:
        raise Abstain("columns malformed")
    man_names = []
    for c in columns:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or "type" not in c:
            raise Abstain("columns entry malformed")
        man_names.append(c["name"])
    if len(set(man_names)) != len(man_names):
        raise Abstain("manifest columns contain duplicates")

    changes = manifest["schema_changes"]
    if changes is None:
        changes = []
    if not isinstance(changes, list):
        raise Abstain("schema_changes malformed")
    renames = {}
    for ch in changes:
        if not isinstance(ch, dict):
            raise Abstain("schema_changes entry malformed")
        if ch.get("type") == "rename":
            fr, to = ch.get("from"), ch.get("to")
            if not isinstance(fr, str) or not isinstance(to, str):
                raise Abstain("rename entry malformed")
            if fr in renames and renames[fr] != to:
                raise Abstain("conflicting rename entries")
            renames[fr] = to
    if len(set(renames.values())) != len(renames):
        raise Abstain("multiple columns renamed to same name")

    units = manifest["units"]
    if units is not None and not isinstance(units, (dict, str)):
        raise Abstain("units malformed")

    # 2. Checksum
    if not isinstance(data_path, str) or not os.path.isfile(data_path):
        raise Abstain("data file missing")
    h = hashlib.sha256()
    try:
        with open(data_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except Exception:
        raise Abstain("data file unreadable")
    if h.hexdigest().lower() != sha.strip().lower():
        raise Abstain("file checksum mismatch")

    # 3. Read and row count
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(data_path)
    except Exception:
        raise Abstain("data file is not readable Parquet")
    if table.num_rows != row_count:
        raise Abstain(f"row count {table.num_rows} != manifest {row_count}")
    delivered = list(table.column_names)
    if len(set(delivered)) != len(delivered):
        raise Abstain("duplicate column names in data")
    try:
        df = table.to_pandas()
    except Exception:
        try:
            df = table.to_pandas(timestamp_as_object=True)
        except Exception:
            raise Abstain("data could not be converted")

    # 4. Column agreement
    dset = set(delivered)
    inverse = {to: fr for fr, to in renames.items()}
    rename_map = {}
    for name in man_names:
        if name in renames:
            orig, new = name, renames[name]
        elif name in inverse:
            orig, new = inverse[name], name
        else:
            orig, new = name, name
        if orig != new:
            has_new, has_orig = new in dset, orig in dset
            if has_new and has_orig:
                raise Abstain(f"both {orig} and renamed {new} present")
            if has_new:
                rename_map[new] = orig
            elif not has_orig:
                raise Abstain(f"declared column {orig}/{new} missing")
        elif name not in dset:
            raise Abstain(f"declared column {name} missing from data")
    # renames of contract columns not listed in manifest columns
    for fr, to in renames.items():
        if to in dset and to not in rename_map and fr not in dset:
            rename_map[to] = fr
    df = df.rename(columns=rename_map)
    if len(set(df.columns)) != len(df.columns):
        raise Abstain("column name collision after renames")
    for c in REQUIRED:
        if c not in df.columns:
            raise Abstain(f"required column {c} missing")

    present = [c for c in CONTRACT_COLUMNS if c in df.columns]
    df = df[present].copy()

    # Unit lookup helper
    def unit_for(col):
        if not isinstance(units, dict):
            return None
        for key in (col, renames.get(col)):
            if key is not None and key in units:
                return units[key]
        return None

    # Normalize types
    norm = pd.DataFrame(index=df.index)
    for c in present:
        if c in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
            if c == "tpep_pickup_datetime":
                norm[c] = _to_datetime_local(df[c], c)
            else:
                try:
                    norm[c] = _to_datetime_local(df[c], c)
                except Abstain:
                    norm[c] = df[c].astype(str)
        elif c in INT_COLS:
            if c in ("record_id", "payment_type"):
                norm[c] = _to_integer_valued(df[c], c)
            else:
                try:
                    norm[c] = _to_numeric(df[c], c)
                except Abstain:
                    norm[c] = df[c].astype(str)
        elif c == "trip_distance":
            norm[c] = _to_numeric(df[c], c) * _unit_factor(unit_for(c), "distance", c)
        elif c in MONEY_COLS:
            if c == "fare_amount":
                norm[c] = _to_numeric(df[c], c) * _unit_factor(unit_for(c), "money", c)
            else:
                try:
                    norm[c] = _to_numeric(df[c], c) * _unit_factor(unit_for(c), "money", c)
                except Abstain:
                    norm[c] = df[c].astype(str)
    if isinstance(units, str):
        # A single unit string: only accept if it names standard units
        u = units.strip().lower()
        if any(k in u for k in ("km", "kilomet", "cent", "meter", "metre", "eur")):
            raise Abstain(f"unsupported global units {units!r}")

    # 6a. Required non-null
    for c in REQUIRED:
        if norm[c].isna().any():
            raise Abstain(f"null values in required column {c}")

    # 5. Reporting period
    start = _parse_period_bound(rp["start"], tz)
    end = _parse_period_bound(rp["end"], tz)
    if not start < end:
        raise Abstain("reporting_period start not before end")
    pu = norm["tpep_pickup_datetime"]
    if ((pu < start) | (pu >= end)).any():
        raise Abstain("pickup timestamps outside reporting_period")

    # 6b. Codes and ranges
    pt = norm["payment_type"]
    if not pt.isin(list(VALID_PAYMENT)).all():
        raise Abstain("undocumented payment_type codes")
    td = norm["trip_distance"]
    if not (np.isfinite(td.values).all() and ((td > 0) & (td <= 100)).all()):
        raise Abstain("trip_distance out of valid range")
    fa = norm["fare_amount"]
    if not (np.isfinite(fa.values).all() and ((fa >= 0) & (fa <= 1000)).all()):
        raise Abstain("fare_amount out of valid range")

    # 6c. Replays and conflicts
    dedup = norm.drop_duplicates(keep="first")
    if dedup["record_id"].duplicated().any():
        raise Abstain("conflicting records share a record_id")

    n = int(len(dedup))
    if n == 0:
        raise Abstain("no trips to summarize")

    fares = dedup["fare_amount"].astype("float64")
    total_fare = float(fares.sum())
    return {
        "status": "ok",
        "metrics": {
            "trip_count": n,
            "total_fare": total_fare,
            "mean_fare": float(total_fare / n),
            "mean_trip_distance": float(dedup["trip_distance"].astype("float64").mean()),
            "card_share": float((dedup["payment_type"] == 1).sum() / n),
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _run(data_path, manifest_path)
    except Abstain as e:
        return _abstain(str(e) or "contract violation")
    except Exception as e:  # any unexpected failure => untrustworthy
        return _abstain(f"unexpected error: {type(e).__name__}")
