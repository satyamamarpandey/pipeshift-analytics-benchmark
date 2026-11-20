import hashlib
import json
import math

import pandas as pd
import pyarrow.parquet as pq

_REQUIRED_MANIFEST = ["dataset", "snapshot_id", "snapshot_version", "reporting_period",
                      "file", "file_sha256", "row_count", "columns", "units",
                      "schema_changes", "description"]
_CONTRACT = ["record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
             "PULocationID", "DOLocationID", "trip_distance", "payment_type",
             "fare_amount", "tip_amount", "total_amount"]
_REQ = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
_NY = "America/New_York"


class _Abstain(Exception):
    pass


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _to_numeric(s, name):
    try:
        out = pd.to_numeric(s, errors="raise")
    except Exception:
        raise _Abstain(f"column {name} has non-numeric values")
    out = out.astype("float64")
    if out.isna().any():
        raise _Abstain(f"nulls or NaN in required column {name}")
    if not (out.abs() != float("inf")).all():
        raise _Abstain(f"infinite values in {name}")
    return out


def _to_local(s, tz):
    """Parse to naive local timestamps."""
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(tz).dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        r = pd.to_datetime(s, errors="raise")
        if isinstance(r.dtype, pd.DatetimeTZDtype):
            return r.dt.tz_convert(tz).dt.tz_localize(None)
        return r
    except Exception:
        pass
    try:
        r = pd.to_datetime(s, errors="raise", utc=True)
        return r.dt.tz_convert(tz).dt.tz_localize(None)
    except Exception:
        raise _Abstain("unparseable pickup timestamps")


def _bound(v, tz):
    t = pd.Timestamp(v)
    if t.tzinfo is not None:
        t = t.tz_convert(tz).tz_localize(None)
    return t


def _analyze(data_path, manifest_path):
    # 1. manifest
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)
    except Exception:
        raise _Abstain("manifest missing or invalid JSON")
    if not isinstance(man, dict):
        raise _Abstain("manifest is not a JSON object")
    for k in _REQUIRED_MANIFEST:
        if k not in man:
            raise _Abstain(f"manifest missing field {k}")
    rp = man["reporting_period"]
    if not isinstance(rp, dict) or "start" not in rp or "end" not in rp:
        raise _Abstain("manifest reporting_period invalid")
    tz = rp.get("timezone") or _NY
    if tz != _NY:
        raise _Abstain("unexpected reporting timezone")
    try:
        start = _bound(rp["start"], tz)
        end = _bound(rp["end"], tz)
    except Exception:
        raise _Abstain("unparseable reporting period")
    if not start < end:
        raise _Abstain("empty reporting period")
    if not isinstance(man["columns"], list) or not isinstance(man["schema_changes"], list):
        raise _Abstain("manifest columns/schema_changes invalid")

    # units
    try:
        u = json.dumps(man["units"]).lower()
    except Exception:
        u = ""
    for bad in ("kilomet", "km", "cent", "meter", "metre"):
        if bad in u.replace("kilometer", "kilomet") and not (bad == "meter" and "kilomet" in u) \
                and not (bad == "metre" and "kilomet" in u):
            if bad in ("km", "kilomet", "cent") or bad in ("meter", "metre"):
                raise _Abstain("units differ from contract (miles/USD)")

    # 2. hash
    try:
        h = hashlib.sha256()
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except Exception:
        raise _Abstain("data file unreadable")
    if h.hexdigest().lower() != str(man["file_sha256"]).strip().lower():
        raise _Abstain("SHA-256 mismatch")

    # 3. row count
    try:
        pf = pq.ParquetFile(data_path)
        nrows = pf.metadata.num_rows
        df = pf.read().to_pandas()
    except Exception:
        raise _Abstain("data file is not readable Parquet")
    try:
        rc = int(man["row_count"])
    except Exception:
        raise _Abstain("invalid row_count")
    if nrows != rc or len(df) != rc:
        raise _Abstain("row count mismatch")

    # 4. columns
    canon = {}
    for ch in man["schema_changes"]:
        if not isinstance(ch, dict):
            raise _Abstain("invalid schema change")
        if ch.get("type") == "rename":
            a, b = ch.get("from"), ch.get("to")
            if a is None or b is None:
                raise _Abstain("invalid rename")
            if b in _CONTRACT:
                canon[a] = b
            elif a in _CONTRACT:
                canon[b] = a
            else:
                canon[a] = b
                canon[b] = b
        else:
            raise _Abstain("unsupported schema change")
    delivered = list(df.columns)
    if len(set(delivered)) != len(delivered):
        raise _Abstain("duplicate columns")
    mapped = [canon.get(c, c) for c in delivered]
    if len(set(mapped)) != len(mapped):
        raise _Abstain("column name collision after renames")
    mset = set(mapped)
    try:
        mcols = [c["name"] for c in man["columns"]]
    except Exception:
        raise _Abstain("manifest columns malformed")
    for c in mcols:
        if canon.get(c, c) not in mset:
            raise _Abstain(f"manifest column {c} not delivered")
    for c in _REQ:
        if c not in mset:
            raise _Abstain(f"required column {c} missing")
    df.columns = mapped

    if len(df) == 0:
        raise _Abstain("no rows")

    # 6a. parse and validate required
    rid = df["record_id"]
    if rid.isna().any():
        raise _Abstain("null record_id")
    rid = _to_numeric(rid, "record_id")
    if not (rid == rid.round()).all():
        raise _Abstain("non-integer record_id")
    pu = df["tpep_pickup_datetime"]
    if pu.isna().any():
        raise _Abstain("null pickup timestamp")
    pu = _to_local(pu, tz)
    if pu.isna().any():
        raise _Abstain("null pickup timestamp")
    dist = _to_numeric(df["trip_distance"], "trip_distance")
    pay = _to_numeric(df["payment_type"], "payment_type")
    fare = _to_numeric(df["fare_amount"], "fare_amount")

    norm = pd.DataFrame({
        "record_id": rid.astype("int64"),
        "pickup": pu,
        "dist": dist,
        "pay": pay,
        "fare": fare,
    })
    for c in _CONTRACT:
        if c in df.columns and c not in _REQ:
            norm["x_" + c] = df[c].astype(str)

    # dedupe / conflicts
    norm = norm.drop_duplicates()
    if norm["record_id"].duplicated().any():
        raise _Abstain("conflicting duplicate record_id")

    # 5. period
    if ((norm["pickup"] < start) | (norm["pickup"] >= end)).any():
        raise _Abstain("pickup outside reporting period")

    # 6b. ranges and codes
    if not (norm["pay"] == norm["pay"].round()).all() or not norm["pay"].between(0, 6).all():
        raise _Abstain("undocumented payment_type code")
    if not ((norm["dist"] > 0) & (norm["dist"] <= 100)).all():
        raise _Abstain("trip_distance out of range")
    if not norm["fare"].between(0, 1000).all():
        raise _Abstain("fare_amount out of range")

    n = len(norm)
    total = math.fsum(norm["fare"].tolist())
    metrics = {
        "trip_count": int(n),
        "total_fare": float(total),
        "mean_fare": float(total / n),
        "mean_trip_distance": float(math.fsum(norm["dist"].tolist()) / n),
        "card_share": float((norm["pay"] == 1).sum() / n),
    }
    return {"status": "ok", "metrics": metrics, "reason": ""}


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except _Abstain as e:
        return _abstain(str(e))
    except Exception as e:
        return _abstain(f"unexpected error: {type(e).__name__}")
