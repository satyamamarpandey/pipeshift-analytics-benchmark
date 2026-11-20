import hashlib
import json
import math
import os


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": str(reason)}


REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
OPTIONAL = ["VendorID", "tpep_dropoff_datetime", "PULocationID", "DOLocationID",
            "tip_amount", "total_amount"]
MANIFEST_FIELDS = ["dataset", "snapshot_id", "snapshot_version", "reporting_period", "file",
                   "file_sha256", "row_count", "columns", "units", "schema_changes", "description"]


def _to_numeric(series, name):
    import pandas as pd
    import numpy as np
    if series.dtype == bool:
        raise ValueError(f"{name} is boolean")
    if pd.api.types.is_numeric_dtype(series):
        out = pd.to_numeric(series, errors="raise")
    else:
        s = series.map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.replace("", None)
        out = pd.to_numeric(s, errors="raise")
    out = out.astype("float64")
    if np.isinf(out.to_numpy()).any():
        raise ValueError(f"{name} contains infinite values")
    return out


def _to_local_naive(series, tzname):
    import pandas as pd
    from zoneinfo import ZoneInfo
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError("pickup timestamps delivered as bare numbers (ambiguous unit)")
    if not pd.api.types.is_datetime64_any_dtype(series):
        try:
            s = pd.to_datetime(series, errors="raise")
        except (ValueError, TypeError):
            s = pd.to_datetime(series, errors="raise", utc=True)
    else:
        s = series
    if not pd.api.types.is_datetime64_any_dtype(s):
        # mixed offsets produce object dtype
        s = pd.to_datetime(series, errors="raise", utc=True)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(ZoneInfo(tzname)).dt.tz_localize(None)
    return s


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except Exception as e:  # any unexpected failure -> abstain
        return _abstain(f"processing error: {type(e).__name__}: {e}")


def _analyze(data_path, manifest_path):
    import pandas as pd
    import numpy as np
    import pyarrow.parquet as pq

    # 1. manifest
    if not manifest_path or not os.path.isfile(manifest_path):
        return _abstain("manifest missing")
    try:
        with open(manifest_path, "rb") as f:
            manifest = json.loads(f.read().decode("utf-8"))
    except Exception:
        return _abstain("manifest is not valid JSON")
    if not isinstance(manifest, dict):
        return _abstain("manifest is not a JSON object")
    missing = [k for k in MANIFEST_FIELDS if k not in manifest]
    if missing:
        return _abstain(f"manifest missing fields: {missing}")
    rp = manifest["reporting_period"]
    if not isinstance(rp, dict) or not all(k in rp for k in ("start", "end", "timezone")):
        return _abstain("manifest reporting_period invalid")
    if not isinstance(manifest["columns"], list) or not isinstance(manifest["schema_changes"], list):
        return _abstain("manifest columns/schema_changes invalid")
    try:
        expected_rows = int(manifest["row_count"])
        if isinstance(manifest["row_count"], bool) or expected_rows != float(manifest["row_count"]):
            return _abstain("manifest row_count invalid")
    except Exception:
        return _abstain("manifest row_count invalid")

    # 2. checksum
    if not data_path or not os.path.isfile(data_path):
        return _abstain("data file missing")
    h = hashlib.sha256()
    with open(data_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest().lower() != str(manifest["file_sha256"]).strip().lower():
        return _abstain("SHA-256 mismatch")

    # units
    units_txt = json.dumps(manifest["units"]).lower()
    for bad in ("km", "kilomet", "meter", "cent"):
        if bad in units_txt and "mile" not in units_txt.replace("kilomet", ""):
            return _abstain(f"unsupported units declared: {manifest['units']}")
    if any(b in units_txt for b in ("km", "kilomet", "cents")):
        return _abstain(f"unsupported units declared: {manifest['units']}")

    # 3. row count
    table = pq.read_table(data_path)
    if table.num_rows != expected_rows:
        return _abstain("row count mismatch")
    df = table.to_pandas()
    if len(df) != expected_rows:
        return _abstain("row count mismatch")

    # 4. columns / schema changes
    renames = {}
    for ch in manifest["schema_changes"]:
        if not isinstance(ch, dict):
            return _abstain("schema_changes entry invalid")
        if ch.get("type") == "rename":
            if "from" not in ch or "to" not in ch:
                return _abstain("rename entry invalid")
            renames[ch["from"]] = ch["to"]
        else:
            return _abstain(f"unsupported schema change: {ch.get('type')}")
    cols = list(df.columns)
    if len(set(cols)) != len(cols):
        return _abstain("duplicate column names")
    mapping = {}
    for src, dst in renames.items():
        if src in cols:
            if dst in cols and dst != src:
                return _abstain("rename target already present")
            mapping[src] = dst
    df = df.rename(columns=mapping)
    delivered = set(df.columns)
    for c in manifest["columns"]:
        if not isinstance(c, dict) or "name" not in c:
            return _abstain("manifest column entry invalid")
        n = c["name"]
        if n not in delivered and renames.get(n) not in delivered:
            return _abstain(f"manifest column not delivered: {n}")
    for c in REQUIRED:
        if c not in delivered:
            return _abstain(f"required column missing: {c}")

    # nulls
    for c in REQUIRED:
        if df[c].isna().any():
            return _abstain(f"null values in required column {c}")

    # normalize
    tzname = rp["timezone"]
    from zoneinfo import ZoneInfo
    ZoneInfo(tzname)
    pickup = _to_local_naive(df["tpep_pickup_datetime"], tzname)
    if pickup.isna().any():
        return _abstain("unparseable pickup timestamps")
    start = pd.Timestamp(rp["start"])
    end = pd.Timestamp(rp["end"])
    if start.tzinfo is not None:
        start = start.tz_convert(ZoneInfo(tzname)).tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert(ZoneInfo(tzname)).tz_localize(None)
    pickup = pickup.astype("datetime64[ns]")
    if ((pickup < start) | (pickup >= end)).any():
        return _abstain("pickup timestamps outside reporting period")

    rid = _to_numeric(df["record_id"], "record_id")
    if (rid != np.floor(rid)).any():
        return _abstain("record_id not integral")
    dist = _to_numeric(df["trip_distance"], "trip_distance")
    fare = _to_numeric(df["fare_amount"], "fare_amount")
    pay = _to_numeric(df["payment_type"], "payment_type")
    if (pay != np.floor(pay)).any() or (~pay.isin([0, 1, 2, 3, 4, 5, 6])).any():
        return _abstain("undocumented payment_type codes")
    if ((dist <= 0) | (dist > 100)).any():
        return _abstain("trip_distance out of range")
    if ((fare < 0) | (fare > 1000)).any():
        return _abstain("fare_amount out of range")

    norm = pd.DataFrame({
        "record_id": rid.astype("int64"),
        "pickup": pickup,
        "trip_distance": dist,
        "payment_type": pay.astype("int64"),
        "fare_amount": fare,
    })
    for c in OPTIONAL:
        if c in df.columns:
            col = df[c]
            if c == "tpep_dropoff_datetime":
                try:
                    col = _to_local_naive(col.dropna(), tzname).reindex(df.index)
                except Exception:
                    pass
            norm["opt_" + c] = col.astype(str).where(col.notna(), "<NA>")

    # replay / conflict handling
    uniq = norm.drop_duplicates()
    if uniq["record_id"].duplicated().any():
        return _abstain("conflicting duplicate record_id values")
    n = len(uniq)
    if n == 0:
        return _abstain("no trips")

    total_fare = float(uniq["fare_amount"].sum())
    metrics = {
        "trip_count": int(n),
        "total_fare": total_fare,
        "mean_fare": float(total_fare / n),
        "mean_trip_distance": float(uniq["trip_distance"].mean()),
        "card_share": float((uniq["payment_type"] == 1).sum() / n),
    }
    for v in metrics.values():
        if isinstance(v, float) and not math.isfinite(v):
            return _abstain("non-finite metric")
    return {"status": "ok", "metrics": metrics, "reason": ""}
