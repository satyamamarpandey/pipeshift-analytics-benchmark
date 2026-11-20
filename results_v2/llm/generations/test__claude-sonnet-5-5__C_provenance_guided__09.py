import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": str(reason)}


_REQUIRED = ["record_id", "tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
_CONTRACT = [
    "record_id", "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]
_MANIFEST_FIELDS = [
    "dataset", "snapshot_id", "snapshot_version", "reporting_period", "file",
    "file_sha256", "row_count", "columns", "units", "schema_changes", "description",
]


def _to_local_naive(series, tz):
    """Convert a series to naive local timestamps; raise ValueError on failure."""
    s = series
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        if getattr(s.dtype, "tz", None) is not None:
            s = s.dt.tz_convert(tz).dt.tz_localize(None)
        return s.astype("datetime64[ns]")
    if pd.api.types.is_numeric_dtype(s.dtype):
        raise ValueError("numeric timestamps are ambiguous")
    try:
        parsed = pd.to_datetime(s, errors="raise")
        if getattr(parsed.dtype, "tz", None) is not None:
            parsed = parsed.dt.tz_convert(tz).dt.tz_localize(None)
        return parsed.astype("datetime64[ns]")
    except Exception:
        pass
    parsed = pd.to_datetime(s, errors="raise", utc=True)
    # Mixed: naive strings would be assumed UTC here, which is unsafe.
    return parsed.dt.tz_convert(tz).dt.tz_localize(None).astype("datetime64[ns]")


def _to_numeric(series, name):
    if pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError(f"{name} is boolean")
    num = pd.to_numeric(series, errors="coerce")
    bad = num.isna() & series.notna()
    if bad.any():
        raise ValueError(f"{name} has non-numeric values")
    num = num.astype("float64")
    if np.isinf(num.to_numpy()).any():
        raise ValueError(f"{name} has infinite values")
    return num


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # 1. Manifest
        if not os.path.isfile(manifest_path):
            return _abstain("manifest missing")
        try:
            with open(manifest_path, "rb") as f:
                manifest = json.loads(f.read().decode("utf-8"))
        except Exception:
            return _abstain("manifest is not valid JSON")
        if not isinstance(manifest, dict):
            return _abstain("manifest is not a JSON object")
        missing = [k for k in _MANIFEST_FIELDS if k not in manifest]
        if missing:
            return _abstain(f"manifest missing fields: {missing}")
        period = manifest["reporting_period"]
        if not isinstance(period, dict) or "start" not in period or "end" not in period:
            return _abstain("manifest reporting_period invalid")
        tz = period.get("timezone") or "America/New_York"
        try:
            start = pd.Timestamp(period["start"])
            end = pd.Timestamp(period["end"])
            if start.tzinfo is not None:
                start = start.tz_convert(tz).tz_localize(None)
            if end.tzinfo is not None:
                end = end.tz_convert(tz).tz_localize(None)
        except Exception:
            return _abstain("reporting_period unparsable")
        if not start < end:
            return _abstain("reporting_period empty or inverted")

        # units
        units_txt = json.dumps(manifest["units"]).lower()
        if any(w in units_txt for w in ("kilomet", "km", "cent", "meter")) and "mile" not in units_txt:
            return _abstain("units do not match contract")
        if any(w in units_txt for w in ("kilomet", "km", "cents")):
            return _abstain("units do not match contract")

        # 2. Hash
        if not os.path.isfile(data_path):
            return _abstain("data file missing")
        h = hashlib.sha256()
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest().lower() != str(manifest["file_sha256"]).strip().lower():
            return _abstain("SHA-256 mismatch")

        # 3. Row count
        try:
            table = pq.read_table(data_path)
        except Exception as e:
            return _abstain(f"cannot read parquet: {e}")
        try:
            expected_rows = int(manifest["row_count"])
        except Exception:
            return _abstain("row_count invalid")
        if table.num_rows != expected_rows:
            return _abstain("row count mismatch")
        df = table.to_pandas()

        # 4. Columns
        changes = manifest["schema_changes"] or []
        renames = {}
        for ch in changes:
            if not isinstance(ch, dict) or ch.get("type") != "rename":
                return _abstain("unsupported schema change")
            renames[ch["from"]] = ch["to"]
        delivered = list(df.columns)
        if len(set(delivered)) != len(delivered):
            return _abstain("duplicate column names")
        for old, new in renames.items():
            if old in delivered and new in delivered:
                return _abstain("rename conflict")
        df = df.rename(columns={o: n for o, n in renames.items() if o in df.columns})
        dcols = set(df.columns)
        mcols = []
        for c in manifest["columns"]:
            if not isinstance(c, dict) or "name" not in c:
                return _abstain("manifest columns invalid")
            mcols.append(renames.get(c["name"], c["name"]))
        if not set(mcols) <= dcols:
            return _abstain("delivered columns disagree with manifest")
        absent = [c for c in _REQUIRED if c not in dcols]
        if absent:
            return _abstain(f"required columns missing: {absent}")

        cols = [c for c in _CONTRACT if c in df.columns]
        df = df[cols].copy()

        # Normalize types
        for c in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
            if c in df.columns:
                orig_null = df[c].isna()
                try:
                    df[c] = _to_local_naive(df[c], tz)
                except Exception as e:
                    return _abstain(f"cannot parse {c}: {e}")
                if (df[c].isna() & ~orig_null).any():
                    return _abstain(f"unparsable values in {c}")
        for c in cols:
            if c in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
                continue
            try:
                df[c] = _to_numeric(df[c], c)
            except Exception as e:
                return _abstain(str(e))

        # Required non-null
        if df[_REQUIRED].isna().any().any():
            return _abstain("nulls in required fields")

        # Integer-valued checks
        for c in ("record_id", "payment_type"):
            if (df[c] != np.floor(df[c])).any():
                return _abstain(f"{c} not integer-valued")
        if (~df["payment_type"].isin([0, 1, 2, 3, 4, 5, 6])).any():
            return _abstain("undocumented payment_type code")

        # Ranges
        if not ((df["trip_distance"] > 0) & (df["trip_distance"] <= 100)).all():
            return _abstain("trip_distance out of range")
        if not ((df["fare_amount"] >= 0) & (df["fare_amount"] <= 1000)).all():
            return _abstain("fare_amount out of range")

        # 5. Period
        pu = df["tpep_pickup_datetime"]
        if not ((pu >= start) & (pu < end)).all():
            return _abstain("pickup outside reporting period")

        # Replay / conflict
        df = df.drop_duplicates()
        if df["record_id"].duplicated().any():
            return _abstain("conflicting duplicate record_id")

        n = len(df)
        if n == 0:
            return _abstain("no trips")

        total_fare = float(df["fare_amount"].sum())
        metrics = {
            "trip_count": int(n),
            "total_fare": total_fare,
            "mean_fare": float(df["fare_amount"].mean()),
            "mean_trip_distance": float(df["trip_distance"].mean()),
            "card_share": float((df["payment_type"] == 1).sum() / n),
        }
        return {"status": "ok", "metrics": metrics, "reason": ""}
    except Exception as e:
        return _abstain(f"unexpected error: {e}")
