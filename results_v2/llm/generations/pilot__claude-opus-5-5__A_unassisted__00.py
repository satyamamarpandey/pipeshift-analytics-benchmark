"""Monthly summary of NYC TLC yellow taxi trips with manifest-driven validation."""

import hashlib
import json
import math
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_REQUIRED = ["fare_amount", "trip_distance", "payment_type"]

_BAD_STATUS = {
    "deprecated", "retracted", "superseded", "invalid", "draft", "incomplete",
    "partial", "corrupt", "corrupted", "withdrawn", "failed", "error",
    "quarantined", "obsolete", "revoked", "truncated",
}


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _flatten(obj, prefix=()):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_flatten(v, prefix + (str(k).lower(),)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_flatten(v, prefix + (str(i),)))
    else:
        out.append((prefix, obj))
    return out


def _file_hashes(path):
    hs = {"md5": hashlib.md5(), "sha1": hashlib.sha1(), "sha256": hashlib.sha256()}
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            for h in hs.values():
                h.update(chunk)
    return {k: v.hexdigest().lower() for k, v in hs.items()}


def _check_checksums(flat, data_path):
    hashes = None
    for path, val in flat:
        if not path or not isinstance(val, str):
            continue
        key = path[-1]
        joined = ".".join(path)
        if not any(t in joined for t in ("sha256", "sha1", "md5", "checksum", "hash", "digest")):
            continue
        if "manifest" in joined:
            continue
        v = val.strip().lower()
        algo = None
        if ":" in v:
            a, v = v.split(":", 1)
            a = a.replace("-", "")
            if a in ("md5", "sha1", "sha256"):
                algo = a
        if not re.fullmatch(r"[0-9a-f]+", v):
            continue
        if algo is None:
            for a in ("sha256", "sha1", "md5"):
                if a in joined:
                    algo = a
                    break
        if algo is None:
            algo = {32: "md5", 40: "sha1", 64: "sha256"}.get(len(v))
        if algo is None:
            continue
        if hashes is None:
            hashes = _file_hashes(data_path)
        if hashes[algo] != v:
            return f"data file {algo} checksum does not match manifest ({key})"
    return None


def _to_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\d+\s*", v):
        return int(v)
    return None


def _parse_month(v):
    if isinstance(v, str):
        m = re.search(r"(\d{4})[-/_]?(\d{2})", v)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12 and 2000 <= y <= 2100:
                return y, mo
    return None


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except Exception as e:  # noqa: BLE001
        return _abstain(f"unexpected error: {type(e).__name__}")


def _analyze(data_path, manifest_path):
    # ---------- manifest ----------
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return _abstain("manifest unreadable or not valid JSON")
    if not isinstance(manifest, (dict, list)):
        return _abstain("manifest has unexpected structure")
    flat = _flatten(manifest)

    # status / validity flags
    for path, val in flat:
        if not path:
            continue
        key = path[-1]
        if key in ("status", "state", "quality", "data_status", "release_status", "lifecycle"):
            if isinstance(val, str) and val.strip().lower() in _BAD_STATUS:
                return _abstain(f"manifest marks data as {val.strip().lower()}")
        if key in ("deprecated", "retracted", "superseded", "withdrawn", "incomplete",
                   "is_partial", "partial", "corrupt", "is_deprecated", "truncated"):
            if val is True or (isinstance(val, str) and val.strip().lower() in ("true", "yes")):
                return _abstain(f"manifest flags data as {key}")
        if key in ("valid", "is_valid", "complete", "is_complete", "validated"):
            if val is False or (isinstance(val, str) and val.strip().lower() in ("false", "no")):
                return _abstain(f"manifest flags data as not {key}")
        if key in ("superseded_by", "replaced_by") and val:
            return _abstain("manifest says data has been superseded")

    # dataset type
    for path, val in flat:
        if path and path[-1] in ("dataset", "taxi_type", "service", "service_type",
                                 "vehicle_type", "type", "name", "dataset_name") \
                and isinstance(val, str):
            v = val.lower()
            if ("green" in v or "fhv" in v or "for-hire" in v or "for_hire" in v) and "yellow" not in v:
                return _abstain(f"manifest describes non-yellow dataset: {val}")

    # checksum
    msg = _check_checksums(flat, data_path)
    if msg:
        return _abstain(msg)

    # units
    fare_scale = 1.0
    dist_scale = 1.0
    for path, val in flat:
        if not path or not isinstance(val, str):
            continue
        joined = ".".join(path)
        v = val.strip().lower()
        is_unitish = any(t in joined for t in ("unit", "currency", "denomination", "scale"))
        if not is_unitish:
            continue
        if "fare" in joined and "total" not in joined and "tip" not in joined:
            if "cent" in v:
                fare_scale = 0.01
            elif v in ("usd", "us$", "$", "dollar", "dollars", "us dollar", "us dollars") or "usd" in v or "dollar" in v:
                fare_scale = 1.0
            else:
                return _abstain(f"fare_amount unit/currency not USD: {val}")
        elif "distance" in joined:
            if v in ("mi", "mile", "miles") or "mile" in v:
                dist_scale = 1.0
            elif v in ("km", "kilometer", "kilometers", "kilometre", "kilometres") or "kilomet" in v:
                dist_scale = 1.0 / 1.609344
            elif v in ("m", "meter", "meters", "metre", "metres"):
                dist_scale = 1.0 / 1609.344
            elif v in ("ft", "feet", "foot"):
                dist_scale = 1.0 / 5280.0
            else:
                return _abstain(f"unrecognized trip_distance unit: {val}")
        elif path[-1] == "currency" or joined.endswith("currency"):
            if "usd" not in v and "dollar" not in v and v != "$":
                return _abstain(f"manifest currency is not USD: {val}")

    # credit card payment codes
    card_codes = set()
    for path, val in flat:
        if not path:
            continue
        joined = ".".join(path)
        if "payment" not in joined:
            continue
        if isinstance(val, str) and "credit" in val.lower():
            c = _to_int(path[-1])
            if c is not None:
                card_codes.add(c)
        elif "credit" in path[-1]:
            c = _to_int(val)
            if c is not None:
                card_codes.add(c)
    if not card_codes:
        card_codes = {1}

    # expected row count
    expected_rows = []
    for path, val in flat:
        if path and path[-1] in ("row_count", "num_rows", "n_rows", "rows", "record_count",
                                 "num_records", "n_records", "total_rows", "nrows",
                                 "records", "trip_count", "expected_rows", "row_total"):
            c = _to_int(val)
            if c is not None:
                expected_rows.append(c)

    # period
    period = None
    year = month = None
    for path, val in flat:
        if not path:
            continue
        key = path[-1]
        if key in ("month", "period", "year_month", "data_month", "coverage_month",
                   "reporting_period", "yearmonth", "coverage", "partition"):
            p = _parse_month(val)
            if p:
                period = p
                break
            if key == "month":
                month = _to_int(val)
        if key == "year":
            year = _to_int(val)
    if period is None and year and month and 1 <= month <= 12:
        period = (year, month)

    # ---------- data ----------
    try:
        pf = pq.ParquetFile(data_path)
        raw_rows = pf.metadata.num_rows
        df = pf.read().to_pandas()
    except Exception:
        return _abstain("data file unreadable as Parquet")

    if len(df) != raw_rows:
        return _abstain("row count mismatch while reading data")
    if len(df) == 0:
        return _abstain("data file contains no rows")

    lower_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in lower_map:
            return _abstain(f"ambiguous duplicate column {c}")
        lower_map[lc] = c
    df = df.rename(columns={v: k for k, v in lower_map.items()})
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        return _abstain(f"missing required columns: {', '.join(missing)}")

    # duplicates
    n_after_dedup = len(df)
    if "record_id" in df.columns:
        if df["record_id"].isna().any():
            return _abstain("null record_id values")
        dup_mask = df["record_id"].duplicated(keep=False)
        if dup_mask.any():
            dups = df[dup_mask]
            try:
                exact = dups.drop_duplicates()
                conflicting = exact["record_id"].duplicated().any()
            except TypeError:
                conflicting = True
            if conflicting:
                return _abstain("conflicting records share the same record_id")
            df = df.drop_duplicates(subset=["record_id"], keep="first")
            n_after_dedup = len(df)

    if expected_rows:
        if not any(e in (raw_rows, n_after_dedup) for e in expected_rows):
            return _abstain(
                f"row count {raw_rows} does not match manifest {expected_rows[0]}")

    # numeric coercion
    fare = pd.to_numeric(df["fare_amount"], errors="coerce")
    dist = pd.to_numeric(df["trip_distance"], errors="coerce")
    if fare.isna().any() or df["fare_amount"].isna().any():
        return _abstain("missing or non-numeric fare_amount values")
    if dist.isna().any() or df["trip_distance"].isna().any():
        return _abstain("missing or non-numeric trip_distance values")
    if not np.isfinite(fare.to_numpy(dtype=float)).all() or \
            not np.isfinite(dist.to_numpy(dtype=float)).all():
        return _abstain("non-finite fare or distance values")

    # period filter
    keep = pd.Series(True, index=df.index)
    if period is not None:
        if "tpep_pickup_datetime" not in df.columns:
            return _abstain("cannot verify period: pickup datetime missing")
        pu = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        if pu.isna().mean() > 0.01:
            return _abstain("unparseable pickup datetimes")
        try:
            if getattr(pu.dt, "tz", None) is not None:
                pu = pu.dt.tz_convert("America/New_York").dt.tz_localize(None)
        except Exception:
            pass
        in_period = (pu.dt.year == period[0]) & (pu.dt.month == period[1])
        frac_out = 1.0 - float(in_period.mean())
        if frac_out > 0.10:
            return _abstain(
                f"{frac_out:.1%} of trips fall outside manifest period {period[0]}-{period[1]:02d}")
        keep = in_period.fillna(False)

    fare = fare[keep].astype(float) * fare_scale
    dist = dist[keep].astype(float) * dist_scale
    pay = df.loc[keep, "payment_type"]

    n = int(len(fare))
    if n == 0:
        return _abstain("no trips remain for the period")

    # payment classification
    if pay.isna().mean() > 0.5:
        return _abstain("payment_type mostly missing")
    pay_num = pd.to_numeric(pay, errors="coerce")
    non_null = pay.notna()
    if (pay_num.notna() | ~non_null).all():
        is_card = pay_num.isin(list(card_codes))
    else:
        s = pay.astype(str).str.strip().str.lower()
        codes_str = {str(c) for c in card_codes}
        is_card = s.str.contains("credit", na=False) | s.isin({"card", "crd", "cc"} | codes_str)

    # sanity checks on scale
    med_fare = float(np.median(fare.to_numpy()))
    med_dist = float(np.median(dist.to_numpy()))
    if med_fare > 200 or med_fare < 0:
        return _abstain("implausible fare scale (possible unit mismatch)")
    if med_dist > 100 or med_dist < 0:
        return _abstain("implausible trip distance scale (possible unit mismatch)")

    total_fare = float(fare.sum())
    metrics = {
        "trip_count": n,
        "total_fare": total_fare,
        "mean_fare": total_fare / n,
        "mean_trip_distance": float(dist.sum()) / n,
        "card_share": float(is_card.sum()) / n,
    }
    for k, v in metrics.items():
        if isinstance(v, float) and not math.isfinite(v):
            return _abstain(f"non-finite metric {k}")
    return {"status": "ok", "metrics": metrics, "reason": ""}
