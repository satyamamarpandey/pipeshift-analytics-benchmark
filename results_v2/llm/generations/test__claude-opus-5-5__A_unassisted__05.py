import hashlib
import json
import math
import os
import re

import numpy as np
import pandas as pd

REQUIRED = ["tpep_pickup_datetime", "fare_amount", "trip_distance", "payment_type"]
BAD_STATUS = ("retract", "invalid", "corrupt", "do not use", "do_not_use", "withdrawn",
              "deprecated", "superseded", "incomplete", "partial", "draft", "failed",
              "error", "quarantin", "embargo", "revoked", "obsolete", "preliminary")


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _flatten(obj, prefix=()):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_flatten(v, prefix + (str(k),)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_flatten(v, prefix + (str(i),)))
    else:
        out.append((prefix, obj))
    return out


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _find(flat, names, scalar=True):
    names = {_norm(n) for n in names}
    for path, val in flat:
        if path and _norm(path[-1]) in names:
            if not scalar or not isinstance(val, (dict, list)):
                return path, val
    return None, None


def _find_all(flat, names):
    names = {_norm(n) for n in names}
    return [(p, v) for p, v in flat if p and _norm(p[-1]) in names]


def _file_hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unit_for(manifest_flat, column):
    col = _norm(column)
    for path, val in manifest_flat:
        if not isinstance(val, str):
            continue
        np_ = [_norm(p) for p in path]
        if col in np_ and any(k in np_[-1] for k in ("unit", "units", "currency", "scale")):
            return val
        if len(np_) >= 2 and np_[-1] == col and any("unit" in p for p in np_[:-1]):
            return val
    return None


def _parse_period(flat):
    p, v = _find(flat, ["period", "month", "year_month", "yearmonth", "data_month",
                        "reporting_period", "coverage_month", "billing_month"])
    if isinstance(v, str):
        m = re.search(r"(\d{4})[-/_]?(\d{1,2})", v)
        if m and 1 <= int(m.group(2)) <= 12:
            return int(m.group(1)), int(m.group(2))
    _, y = _find(flat, ["year"])
    _, mo = _find(flat, ["month"])
    try:
        if y is not None and mo is not None:
            y, mo = int(y), int(mo)
            if 1 <= mo <= 12:
                return y, mo
    except (TypeError, ValueError):
        pass
    _, s = _find(flat, ["start_date", "period_start", "start", "from", "coverage_start"])
    if isinstance(s, str):
        m = re.search(r"(\d{4})-(\d{2})", s)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _card_codes(flat, series):
    codes = set()
    for path, val in flat:
        np_ = [_norm(p) for p in path]
        if any("payment" in p for p in np_) and isinstance(val, str):
            lv = val.lower()
            if "credit" in lv or lv.strip() in ("card", "crd", "cc"):
                key = path[-1]
                codes.add(_norm(key))
    if not codes:
        codes = {"1", "10", "creditcard", "credit", "crd", "card", "cc"}
    def is_card(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return False
        if isinstance(x, (int, np.integer)):
            k = str(int(x))
        elif isinstance(x, (float, np.floating)):
            k = str(int(x)) if float(x).is_integer() else str(x)
        else:
            k = _norm(x)
            if re.fullmatch(r"\d+(\.0+)?", str(x).strip()):
                k = str(int(float(x)))
        return k in codes
    return series.map(is_card).astype(bool)


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except Exception as e:  # noqa: BLE001
        return _abstain(f"unexpected error: {type(e).__name__}")


def _analyze(data_path, manifest_path):
    # ---- manifest ----
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return _abstain("manifest unreadable or invalid JSON")
    flat = _flatten(manifest)

    for path, val in _find_all(flat, ["status", "state", "data_status", "quality",
                                      "quality_status", "validation_status", "release_status"]):
        if isinstance(val, str) and any(b in val.lower() for b in BAD_STATUS):
            return _abstain(f"manifest status indicates unusable data: {val}")
    for path, val in _find_all(flat, ["retracted", "deprecated", "superseded", "withdrawn",
                                      "corrupt", "corrupted", "invalid", "do_not_use", "partial",
                                      "truncated", "embargoed"]):
        if val is True or (isinstance(val, str) and val.strip().lower() in ("true", "yes")):
            return _abstain(f"manifest flag {path[-1]} set")
    for path, val in _find_all(flat, ["superseded_by", "replaced_by", "retracted_at",
                                      "withdrawn_at", "deprecated_at"]):
        if val not in (None, "", False):
            return _abstain(f"manifest indicates {path[-1]}")
    for path, val in _find_all(flat, ["complete", "is_complete", "validated", "is_valid",
                                      "valid", "verified", "qa_passed", "final", "is_final"]):
        if val is False or (isinstance(val, str) and val.strip().lower() in ("false", "no")):
            return _abstain(f"manifest flag {path[-1]} is false")

    # checksums
    if not os.path.exists(data_path):
        return _abstain("data file missing")
    for algo, names, ln in (("sha256", ["sha256", "sha256sum", "sha_256"], 64),
                            ("md5", ["md5", "md5sum"], 32),
                            ("sha1", ["sha1"], 40)):
        _, v = _find(flat, names)
        if isinstance(v, str):
            v2 = v.split(":")[-1].strip().lower()
            if re.fullmatch(r"[0-9a-f]+", v2) and len(v2) == ln:
                if _file_hash(data_path, algo) != v2:
                    return _abstain(f"{algo} checksum mismatch")
    _, v = _find(flat, ["checksum", "hash", "file_hash", "digest"])
    if isinstance(v, str):
        v2 = v.split(":")[-1].strip().lower()
        algo = {64: "sha256", 32: "md5", 40: "sha1"}.get(len(v2))
        if algo and re.fullmatch(r"[0-9a-f]+", v2):
            if _file_hash(data_path, algo) != v2:
                return _abstain("checksum mismatch")
    _, v = _find(flat, ["file_size", "size_bytes", "bytes", "filesize"])
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        if int(v) != os.path.getsize(data_path):
            return _abstain("file size mismatch")

    # ---- data ----
    try:
        df = pd.read_parquet(data_path)
    except Exception:
        return _abstain("data file unreadable")
    raw_n = len(df)
    if raw_n == 0:
        return _abstain("no records")

    colmap = {_norm(c): c for c in df.columns}
    ren = {}
    for req in REQUIRED + ["record_id", "tpep_dropoff_datetime", "total_amount", "tip_amount"]:
        if req not in df.columns and _norm(req) in colmap:
            ren[colmap[_norm(req)]] = req
    df = df.rename(columns=ren)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return _abstain(f"missing columns: {missing}")

    # duplicates
    dedup = df.drop_duplicates()
    if "record_id" in dedup.columns:
        if dedup["record_id"].isna().any():
            pass
        elif dedup["record_id"].duplicated().any():
            return _abstain("conflicting records share the same record_id")
    df = dedup
    n_dedup = len(df)

    # row count check
    _, rc = _find(flat, ["row_count", "num_rows", "n_rows", "record_count", "records",
                         "rows", "trip_count", "num_records", "count", "total_rows"])
    if isinstance(rc, (int, float)) and not isinstance(rc, bool):
        if int(rc) not in (raw_n, n_dedup):
            return _abstain(f"row count mismatch (manifest {int(rc)}, data {raw_n})")

    # declared columns
    _, cols = _find(flat, ["columns", "schema", "fields"], scalar=False)
    # (not strictly checked; column presence already validated)

    # numeric coercion
    fare = pd.to_numeric(df["fare_amount"], errors="coerce").astype(float)
    dist = pd.to_numeric(df["trip_distance"], errors="coerce").astype(float)
    n = len(df)
    if fare.isna().mean() > 0.01 or dist.isna().mean() > 0.01:
        return _abstain("too many missing/non-numeric fare or distance values")
    if not np.isfinite(fare.dropna()).all() or not np.isfinite(dist.dropna()).all():
        return _abstain("non-finite values in fare or distance")

    # units
    fu = _unit_for(flat, "fare_amount")
    _, cur = _find(flat, ["currency", "fare_currency"])
    for u in (fu, cur):
        if isinstance(u, str):
            ul = u.lower()
            if "cent" in ul:
                fare = fare / 100.0
                break
            if ul.strip() not in ("usd", "us dollars", "dollars", "dollar", "$", "us$", "usd ($)") \
                    and "usd" not in ul and "dollar" not in ul:
                return _abstain(f"fare in unsupported unit/currency: {u}")
    du = _unit_for(flat, "trip_distance")
    _, du2 = _find(flat, ["distance_unit", "distance_units"])
    du = du or du2
    if isinstance(du, str):
        dl = du.lower().strip()
        if dl in ("km", "kilometers", "kilometres", "kilometer", "kilometre") or "kilomet" in dl:
            dist = dist / 1.609344
        elif dl in ("m", "meters", "metres", "meter", "metre"):
            dist = dist / 1609.344
        elif "mile" in dl or dl in ("mi",):
            pass
        else:
            return _abstain(f"unsupported distance unit: {du}")

    # plausibility
    med_fare = float(fare.median())
    med_dist = float(dist.median())
    if not (1.0 <= med_fare <= 150.0):
        return _abstain(f"implausible median fare {med_fare}; units unclear")
    if not (0.05 <= med_dist <= 60.0):
        return _abstain(f"implausible median distance {med_dist}; units unclear")

    # period filtering
    pu = df["tpep_pickup_datetime"]
    if not pd.api.types.is_datetime64_any_dtype(pu):
        pu = pd.to_datetime(pu, errors="coerce")
    if pu.isna().mean() > 0.01:
        return _abstain("too many unparseable pickup timestamps")
    _, tz = _find(flat, ["timezone", "time_zone", "tz"])
    try:
        if getattr(pu.dt, "tz", None) is not None:
            pu = pu.dt.tz_convert("America/New_York").dt.tz_localize(None)
        elif isinstance(tz, str) and tz.strip().upper() in ("UTC", "Z", "GMT", "ETC/UTC"):
            pu = pu.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    except Exception:
        return _abstain("cannot normalize timestamps to New York time")

    period = _parse_period(flat)
    mask = pd.Series(True, index=df.index)
    if period is not None:
        y, m = period
        start = pd.Timestamp(year=y, month=m, day=1)
        end = start + pd.offsets.MonthBegin(1)
        mask = (pu >= start) & (pu < end)
        if mask.mean() < 0.95:
            return _abstain("large share of records fall outside the declared month")
    else:
        months = pu.dt.to_period("M").value_counts()
        if len(months) == 0:
            return _abstain("no valid timestamps")
        top = months.index[0]
        if months.iloc[0] / n < 0.95:
            return _abstain("data does not correspond to a single month")
        mask = pu.dt.to_period("M") == top

    fare = fare[mask]
    dist = dist[mask]
    pay = df.loc[mask, "payment_type"]
    trip_count = int(mask.sum())
    if trip_count == 0:
        return _abstain("no trips in period")

    card = _card_codes(flat, pay)
    metrics = {
        "trip_count": trip_count,
        "total_fare": float(fare.sum()),
        "mean_fare": float(fare.mean()),
        "mean_trip_distance": float(dist.mean()),
        "card_share": float(card.sum()) / trip_count,
    }
    for k, v in metrics.items():
        if isinstance(v, float) and not math.isfinite(v):
            return _abstain(f"non-finite metric {k}")
    return {"status": "ok", "metrics": metrics, "reason": ""}
