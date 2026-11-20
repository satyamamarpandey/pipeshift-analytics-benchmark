import hashlib
import json
import math
import os
import re

import numpy as np
import pandas as pd

REQUIRED = ["tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
BAD_STATUS = {"draft", "deprecated", "superseded", "retracted", "invalid", "corrupt",
              "corrupted", "incomplete", "partial", "preliminary", "do_not_use",
              "embargoed", "withdrawn", "failed", "error", "quarantined", "revoked"}
ISSUE_WORDS = ("fare", "distance", "payment", "duplicate", "incomplete", "corrupt",
               "missing", "truncat", "partial", "wrong", "invalid", "unit", "currency",
               "shift", "error", "do not use", "retract", "supersede")


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _flatten(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{str(k).lower()}" if prefix else str(k).lower()
            out.append((p, v))
            out.extend(_flatten(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            out.append((p, v))
            out.extend(_flatten(v, p))
    return out


def _last(path):
    return re.sub(r"\[\d+\]$", "", path.split(".")[-1])


def _find(flat, names):
    for p, v in flat:
        if _last(p) in names and not isinstance(v, (dict, list)):
            return p, v
    return None, None


def _file_hashes(path):
    hs = {"md5": hashlib.md5(), "sha1": hashlib.sha1(), "sha256": hashlib.sha256()}
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            for h in hs.values():
                h.update(chunk)
    return {k: h.hexdigest().lower() for k, h in hs.items()}


def _parse_month(v):
    if v is None:
        return None
    m = re.search(r"(\d{4})[-_/]?(\d{2})", str(v))
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return y, mo
    return None


def _to_int(v):
    try:
        if isinstance(v, bool):
            return None
        f = float(v)
        if math.isfinite(f) and f == int(f):
            return int(f)
    except (TypeError, ValueError):
        pass
    return None


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _analyze(data_path, manifest_path)
    except Exception as e:  # never crash
        return _abstain(f"unexpected error: {type(e).__name__}")


def _analyze(data_path, manifest_path):
    # ---------- manifest ----------
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return _abstain("manifest unreadable or invalid JSON")
    if not isinstance(manifest, dict):
        return _abstain("manifest is not a JSON object")
    flat = _flatten(manifest)

    # status / quality flags
    for p, v in flat:
        lk = _last(p)
        if lk in ("status", "state", "quality", "quality_status", "release_status",
                  "data_status", "validation_status") and isinstance(v, str):
            if v.strip().lower().replace(" ", "_").replace("-", "_") in BAD_STATUS:
                return _abstain(f"manifest marks data as {v}")
        if lk in ("valid", "is_valid", "validated", "complete", "is_complete",
                  "usable", "fit_for_use") and v is False:
            return _abstain(f"manifest flag {lk} is false")
        if lk in ("deprecated", "superseded", "retracted", "is_partial", "partial",
                  "incomplete", "do_not_use", "embargoed") and v is True:
            return _abstain(f"manifest flag {lk} is true")
        if lk in ("known_issues", "issues", "warnings", "caveats", "errata"):
            items = v if isinstance(v, list) else ([v] if isinstance(v, str) else [])
            for it in items:
                txt = json.dumps(it).lower() if not isinstance(it, str) else it.lower()
                if txt.strip() and any(w in txt for w in ISSUE_WORDS):
                    return _abstain("manifest reports known data issues")

    # checksum
    if not os.path.isfile(data_path):
        return _abstain("data file missing")
    claimed = []
    for p, v in flat:
        lk = _last(p)
        if isinstance(v, str) and any(t in lk for t in ("sha256", "sha1", "md5", "checksum", "hash", "digest")):
            s = v.strip().lower()
            s = s.split(":", 1)[1] if ":" in s else s
            if re.fullmatch(r"[0-9a-f]{32}|[0-9a-f]{40}|[0-9a-f]{64}", s):
                claimed.append(s)
    if claimed:
        actual = set(_file_hashes(data_path).values())
        if not any(c in actual for c in claimed):
            return _abstain("data file checksum does not match manifest")

    # ---------- data ----------
    try:
        df = pd.read_parquet(data_path)
    except Exception:
        return _abstain("data file unreadable as Parquet")

    lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    for col in REQUIRED + ["record_id", "tpep_dropoff_datetime", "total_amount", "tip_amount"]:
        if col not in df.columns and col.lower() in lower_map:
            rename[lower_map[col.lower()]] = col
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return _abstain(f"missing required columns: {missing}")
    raw_n = len(df)
    if raw_n == 0:
        return _abstain("no records")

    # duplicates
    try:
        df = df.drop_duplicates()
    except TypeError:
        pass
    if "record_id" in df.columns:
        if df["record_id"].isna().any():
            return _abstain("null record_id values")
        if df["record_id"].duplicated().any():
            return _abstain("conflicting duplicate record_id values")
    dedup_n = len(df)

    # row count check
    _, rc = _find(flat, {"row_count", "num_rows", "n_rows", "record_count", "records",
                         "rows", "nrows", "expected_rows", "total_rows", "row_cnt",
                         "num_records", "trip_count", "count"})
    rc_i = _to_int(rc)
    if rc_i is not None and rc_i not in (raw_n, dedup_n):
        return _abstain(f"row count {raw_n} does not match manifest {rc_i}")

    # units
    dist_factor = 1.0
    fare_factor = 1.0
    for p, v in flat:
        if not isinstance(v, str):
            continue
        lk = _last(p)
        val = v.strip().lower()
        is_dist = ("distance" in lk and "unit" in p) or lk in ("distance_unit", "distance_units")
        is_fare = (("fare" in lk or "amount" in lk) and "unit" in p) or lk in (
            "currency", "fare_currency", "currency_unit", "money_unit", "amount_unit", "amount_units")
        if is_dist and "distance" in p:
            if val in ("mi", "mile", "miles", "statute_miles", "statute miles"):
                dist_factor = 1.0
            elif val in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
                dist_factor = 1 / 1.609344
            elif val in ("m", "meter", "meters", "metre", "metres"):
                dist_factor = 1 / 1609.344
            elif val in ("double", "float", "float64", "float32", "number", "decimal"):
                pass
            else:
                return _abstain(f"unsupported distance unit: {v}")
        elif is_fare:
            if val in ("usd", "us dollars", "dollars", "dollar", "$", "us$", "usd ($)"):
                pass
            elif val in ("cents", "usd cents", "cent", "usd_cents"):
                fare_factor = 0.01
            elif val in ("double", "float", "float64", "float32", "number", "decimal"):
                pass
            else:
                return _abstain(f"unsupported currency/unit: {v}")

    # numeric columns
    fare = pd.to_numeric(df["fare_amount"], errors="coerce").astype("float64") * fare_factor
    dist = pd.to_numeric(df["trip_distance"], errors="coerce").astype("float64") * dist_factor

    # period filtering
    pu = df["tpep_pickup_datetime"]
    if not pd.api.types.is_datetime64_any_dtype(pu):
        if pd.api.types.is_numeric_dtype(pu):
            mx = pd.to_numeric(pu, errors="coerce").abs().max()
            unit = "s" if mx < 1e11 else "ms" if mx < 1e14 else "us" if mx < 1e17 else "ns"
            pu = pd.to_datetime(pu, unit=unit, errors="coerce")
        else:
            pu = pd.to_datetime(pu, errors="coerce")
    if getattr(pu.dt, "tz", None) is not None:
        pu = pu.dt.tz_convert("America/New_York").dt.tz_localize(None)

    ym = None
    for p, v in flat:
        if _last(p) in ("month", "period", "year_month", "data_month", "yearmonth",
                        "reporting_month", "reporting_period", "coverage_month") and not isinstance(v, (dict, list)):
            ym = _parse_month(v)
            if ym is None and _to_int(v) is not None and 1 <= _to_int(v) <= 12:
                _, yv = _find(flat, {"year", "data_year"})
                if _to_int(yv):
                    ym = (_to_int(yv), _to_int(v))
            if ym:
                break
    if ym is None:
        for p, v in flat:
            if _last(p) in ("period_start", "start", "start_date", "coverage_start"):
                ym = _parse_month(v)
                if ym:
                    break
    if ym is None:
        ym = _parse_month(os.path.basename(data_path)) if re.search(
            r"\d{4}-\d{2}", os.path.basename(data_path)) else None

    mask = pd.Series(True, index=df.index)
    if ym is not None:
        y, m = ym
        start = pd.Timestamp(year=y, month=m, day=1)
        end = start + pd.offsets.MonthBegin(1)
        mask = (pu >= start) & (pu < end)
        outside = int((~mask).sum())
        if outside > 0.05 * len(df):
            return _abstain("many trips fall outside the stated month")
    else:
        if pu.isna().mean() > 0.05:
            return _abstain("pickup timestamps unparseable")
        months = pu.dropna().dt.to_period("M")
        if len(months) == 0:
            return _abstain("no valid pickup timestamps")
        top = months.value_counts()
        if top.iloc[0] < 0.95 * len(months):
            return _abstain("data spans multiple months; period unknown")
        per = top.index[0]
        mask = pu.dt.to_period("M") == per

    fare = fare[mask]
    dist = dist[mask]
    pay = df.loc[mask, "payment_type"]
    n = int(mask.sum())
    if n == 0:
        return _abstain("no trips in period")

    if fare.isna().mean() > 0.05 or dist.isna().mean() > 0.05:
        return _abstain("too many missing fare or distance values")
    if not np.isfinite(fare.dropna()).all() or not np.isfinite(dist.dropna()).all():
        return _abstain("non-finite values in fare or distance")
    if fare.median() > 500 or abs(fare.median()) > 500:
        return _abstain("fare values implausible for USD")
    if dist.median() > 100:
        return _abstain("trip distances implausible for miles")

    # payment mapping
    card_codes = None
    for p, v in flat:
        if "payment" in p and isinstance(v, dict):
            codes = [k for k, lab in v.items() if isinstance(lab, str) and "credit" in lab.lower()]
            if codes:
                card_codes = set()
                for c in codes:
                    ci = _to_int(c)
                    card_codes.add(ci if ci is not None else str(c).strip().lower())
                break
    if card_codes is None:
        card_codes = {1}

    if pd.api.types.is_numeric_dtype(pay):
        pnum = pd.to_numeric(pay, errors="coerce")
        is_card = pnum.isin([c for c in card_codes if isinstance(c, int)])
    else:
        ps = pay.astype("string").str.strip().str.lower()
        pnum = pd.to_numeric(ps, errors="coerce")
        is_card = pnum.isin([c for c in card_codes if isinstance(c, int)]) | \
            ps.isin([c for c in card_codes if isinstance(c, str)]).fillna(False) | \
            ps.str.contains("credit", na=False) | (ps == "card")
        is_card = is_card.fillna(False)
    if pd.api.types.is_numeric_dtype(pay):
        valid = pnum.dropna()
        if len(valid) and (~valid.isin([0, 1, 2, 3, 4, 5, 6])).mean() > 0.05:
            return _abstain("unexpected payment_type codes")

    metrics = {
        "trip_count": n,
        "total_fare": float(fare.sum()),
        "mean_fare": float(fare.mean()),
        "mean_trip_distance": float(dist.mean()),
        "card_share": float(is_card.sum()) / n,
    }
    for k, v in metrics.items():
        if isinstance(v, float) and not math.isfinite(v):
            return _abstain(f"metric {k} is not finite")
    return {"status": "ok", "metrics": metrics, "reason": ""}
