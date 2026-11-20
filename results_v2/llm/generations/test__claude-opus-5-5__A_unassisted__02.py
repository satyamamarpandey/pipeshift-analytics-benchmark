import hashlib
import json
import math
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REQUIRED = ["tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
BAD_STATUS = ("incomplete", "partial", "draft", "retracted", "deprecated", "invalid",
              "corrupt", "failed", "withdrawn", "superseded", "truncated", "do not use")


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _flatten(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, path + (str(k).lower(),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            label = str(i)
            if isinstance(v, dict):
                for nk in ("name", "column", "field", "col"):
                    if isinstance(v.get(nk), str):
                        label = v[nk].lower()
                        break
            yield from _flatten(v, path + (label,))
    else:
        yield path, obj


def _fare_factor(manifest_items):
    vals = []
    specific = [(p, v) for p, v in manifest_items
                if isinstance(v, str) and p and ("unit" in p[-1] or "currency" in p[-1])
                and any("fare" in e for e in p)]
    generic = [(p, v) for p, v in manifest_items
               if isinstance(v, str) and p and
               (p[-1] in ("currency", "currency_unit", "amount_unit", "amount_units",
                          "monetary_unit", "money_unit", "amounts_unit")
                or ("unit" in p[-1] and any(("amount" in e or "money" in e or "monetary" in e)
                                            and "distance" not in e for e in p)))
               and not any("distance" in e for e in p)]
    for _, v in (specific or generic):
        s = v.strip().lower()
        if "cent" in s:
            vals.append(0.01)
        elif "usd" in s or "dollar" in s or s in ("$", "us$"):
            vals.append(1.0)
        else:
            return None, f"unsupported monetary unit/currency: {v}"
    if not vals:
        return 1.0, ""
    if len(set(vals)) > 1:
        return None, "conflicting monetary units in manifest"
    return vals[0], ""


def _distance_factor(manifest_items):
    vals = []
    for p, v in manifest_items:
        if not (isinstance(v, str) and p and "unit" in p[-1] and any("distance" in e for e in p)):
            continue
        s = v.strip().lower()
        if "km" in s or "kilomet" in s:
            vals.append(1 / 1.609344)
        elif "meter" in s or "metre" in s or s == "m":
            vals.append(1 / 1609.344)
        elif "mile" in s or s in ("mi", "miles"):
            vals.append(1.0)
        else:
            return None, f"unsupported distance unit: {v}"
    if not vals:
        return 1.0, ""
    if len(set(vals)) > 1:
        return None, "conflicting distance units in manifest"
    return vals[0], ""


def _card_codes(manifest_items):
    codes = set()
    for p, v in manifest_items:
        if not any("payment" in e for e in p) or not p:
            continue
        key = p[-1]
        if isinstance(v, str) and ("credit" in v.lower() or v.strip().lower() in ("card", "crd")):
            codes.add(key.strip())
        elif "credit" in key and isinstance(v, (int, float)) and not isinstance(v, bool):
            codes.add(str(int(v)))
    codes = {c for c in codes if re.fullmatch(r"-?\d+", c)}
    return codes or {"1"}


def _period(manifest_items):
    found = set()
    year = month = None
    for p, v in manifest_items:
        if not p:
            continue
        k = p[-1]
        if isinstance(v, str) and k in ("month", "period", "year_month", "data_month",
                                        "coverage_month", "yearmonth", "service_month",
                                        "reporting_period", "coverage", "data_period"):
            m = re.search(r"(\d{4})-(\d{2})", v)
            if m:
                found.add((int(m.group(1)), int(m.group(2))))
        elif isinstance(v, int) and not isinstance(v, bool):
            if k == "year" and 2000 <= v <= 2100:
                year = v
            elif k == "month" and 1 <= v <= 12:
                month = v
    if year and month:
        found.add((year, month))
    if len(found) > 1:
        return None, "conflicting periods in manifest"
    return (next(iter(found)) if found else None), ""


def _row_counts(manifest_items):
    keys = ("row_count", "rows", "num_rows", "n_rows", "record_count", "records",
            "trip_count", "num_records", "total_rows", "nrows", "row_cnt", "number_of_rows")
    out = set()
    for p, v in manifest_items:
        if p and p[-1] in keys and isinstance(v, int) and not isinstance(v, bool) \
                and not any("column" in e for e in p[:-1]):
            out.add(v)
    return out


def _check_checksum(data_path, manifest_items):
    cands = []
    for p, v in manifest_items:
        if p and isinstance(v, str) and any(t in p[-1] for t in ("sha256", "md5", "sha1", "checksum", "hash", "digest")):
            s = v.strip().lower()
            if ":" in s:
                s = s.split(":", 1)[1].strip()
            if re.fullmatch(r"[0-9a-f]+", s) and len(s) in (32, 40, 64):
                cands.append(s)
    if not cands:
        return ""
    with open(data_path, "rb") as f:
        data = f.read()
    digests = {32: hashlib.md5(data).hexdigest(), 40: hashlib.sha1(data).hexdigest(),
               64: hashlib.sha256(data).hexdigest()}
    for c in cands:
        if digests[len(c)] != c:
            return "data file checksum does not match manifest"
    return ""


def _check_status(manifest_items):
    for p, v in manifest_items:
        if not p:
            continue
        k = p[-1]
        if isinstance(v, str) and k in ("status", "state", "quality", "validation",
                                        "data_status", "release_status", "quality_status"):
            s = v.lower()
            if any(b in s for b in BAD_STATUS):
                return f"manifest flags data as {v}"
        if isinstance(v, bool) and not v and k in ("complete", "is_complete", "valid",
                                                    "is_valid", "validated", "passed"):
            return f"manifest flags {k}=false"
        if isinstance(v, bool) and v and k in ("retracted", "deprecated", "superseded",
                                                "partial", "is_partial", "truncated", "withdrawn"):
            return f"manifest flags {k}=true"
    return ""


def _to_numeric(s):
    before = s.isna().sum()
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().sum() > before:
        return None
    return out.astype("float64")


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return _abstain(f"cannot read manifest: {type(e).__name__}")
        items = list(_flatten(manifest))

        r = _check_status(items)
        if r:
            return _abstain(r)
        try:
            r = _check_checksum(data_path, items)
        except Exception as e:
            return _abstain(f"cannot read data file: {type(e).__name__}")
        if r:
            return _abstain(r)

        fare_k, r = _fare_factor(items)
        if fare_k is None:
            return _abstain(r)
        dist_k, r = _distance_factor(items)
        if dist_k is None:
            return _abstain(r)
        period, r = _period(items)
        if r:
            return _abstain(r)
        expected_counts = _row_counts(items)
        card_codes = _card_codes(items)

        try:
            df = pq.read_table(data_path).to_pandas()
        except Exception as e:
            return _abstain(f"cannot read parquet: {type(e).__name__}")

        lower = {c.lower(): c for c in df.columns}
        rename = {}
        for need in REQUIRED + ["record_id"]:
            if need not in df.columns and need.lower() in lower:
                rename[lower[need.lower()]] = need
        df = df.rename(columns=rename)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            return _abstain(f"missing columns: {missing}")

        raw_n = len(df)
        if raw_n == 0:
            return _abstain("no records")

        if "record_id" in df.columns:
            if df["record_id"].isna().any():
                return _abstain("null record_id values")
            dup_mask = df.duplicated(subset=["record_id"], keep=False)
            if dup_mask.any():
                dups = df[dup_mask]
                try:
                    full_dup_n = dups.astype(str).drop_duplicates().shape[0]
                except Exception:
                    full_dup_n = dups.drop_duplicates().shape[0]
                if full_dup_n != dups["record_id"].nunique():
                    return _abstain("conflicting records share the same record_id")
                df = df.drop_duplicates(subset=["record_id"], keep="first")
        dedup_n = len(df)

        fare = _to_numeric(df["fare_amount"])
        dist = _to_numeric(df["trip_distance"])
        if fare is None or dist is None:
            return _abstain("non-numeric fare or distance values")
        if fare.isna().any():
            return _abstain("missing fare_amount values")
        if dist.isna().any():
            return _abstain("missing trip_distance values")
        if not np.isfinite(fare.to_numpy()).all() or not np.isfinite(dist.to_numpy()).all():
            return _abstain("non-finite fare or distance values")

        pu = df["tpep_pickup_datetime"]
        try:
            if not pd.api.types.is_datetime64_any_dtype(pu):
                pu = pd.to_datetime(pu, errors="coerce")
            if getattr(pu.dt, "tz", None) is not None:
                try:
                    pu = pu.dt.tz_convert("America/New_York").dt.tz_localize(None)
                except Exception:
                    pu = pu.dt.tz_convert(None)
        except Exception:
            return _abstain("unparseable pickup datetimes")
        if pu.isna().any():
            return _abstain("missing or invalid pickup datetimes")

        mask = pd.Series(True, index=df.index)
        if period is not None:
            y, m = period
            mask = (pu.dt.year == y) & (pu.dt.month == m)
            in_frac = mask.mean()
            if in_frac < 0.9:
                return _abstain(f"only {in_frac:.1%} of trips fall in manifest period {y}-{m:02d}")
        else:
            ym = pu.dt.year * 100 + pu.dt.month
            top = ym.value_counts()
            if top.iloc[0] / len(ym) < 0.9:
                return _abstain("data does not cover a single identifiable month")
            mask = ym == top.index[0]
        filt_n = int(mask.sum())

        if expected_counts and not (expected_counts & {raw_n, dedup_n, filt_n}):
            return _abstain(f"row count mismatch: manifest {sorted(expected_counts)}, file {raw_n}")

        fare = fare[mask] * fare_k
        dist = dist[mask] * dist_k
        pay = df.loc[mask, "payment_type"]
        n = int(filt_n)
        if n == 0:
            return _abstain("no trips in period")

        if float(fare.median()) > 150:
            return _abstain("implausible fare scale (possible unit mismatch)")
        if float(dist.median()) > 100:
            return _abstain("implausible distance scale (possible unit mismatch)")

        if pd.api.types.is_numeric_dtype(pay):
            pnum = pd.to_numeric(pay, errors="coerce")
            codes = {int(c) for c in card_codes}
            is_card = pnum.isin(codes)
        else:
            ps = pay.astype("string").str.strip().str.lower()
            num = pd.to_numeric(ps, errors="coerce")
            is_card = num.isin({int(c) for c in card_codes}) | ps.str.contains(
                r"credit|^card$|^crd$", regex=True, na=False)
            is_card = is_card.fillna(False).astype(bool)

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
    except Exception as e:
        return _abstain(f"unexpected error: {type(e).__name__}")
