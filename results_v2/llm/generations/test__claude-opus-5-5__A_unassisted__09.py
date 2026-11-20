import hashlib
import json
import math
import os
import re

import numpy as np
import pandas as pd

REQUIRED = ["tpep_pickup_datetime", "trip_distance", "payment_type", "fare_amount"]
BAD_STATUS_WORDS = (
    "incomplete", "partial", "deprecated", "superseded", "retracted", "invalid",
    "failed", "fail", "draft", "corrupt", "withdrawn", "preliminary", "truncated",
    "do not use", "do_not_use", "unverified", "error", "stale", "obsolete",
)
STATUS_KEYS = ("status", "state", "quality", "quality_status", "validation",
               "validation_status", "data_status", "release_status")
COUNT_KEYS = ("row_count", "num_rows", "n_rows", "rows", "record_count",
              "num_records", "n_records", "records", "trip_count", "expected_rows",
              "total_rows", "count")
KM_PER_MILE = 1.609344


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _walk(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = path + (str(k).lower(),)
            yield kp, v
            yield from _walk(v, kp)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, path + (str(i),))


def _scalar_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\d+\s*", v):
        return int(v)
    return None


def _file_hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _column_units(manifest):
    """Collect unit/currency annotations per column name."""
    units = {}
    cols = ("fare_amount", "trip_distance", "total_amount", "tip_amount")

    def visit(obj, ctx_name=None):
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("column") or obj.get("field")
            if isinstance(name, str) and name.lower() in cols:
                ctx_name = name.lower()
            for k, v in obj.items():
                kl = str(k).lower()
                if kl in cols:
                    if isinstance(v, str):
                        units.setdefault(kl, []).append(v.lower())
                    visit(v, kl)
                elif ctx_name and kl in ("unit", "units", "currency", "uom", "measure"):
                    if isinstance(v, str):
                        units.setdefault(ctx_name, []).append(v.lower())
                else:
                    # keys like fare_amount_unit / distance_unit / currency_unit
                    if isinstance(v, str) and ("unit" in kl or "currency" in kl):
                        if "fare" in kl or "amount" in kl or "currency" in kl or "money" in kl:
                            units.setdefault("fare_amount", []).append(v.lower())
                        if "distance" in kl:
                            units.setdefault("trip_distance", []).append(v.lower())
                    visit(v, ctx_name)
        elif isinstance(obj, list):
            for v in obj:
                visit(v, ctx_name)

    visit(manifest)
    return units


def _parse_period(manifest):
    start = end = None
    for kp, v in _walk(manifest):
        k = kp[-1]
        if not isinstance(v, str):
            continue
        s = v.strip()
        if k in ("month", "period", "year_month", "yearmonth", "data_month",
                 "coverage_month", "reporting_period", "coverage"):
            m = re.fullmatch(r"(\d{4})[-/](\d{1,2})", s)
            if m:
                y, mo = int(m.group(1)), int(m.group(2))
                if 1 <= mo <= 12:
                    start = pd.Timestamp(y, mo, 1)
                    end = start + pd.offsets.MonthBegin(1)
                    return start, end
        if k in ("start_date", "period_start", "start", "from", "coverage_start", "min_date"):
            t = pd.to_datetime(s, errors="coerce")
            if not pd.isna(t):
                start = t.tz_localize(None) if t.tzinfo else t
        if k in ("end_date", "period_end", "end", "to", "coverage_end", "max_date"):
            t = pd.to_datetime(s, errors="coerce")
            if not pd.isna(t):
                end = t.tz_localize(None) if t.tzinfo else t
    if start is not None and end is not None:
        if end.hour == 0 and end.minute == 0 and end.second == 0 and end.day != 1:
            end = end + pd.Timedelta(days=1)  # inclusive end date
        elif end.day == 1 and end.hour == 0 and end <= start:
            return None, None
        return start, end
    if start is not None and start.day == 1:
        return start, start + pd.offsets.MonthBegin(1)
    return None, None


def _card_mask(series, manifest):
    # Look for an explicit mapping in the manifest
    card_codes = None
    for kp, v in _walk(manifest):
        if isinstance(v, dict) and any("payment" in p for p in kp):
            codes = [str(code).strip().lower() for code, lab in v.items()
                     if isinstance(lab, str) and ("credit" in lab.lower()
                                                  or lab.lower().strip() in ("card", "cc"))]
            if codes:
                card_codes = set(codes)
    s = series
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        if card_codes:
            nums = set()
            for c in card_codes:
                try:
                    nums.add(float(c))
                except ValueError:
                    pass
            if nums:
                return vals.isin(list(nums)).to_numpy(), None
        return (vals == 1).to_numpy(), None
    st = s.astype("string").str.strip().str.lower()
    if card_codes:
        return st.isin(list(card_codes)).fillna(False).to_numpy(dtype=bool), None
    known = {"1", "1.0", "credit card", "credit", "card", "cc", "credit_card", "creditcard"}
    other = {"2", "3", "4", "5", "6", "0", "2.0", "3.0", "4.0", "5.0", "6.0", "0.0",
             "cash", "no charge", "dispute", "unknown", "voided trip", "voided", "flex fare",
             "no_charge", "nocharge"}
    present = set(st.dropna().unique().tolist())
    unknown = present - known - other
    if unknown:
        return None, "unrecognized payment_type codes: %s" % sorted(list(unknown))[:5]
    return st.isin(list(known)).fillna(False).to_numpy(dtype=bool), None


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # ---------- manifest ----------
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return _abstain("manifest unreadable: %s" % type(e).__name__)
        if not isinstance(manifest, (dict, list)):
            return _abstain("manifest has unexpected structure")

        for kp, v in _walk(manifest):
            k = kp[-1]
            if k in STATUS_KEYS and isinstance(v, str):
                vl = v.lower()
                if any(w in vl for w in BAD_STATUS_WORDS) and "no error" not in vl:
                    return _abstain("manifest status indicates unusable data: %s" % v)
            if k in ("complete", "is_complete", "valid", "is_valid", "validated", "final",
                     "is_final", "passed") and v is False:
                return _abstain("manifest flags data as %s=false" % k)
            if k in ("partial", "is_partial", "deprecated", "is_deprecated", "superseded",
                     "retracted", "is_retracted", "truncated", "is_truncated",
                     "do_not_use", "corrupt", "is_corrupt") and v is True:
                return _abstain("manifest flags data as %s" % k)
            if k in ("superseded_by", "replaced_by", "retracted_at", "withdrawn_at") and v:
                return _abstain("manifest indicates dataset was superseded/retracted")

        if not os.path.isfile(data_path):
            return _abstain("data file missing")

        # checksum
        for kp, v in _walk(manifest):
            k = kp[-1]
            if not isinstance(v, str):
                continue
            hv = v.strip().lower()
            if ":" in hv and hv.split(":", 1)[0] in ("sha256", "md5", "sha1"):
                algo, hv = hv.split(":", 1)
            else:
                algo = None
                for a in ("sha256", "sha1", "md5"):
                    if a in k:
                        algo = a
                        break
                if algo is None and k in ("checksum", "hash", "digest", "file_hash"):
                    algo = {64: "sha256", 40: "sha1", 32: "md5"}.get(len(hv))
            if algo and re.fullmatch(r"[0-9a-f]+", hv):
                if _file_hash(data_path, algo) != hv:
                    return _abstain("data file checksum does not match manifest")
            if k in ("file_size", "size_bytes", "bytes", "filesize"):
                pass
        for kp, v in _walk(manifest):
            if kp[-1] in ("file_size", "size_bytes", "filesize", "file_size_bytes"):
                n = _scalar_int(v)
                if n is not None and n != os.path.getsize(data_path):
                    return _abstain("data file size does not match manifest")

        # ---------- data ----------
        try:
            df = pd.read_parquet(data_path)
        except Exception as e:
            return _abstain("data unreadable: %s" % type(e).__name__)

        colmap = {c.lower(): c for c in df.columns}
        missing = [c for c in REQUIRED if c.lower() not in colmap]
        if missing:
            return _abstain("missing required columns: %s" % missing)
        df = df.rename(columns={colmap[c.lower()]: c for c in REQUIRED + ["record_id"]
                                if c.lower() in colmap})

        raw_n = len(df)
        if raw_n == 0:
            return _abstain("no records")

        # expected row count
        expected = None
        for kp, v in _walk(manifest):
            if kp[-1] in COUNT_KEYS:
                n = _scalar_int(v)
                if n is not None and n > 0:
                    if kp[-1] == "count" and len(kp) > 1 and not any(
                            x in ("rows", "records", "trips", "data") for x in kp[:-1]):
                        continue
                    expected = n
                    break

        # duplicates
        if "record_id" in df.columns:
            dup_mask = df.duplicated(subset=["record_id"], keep="first")
            if dup_mask.any():
                full_dups = df.duplicated(keep="first")
                if (dup_mask & ~full_dups).any():
                    conflicting = int((dup_mask & ~full_dups).sum())
                    if expected is None or expected != raw_n:
                        return _abstain("%d record_ids duplicated with conflicting values"
                                        % conflicting)
                if expected is None or expected != raw_n:
                    df = df.loc[~dup_mask].copy()
        else:
            full_dups = df.duplicated(keep="first")
            if full_dups.any() and (expected is None or expected != raw_n):
                df = df.loc[~full_dups].copy()

        n = len(df)
        if expected is not None and expected != n:
            return _abstain("row count %d does not match manifest (%d)" % (n, expected))

        # ---------- units ----------
        units = _column_units(manifest)
        fare_scale = 1.0
        for u in units.get("fare_amount", []):
            if "cent" in u:
                fare_scale = 0.01
            elif u in ("usd", "dollar", "dollars", "$", "us dollars", "us$") or "usd" in u \
                    or "dollar" in u:
                pass
            else:
                return _abstain("fare currency/unit not USD: %s" % u)
        dist_scale = 1.0
        for u in units.get("trip_distance", []):
            if u in ("km", "kilometer", "kilometers", "kilometre", "kilometres") or "kilomet" in u:
                dist_scale = 1.0 / KM_PER_MILE
            elif u in ("m", "meter", "meters", "metre", "metres"):
                dist_scale = 1.0 / (KM_PER_MILE * 1000.0)
            elif "mile" in u or u in ("mi",):
                pass
            else:
                return _abstain("unrecognized distance unit: %s" % u)

        fare = pd.to_numeric(df["fare_amount"], errors="coerce").astype("float64") * fare_scale
        dist = pd.to_numeric(df["trip_distance"], errors="coerce").astype("float64") * dist_scale

        if fare.isna().mean() > 0.01:
            return _abstain("too many missing/invalid fare values")
        if dist.isna().mean() > 0.01:
            return _abstain("too many missing/invalid distance values")
        if not np.isfinite(fare.dropna()).all() or not np.isfinite(dist.dropna()).all():
            return _abstain("non-finite values in numeric columns")

        # plausibility checks
        med_fare = float(fare.median())
        med_dist = float(dist.median())
        if not (2.0 <= med_fare <= 100.0):
            return _abstain("implausible median fare %.2f (unit issue?)" % med_fare)
        if not (0.2 <= med_dist <= 30.0):
            return _abstain("implausible median trip distance %.2f (unit issue?)" % med_dist)
        if "total_amount" in df.columns:
            tot = pd.to_numeric(df["total_amount"], errors="coerce") * fare_scale
            ok = tot.notna() & fare.notna()
            if ok.sum() > 0 and float(((tot[ok] + 0.01) < fare[ok]).mean()) > 0.2 and \
                    float((fare[ok] > 0).mean()) > 0.5:
                return _abstain("fare_amount inconsistent with total_amount")

        # time coverage
        pu = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        if getattr(pu.dt, "tz", None) is not None:
            pu = pu.dt.tz_convert("America/New_York").dt.tz_localize(None)
        if pu.isna().mean() > 0.01:
            return _abstain("too many invalid pickup timestamps")
        start, end = _parse_period(manifest)
        if start is not None and end is not None:
            inside = (pu >= start) & (pu < end)
            if float((~inside).mean()) > 0.01:
                return _abstain("data period does not match manifest period")

        card, err = _card_mask(df["payment_type"], manifest)
        if err:
            return _abstain(err)
        if pd.Series(df["payment_type"]).isna().mean() > 0.5:
            return _abstain("payment_type mostly missing")

        trip_count = int(n)
        total_fare = float(fare.sum())
        mean_fare = float(fare.mean())
        mean_dist = float(dist.mean())
        card_share = float(np.asarray(card, dtype=bool).sum()) / trip_count

        for v in (total_fare, mean_fare, mean_dist, card_share):
            if not math.isfinite(v):
                return _abstain("non-finite metric")

        return {
            "status": "ok",
            "metrics": {
                "trip_count": trip_count,
                "total_fare": total_fare,
                "mean_fare": mean_fare,
                "mean_trip_distance": mean_dist,
                "card_share": card_share,
            },
            "reason": "",
        }
    except Exception as e:  # never crash the caller
        return _abstain("unexpected error: %s" % type(e).__name__)
