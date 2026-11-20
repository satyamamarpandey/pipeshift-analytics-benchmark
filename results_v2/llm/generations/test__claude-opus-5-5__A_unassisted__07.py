"""Monthly summary of NYC TLC yellow taxi trips, with defensive validation against a manifest."""

import hashlib
import json
import math
import os
import re

import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "record_id", "tpep_pickup_datetime", "trip_distance",
    "payment_type", "fare_amount",
]

BAD_STATUS_WORDS = (
    "retract", "deprecat", "invalid", "corrupt", "withdrawn", "superseded",
    "do not use", "do_not_use", "incomplete", "partial", "draft", "embargo",
    "quarantin", "failed", "error",
)


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": str(reason)[:300]}


def _walk(obj, path=()):
    """Yield (path, key, value) for every dict entry, recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, str(k), v
            yield from _walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, path + (str(i),))


def _find(manifest, names, types=None):
    names = {n.lower() for n in names}
    for _, k, v in _walk(manifest):
        if k.lower() in names and (types is None or isinstance(v, types)):
            return v
    return None


def _to_int(v):
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, str) and re.fullmatch(r"\s*\d+\s*", v):
            return int(v)
    except Exception:
        pass
    return None


def _column_unit(manifest, col):
    """Find a unit/currency string declared for a column, if any."""
    col_l = col.lower()
    unit_keys = ("unit", "units", "currency", "uom")
    for _, k, v in _walk(manifest):
        if k.lower() == col_l:
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                for uk in unit_keys:
                    for kk, vv in v.items():
                        if str(kk).lower() == uk and isinstance(vv, str):
                            return vv
    for _, _, v in _walk(manifest):
        if isinstance(v, dict):
            nm = None
            for nk in ("name", "column", "field"):
                if nk in v and isinstance(v[nk], str):
                    nm = v[nk]
                    break
            if nm is not None and nm.lower() == col_l:
                for kk, vv in v.items():
                    if str(kk).lower() in unit_keys and isinstance(vv, str):
                        return vv
    return None


def _parse_month(manifest):
    for key in ("month", "period", "year_month", "data_month", "billing_month",
                "coverage_month", "reporting_period", "snapshot_month"):
        v = _find(manifest, [key])
        if isinstance(v, str):
            m = re.search(r"(\d{4})[-_/]?(\d{2})", v)
            if m and 1 <= int(m.group(2)) <= 12:
                return int(m.group(1)), int(m.group(2))
        if isinstance(v, dict):
            y, mo = _to_int(v.get("year")), _to_int(v.get("month"))
            if y and mo and 1 <= mo <= 12:
                return y, mo
    y = _to_int(_find(manifest, ["year"]))
    mo = _to_int(_find(manifest, ["month"]))
    if y and mo and 1 <= mo <= 12:
        return y, mo
    for key in ("start_date", "period_start", "start", "from", "coverage_start"):
        v = _find(manifest, [key], str)
        if v:
            m = re.search(r"(\d{4})-(\d{2})", v)
            if m and 1 <= int(m.group(2)) <= 12:
                return int(m.group(1)), int(m.group(2))
    return None


def _check_status(manifest):
    for _, k, v in _walk(manifest):
        kl = k.lower()
        if kl in ("status", "state", "quality", "data_quality", "validity",
                  "release_status", "publication_status") and isinstance(v, str):
            vl = v.lower()
            if any(w in vl for w in BAD_STATUS_WORDS):
                return f"manifest {k}={v!r}"
        if kl in ("retracted", "deprecated", "invalid", "corrupted", "superseded",
                  "do_not_use", "incomplete", "is_partial", "partial") and v is True:
            return f"manifest flag {k}=true"
        if kl in ("valid", "is_valid", "complete", "is_complete", "validated") and v is False:
            return f"manifest flag {k}=false"
    return None


def _check_checksum(manifest, data_path):
    algos = {"sha256": hashlib.sha256, "sha1": hashlib.sha1, "md5": hashlib.md5,
             "sha512": hashlib.sha512}
    wanted = []
    for _, k, v in _walk(manifest):
        kl = k.lower().replace("-", "").replace("_", "")
        if not isinstance(v, str):
            continue
        val = v.strip().lower()
        algo = None
        for a in algos:
            if a in kl:
                algo = a
        if algo is None and ("checksum" in kl or "hash" in kl or "digest" in kl):
            m = re.match(r"(sha256|sha1|md5|sha512)[:=]", val)
            if m:
                algo = m.group(1)
            else:
                algo = {32: "md5", 40: "sha1", 64: "sha256", 128: "sha512"}.get(
                    len(re.sub(r"^[a-z0-9]+[:=]", "", val)))
        if algo:
            hexval = re.sub(r"^[a-z0-9]+[:=]", "", val)
            if re.fullmatch(r"[0-9a-f]+", hexval):
                wanted.append((algo, hexval))
    if not wanted:
        return None
    hashers = {a: algos[a]() for a, _ in wanted}
    with open(data_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            for h in hashers.values():
                h.update(chunk)
    for algo, hexval in wanted:
        if hashers[algo].hexdigest() != hexval:
            return f"{algo} checksum mismatch"
    return None


def _check_size(manifest, data_path):
    for key in ("size_bytes", "file_size", "file_size_bytes", "bytes", "content_length"):
        v = _to_int(_find(manifest, [key]))
        if v is not None:
            actual = os.path.getsize(data_path)
            if v != actual:
                return f"file size {actual} != manifest {v}"
            return None
    return None


def _money_factor(unit):
    if unit is None:
        return 1.0, None
    u = unit.strip().lower()
    if u in ("", "usd", "us$", "$", "dollar", "dollars", "us dollars", "usd ($)"):
        return 1.0, None
    if "cent" in u:
        return 0.01, None
    if "usd" in u or "dollar" in u:
        if "thousand" in u or u.startswith("k"):
            return 1000.0, None
        return 1.0, None
    return None, f"fare currency/unit {unit!r} not USD"


def _distance_factor(unit):
    if unit is None:
        return 1.0, None
    u = unit.strip().lower()
    if u in ("", "mi", "mile", "miles"):
        return 1.0, None
    if "mile" in u and "kilo" not in u:
        return 1.0, None
    if u in ("km", "kms", "kilometer", "kilometers", "kilometre", "kilometres") or "kilomet" in u:
        return 1.0 / 1.609344, None
    if u in ("m", "meter", "meters", "metre", "metres"):
        return 1.0 / 1609.344, None
    if u in ("ft", "feet", "foot"):
        return 1.0 / 5280.0, None
    return None, f"distance unit {unit!r} not understood"


def _to_ny_naive(series, manifest):
    s = series
    if not pd.api.types.is_datetime64_any_dtype(s):
        if pd.api.types.is_numeric_dtype(s):
            mx = pd.to_numeric(s, errors="coerce").abs().max()
            unit = "s"
            if pd.notna(mx):
                if mx > 1e17:
                    unit = "ns"
                elif mx > 1e14:
                    unit = "us"
                elif mx > 1e11:
                    unit = "ms"
            s = pd.to_datetime(s, unit=unit, errors="coerce", utc=True)
        else:
            s = pd.to_datetime(s, errors="coerce", utc=False, format="mixed")
    tz = getattr(s.dt, "tz", None)
    if tz is not None:
        return s.dt.tz_convert("America/New_York").dt.tz_localize(None)
    declared = _find(manifest, ["timezone", "time_zone", "tz", "datetime_timezone",
                                "timestamp_timezone"], str)
    if declared and declared.strip():
        d = declared.strip()
        if d.lower() not in ("america/new_york", "us/eastern", "est", "edt",
                             "local", "eastern", "et", "nyc local", "new york"):
            try:
                return (s.dt.tz_localize(d, ambiguous="NaT", nonexistent="NaT")
                        .dt.tz_convert("America/New_York").dt.tz_localize(None))
            except Exception:
                pass
    return s


def _card_mask(pt, manifest):
    mapping = _find(manifest, ["payment_type", "payment_types", "payment_type_codes",
                               "payment_type_mapping"], dict)
    card_codes = set()
    if isinstance(mapping, dict):
        for k, v in mapping.items():
            if isinstance(v, str) and "credit" in v.lower():
                card_codes.add(str(k).strip().lower())
            elif isinstance(k, str) and "credit" in k.lower() and not isinstance(v, (dict, list)):
                card_codes.add(str(v).strip().lower())
    if pd.api.types.is_numeric_dtype(pt):
        nums = pd.to_numeric(pt, errors="coerce")
        codes = set()
        for c in card_codes:
            try:
                codes.add(float(c))
            except ValueError:
                pass
        if not codes:
            codes = {1.0}
        return nums.isin(list(codes)).to_numpy()
    s = pt.astype("string").str.strip().str.lower()
    if not card_codes:
        card_codes = {"1", "1.0", "crd", "credit", "credit card", "card", "credit_card"}
    m = s.isin(list(card_codes)) | s.str.contains("credit", na=False)
    return m.fillna(False).to_numpy(dtype=bool)


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        try:
            with open(manifest_path, "r", encoding="utf-8-sig") as fh:
                manifest = json.load(fh)
        except Exception as e:
            return _abstain(f"cannot read manifest: {type(e).__name__}")
        if not isinstance(manifest, (dict, list)):
            return _abstain("manifest is not a JSON object")

        reason = _check_status(manifest)
        if reason:
            return _abstain(reason)

        if not os.path.isfile(data_path):
            return _abstain("data file missing")
        reason = _check_size(manifest, data_path) or _check_checksum(manifest, data_path)
        if reason:
            return _abstain(reason)

        try:
            df = pd.read_parquet(data_path)
        except Exception as e:
            return _abstain(f"cannot read parquet: {type(e).__name__}")

        # Case-insensitive column resolution
        lower_map = {str(c).lower(): c for c in df.columns}
        rename = {}
        for c in REQUIRED_COLS + ["tpep_dropoff_datetime", "total_amount", "tip_amount"]:
            if c not in df.columns and c.lower() in lower_map:
                rename[lower_map[c.lower()]] = c
        if rename:
            df = df.rename(columns=rename)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            return _abstain(f"missing columns: {missing}")

        raw_n = len(df)
        if raw_n == 0:
            return _abstain("no rows in data")

        # Duplicates
        df = df.drop_duplicates()
        if df["record_id"].isna().any():
            return _abstain("null record_id values")
        if df["record_id"].duplicated().any():
            return _abstain("conflicting duplicate record_id values")
        dedup_n = len(df)

        expected = None
        for key in ("row_count", "num_rows", "n_rows", "record_count", "num_records",
                    "rows", "expected_rows", "expected_row_count", "total_rows",
                    "n_records", "count"):
            expected = _to_int(_find(manifest, [key]))
            if expected is not None:
                break
        if expected is not None and expected not in (raw_n, dedup_n):
            return _abstain(f"row count {raw_n} (dedup {dedup_n}) != manifest {expected}")

        # Units
        f_factor, err = _money_factor(_column_unit(manifest, "fare_amount")
                                      or _find(manifest, ["currency"], str))
        if err:
            return _abstain(err)
        d_factor, err = _distance_factor(_column_unit(manifest, "trip_distance")
                                         or _find(manifest, ["distance_unit", "distance_units"], str))
        if err:
            return _abstain(err)

        fare = pd.to_numeric(df["fare_amount"], errors="coerce").astype("float64") * f_factor
        dist = pd.to_numeric(df["trip_distance"], errors="coerce").astype("float64") * d_factor

        # Month filtering
        pickup = _to_ny_naive(df["tpep_pickup_datetime"], manifest)
        if pickup.isna().mean() > 0.01:
            return _abstain("unparseable pickup timestamps")
        ym = _parse_month(manifest)
        valid_pick = pickup.dropna()
        if ym is None:
            if valid_pick.empty:
                return _abstain("no valid pickup timestamps")
            per = valid_pick.dt.to_period("M").value_counts()
            p = per.index[0]
            ym = (p.year, p.month)
        start = pd.Timestamp(year=ym[0], month=ym[1], day=1)
        end = start + pd.offsets.MonthBegin(1)
        in_month = ((pickup >= start) & (pickup < end)).fillna(False).to_numpy(dtype=bool)
        frac_in = in_month.mean()
        if frac_in < 0.5:
            return _abstain(f"only {frac_in:.1%} of trips fall in {ym[0]}-{ym[1]:02d}")

        fare_m = fare.to_numpy()[in_month]
        dist_m = dist.to_numpy()[in_month]
        card_m = _card_mask(df["payment_type"], manifest)[in_month]
        n = int(in_month.sum())
        if n == 0:
            return _abstain("no trips in target month")

        if np.isnan(fare_m).mean() > 0.01:
            return _abstain("too many missing fare values")
        if np.isnan(dist_m).mean() > 0.05:
            return _abstain("too many missing trip distances")
        if np.isinf(fare_m).any() or np.isinf(dist_m).any():
            return _abstain("infinite values present")

        fare_valid = fare_m[~np.isnan(fare_m)]
        dist_valid = dist_m[~np.isnan(dist_m)]
        if fare_valid.size == 0 or dist_valid.size == 0:
            return _abstain("no valid fare/distance values")

        # Plausibility checks (catch unit errors not declared in manifest)
        med_fare = float(np.median(fare_valid))
        med_dist = float(np.median(dist_valid))
        if not (1.0 <= med_fare <= 200.0):
            return _abstain(f"implausible median fare {med_fare}")
        if not (0.1 <= med_dist <= 50.0):
            return _abstain(f"implausible median distance {med_dist}")

        total_fare = float(fare_valid.sum())
        metrics = {
            "trip_count": n,
            "total_fare": total_fare,
            "mean_fare": float(fare_valid.mean()),
            "mean_trip_distance": float(dist_valid.mean()),
            "card_share": float(card_m.sum()) / n,
        }
        for k, v in metrics.items():
            if isinstance(v, float) and not math.isfinite(v):
                return _abstain(f"non-finite metric {k}")
        return {"status": "ok", "metrics": metrics, "reason": ""}
    except Exception as e:
        return _abstain(f"unexpected error: {type(e).__name__}")
