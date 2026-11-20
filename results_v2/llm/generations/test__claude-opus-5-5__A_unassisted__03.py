import hashlib
import json
import math
import os
import re

import numpy as np
import pandas as pd

REQUIRED = ["tpep_pickup_datetime", "fare_amount", "trip_distance", "payment_type"]
BAD_STATUS = {
    "deprecated", "retracted", "withdrawn", "invalid", "incomplete", "partial",
    "superseded", "corrupt", "corrupted", "draft", "failed", "error", "embargoed",
    "do_not_use", "do not use", "revoked", "obsolete", "preliminary", "quarantined",
}


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


def _leaf(path):
    return re.split(r"[.\[]", path)[-1].rstrip("]")


def _find_scalar(flat, names):
    for p, v in flat:
        if _leaf(p) in names and isinstance(v, (str, int, float)) and not isinstance(v, bool):
            return p, v
    return None, None


def _column_specs(manifest):
    """Return dict column_name -> spec dict from manifest schema/columns sections."""
    specs = {}

    def visit(o):
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                if kl in ("columns", "schema", "fields") :
                    if isinstance(v, dict):
                        for ck, cv in v.items():
                            if isinstance(cv, dict):
                                specs[str(ck).lower()] = {str(a).lower(): b for a, b in cv.items()}
                            elif isinstance(cv, str):
                                specs.setdefault(str(ck).lower(), {"type": cv})
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                nm = item.get("name") or item.get("column") or item.get("field")
                                if nm is not None:
                                    specs[str(nm).lower()] = {str(a).lower(): b for a, b in item.items()}
                visit(v)
        elif isinstance(o, list):
            for v in o:
                visit(v)

    visit(manifest)
    return specs


def _unit_of(specs, flat, col, kind):
    spec = specs.get(col, {})
    for key in ("unit", "units", "currency", "uom", "measure"):
        if key in spec and isinstance(spec[key], str):
            return spec[key].strip().lower()
    if kind == "fare":
        cands = {"fare_amount_unit", "fare_unit", "fare_units", "fare_currency", "currency",
                 "monetary_unit", "amount_unit", "money_unit", "currency_unit", "amount_units"}
    else:
        cands = {"trip_distance_unit", "distance_unit", "distance_units", "trip_distance_units"}
    for p, v in flat:
        if isinstance(v, str) and _leaf(p) in cands:
            return v.strip().lower()
    # generic "units": {"fare_amount": "...", "trip_distance": "..."}
    for p, v in flat:
        if isinstance(v, str) and _leaf(p) == col and re.search(r"units?$", p.rsplit(".", 1)[0] if "." in p else ""):
            return v.strip().lower()
    return None


def _fare_factor(unit):
    if unit is None:
        return 1.0
    u = unit.replace("_", " ").replace("-", " ").strip()
    if "cent" in u:
        return 0.01
    if u in ("usd", "us dollars", "us dollar", "dollars", "dollar", "$", "usd dollars", "us$"):
        return 1.0
    if "usd" in u and "milli" not in u and "thousand" not in u:
        return 1.0
    return None


def _dist_factor(unit):
    if unit is None:
        return 1.0
    u = unit.replace("_", " ").replace("-", " ").strip()
    if u in ("mi", "mile", "miles", "statute miles", "statute mile"):
        return 1.0
    if u in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
        return 1.0 / 1.609344
    if u in ("m", "meter", "meters", "metre", "metres"):
        return 1.0 / 1609.344
    if u in ("ft", "feet", "foot"):
        return 1.0 / 5280.0
    return None


def _parse_period(flat):
    for p, v in flat:
        lf = _leaf(p)
        if lf in ("month", "period", "year_month", "yearmonth", "data_month", "coverage_month",
                  "reporting_period", "coverage", "snapshot_month", "partition") and isinstance(v, str):
            m = re.search(r"(\d{4})[-_/]?(\d{2})", v)
            if m and 1 <= int(m.group(2)) <= 12:
                return int(m.group(1)), int(m.group(2))
    y = m_ = None
    for p, v in flat:
        lf = _leaf(p)
        if lf == "year" and isinstance(v, (int, str)) and str(v).isdigit():
            y = int(v)
        if lf == "month" and isinstance(v, (int, str)) and str(v).isdigit():
            m_ = int(v)
    if y and m_ and 1 <= m_ <= 12:
        return y, m_
    return None


def _card_codes(flat):
    codes = set()
    for p, v in flat:
        if "payment" in p and isinstance(v, dict):
            for k, lab in v.items():
                if isinstance(lab, str) and "credit" in lab.lower():
                    codes.add(str(k).strip())
                elif isinstance(k, str) and "credit" in k.lower() and isinstance(lab, (int, str)):
                    codes.add(str(lab).strip())
    return codes or {"1"}


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return _abstain(f"manifest unreadable: {type(e).__name__}")
    if not isinstance(manifest, (dict, list)):
        return _abstain("manifest has unexpected structure")
    flat = _flatten(manifest)

    # dataset status flags
    for p, v in flat:
        lf = _leaf(p)
        if lf in ("status", "state", "quality", "quality_status", "validation", "validation_status",
                  "release_status", "data_status") and isinstance(v, str):
            if v.strip().lower().replace("-", "_") in BAD_STATUS or v.strip().lower() in BAD_STATUS:
                return _abstain(f"manifest marks dataset as {v}")
        if lf in ("deprecated", "retracted", "withdrawn", "incomplete", "partial", "superseded",
                  "do_not_use", "corrupted", "is_partial", "is_incomplete") and v is True:
            return _abstain(f"manifest flags dataset as {lf}")
        if lf in ("valid", "is_valid", "complete", "is_complete", "validated") and v is False:
            return _abstain(f"manifest flags dataset {lf}=false")

    # checksum
    if not os.path.isfile(data_path):
        return _abstain("data file missing")
    for p, v in flat:
        lf = _leaf(p)
        if not isinstance(v, str):
            continue
        algo = None
        val = v.strip().lower()
        if ":" in val and val.split(":", 1)[0] in ("sha256", "md5", "sha1", "sha512"):
            algo, val = val.split(":", 1)
        elif lf in ("sha256", "md5", "sha1", "sha512"):
            algo = lf
        elif lf in ("checksum", "hash", "file_hash", "digest", "file_checksum"):
            algo = {32: "md5", 40: "sha1", 64: "sha256", 128: "sha512"}.get(len(val))
        if algo and re.fullmatch(r"[0-9a-f]+", val):
            h = hashlib.new(algo)
            try:
                with open(data_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
            except Exception:
                return _abstain("data file unreadable")
            if h.hexdigest() != val:
                return _abstain(f"{algo} checksum mismatch with manifest")
            break

    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        return _abstain(f"data unreadable: {type(e).__name__}")
    df.columns = [str(c) for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    missing = [c for c in REQUIRED if c.lower() not in lower_map]
    if missing:
        return _abstain(f"missing required columns: {missing}")
    df = df.rename(columns={lower_map[c.lower()]: c for c in REQUIRED + ["record_id"] if c.lower() in lower_map})

    specs = _column_specs(manifest)
    ff = _fare_factor(_unit_of(specs, flat, "fare_amount", "fare"))
    if ff is None:
        return _abstain("unsupported fare unit/currency in manifest")
    df_ = _dist_factor(_unit_of(specs, flat, "trip_distance", "distance"))
    if df_ is None:
        return _abstain("unsupported distance unit in manifest")

    n_raw = len(df)
    if n_raw == 0:
        return _abstain("no records")

    # deduplicate
    if "record_id" in df.columns:
        try:
            df = df.drop_duplicates()
        except TypeError:
            pass
        if df["record_id"].duplicated().any():
            return _abstain("conflicting duplicate record_id values")
    n_dedup = len(df)

    fare = pd.to_numeric(df["fare_amount"], errors="coerce")
    dist = pd.to_numeric(df["trip_distance"], errors="coerce")
    if fare.isna().mean() > 0.05 or dist.isna().mean() > 0.05:
        return _abstain("too many non-numeric fare or distance values")

    pickup = df["tpep_pickup_datetime"]
    if not pd.api.types.is_datetime64_any_dtype(pickup):
        pickup = pd.to_datetime(pickup, errors="coerce")
    period = _parse_period(flat)
    mask = pd.Series(True, index=df.index)
    if period is not None:
        mask = (pickup.dt.year == period[0]) & (pickup.dt.month == period[1])
        if mask.mean() < 0.5:
            return _abstain("most records fall outside the manifest period")
    n_filt = int(mask.sum())

    _, expected = _find_scalar(flat, {"row_count", "num_rows", "rows", "record_count", "n_rows",
                                      "expected_rows", "nrows", "num_records", "n_records",
                                      "total_rows", "row_cnt", "records"})
    if expected is not None:
        try:
            exp = int(float(expected))
        except Exception:
            exp = None
        if exp is not None and exp not in (n_raw, n_dedup, n_filt):
            return _abstain(f"row count mismatch: manifest {exp}, data {n_raw}")

    df = df[mask]
    fare = fare[mask] * ff
    dist = dist[mask] * df_
    if len(df) == 0:
        return _abstain("no records in period")

    med_fare = float(np.nanmedian(fare.to_numpy(dtype=float)))
    med_dist = float(np.nanmedian(dist.to_numpy(dtype=float)))
    if not (0 < med_fare < 150):
        return _abstain("implausible fare scale; units may be wrong")
    if not (0 <= med_dist < 50):
        return _abstain("implausible distance scale; units may be wrong")

    pt = df["payment_type"]
    codes = _card_codes(flat)
    if pd.api.types.is_numeric_dtype(pt):
        num = pd.to_numeric(pt, errors="coerce")
        code_nums = set()
        for c in codes:
            try:
                code_nums.add(float(c))
            except ValueError:
                pass
        is_card = num.isin(list(code_nums))
    else:
        s = pt.astype(str).str.strip().str.lower()
        is_card = s.isin({c.lower() for c in codes}) | s.str.contains("credit", na=False)
        num = pd.to_numeric(s, errors="coerce")
        is_card = is_card | num.isin([float(c) for c in codes if re.fullmatch(r"\d+(\.0+)?", c)])

    total_fare = float(fare.sum(skipna=True))
    mean_fare = float(fare.mean(skipna=True))
    mean_dist = float(dist.mean(skipna=True))
    card_share = float(is_card.sum()) / float(len(df))
    vals = [total_fare, mean_fare, mean_dist, card_share]
    if any(not math.isfinite(v) for v in vals):
        return _abstain("non-finite metric values")

    return {
        "status": "ok",
        "metrics": {
            "trip_count": int(len(df)),
            "total_fare": total_fare,
            "mean_fare": mean_fare,
            "mean_trip_distance": mean_dist,
            "card_share": card_share,
        },
        "reason": "",
    }
