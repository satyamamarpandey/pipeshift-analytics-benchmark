import hashlib
import json
import math
import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REQUIRED_COLUMNS = [
    "record_id",
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "total_amount",
]
NUMERIC_COLUMNS = ["trip_distance", "fare_amount"]
CREDIT_CARD_CODE = 1  # TLC data dictionary: 1 = Credit card
CHECKSUM_KEYS = ("sha256", "sha-256", "checksum_sha256")
ROW_COUNT_KEYS = ("row_count", "num_rows", "record_count", "rows")


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _matches_name(value, base):
    if not isinstance(value, str):
        return False
    norm = value.replace("\\", "/")
    return norm == base or norm.endswith("/" + base)


def _verify_manifest(manifest_path, data_path):
    """Return (ok, reason). Requires the manifest to describe the data file
    and to supply at least one integrity field that can be checked."""
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        return False, f"manifest unreadable: {exc}"

    if not isinstance(manifest, (dict, list)):
        return False, "manifest has unexpected structure"

    base = os.path.basename(data_path)
    candidates = [
        d for d in _walk(manifest)
        if any(_matches_name(v, base) for v in d.values())
    ]
    if len(candidates) != 1:
        return False, "manifest does not uniquely describe the data file"
    entry = candidates[0]

    checks = 0
    for key in CHECKSUM_KEYS:
        if key in entry:
            expected = entry[key]
            if not isinstance(expected, str):
                return False, "manifest checksum is malformed"
            if _sha256_file(data_path).lower() != expected.lower():
                return False, "data file checksum does not match manifest"
            checks += 1
            break

    for key in ROW_COUNT_KEYS:
        if key in entry:
            expected = entry[key]
            if isinstance(expected, bool) or not isinstance(expected, int):
                return False, "manifest row count is malformed"
            checks += 1
            break

    if checks == 0:
        return False, "manifest provides no checksum or row count to verify against"
    return True, ""


def _manifest_row_count(manifest_path, data_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    base = os.path.basename(data_path)
    candidates = [
        d for d in _walk(manifest)
        if any(_matches_name(v, base) for v in d.values())
    ]
    entry = candidates[0]
    for key in ROW_COUNT_KEYS:
        if key in entry:
            return entry[key]
    return None


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        ok, reason = _verify_manifest(manifest_path, data_path)
    except (OSError, ValueError) as exc:
        return _abstain(f"manifest verification failed: {exc}")
    if not ok:
        return _abstain(reason)

    try:
        pf = pq.ParquetFile(data_path)
        present = set(pf.schema_arrow.names)
    except (OSError, pa.ArrowException) as exc:
        return _abstain(f"data file unreadable: {exc}")

    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing:
        return _abstain(f"missing required columns: {', '.join(missing)}")

    try:
        table = pq.read_table(data_path, columns=REQUIRED_COLUMNS)
    except (OSError, pa.ArrowException) as exc:
        return _abstain(f"data file could not be read: {exc}")

    n = table.num_rows
    if n == 0:
        return _abstain("no trip records in data file")

    expected_rows = _manifest_row_count(manifest_path, data_path)
    if expected_rows is not None and expected_rows != n:
        return _abstain(
            f"row count mismatch: manifest says {expected_rows}, file has {n}"
        )

    for col in REQUIRED_COLUMNS:
        if table.column(col).null_count > 0:
            return _abstain(f"null values present in column '{col}'")

    if pc.count_distinct(table.column("record_id"), mode="all").as_py() != n:
        return _abstain("duplicate record_id values; trip count is ambiguous")

    numeric = {}
    for col in NUMERIC_COLUMNS:
        try:
            arr = pc.cast(table.column(col), pa.float64())
        except (pa.ArrowException, ValueError):
            return _abstain(f"column '{col}' is not numeric")
        values = arr.to_pylist()
        if any(v is None or not math.isfinite(v) for v in values):
            return _abstain(f"non-finite values in column '{col}'")
        numeric[col] = values

    if any(v < 0 for v in numeric["trip_distance"]):
        return _abstain("negative trip_distance values present")

    try:
        pay = pc.cast(table.column("payment_type"), pa.int64()).to_pylist()
    except (pa.ArrowException, ValueError):
        return _abstain("payment_type is not integer-coded")
    if any(p is None for p in pay):
        return _abstain("payment_type contains non-integer values")

    fares = numeric["fare_amount"]
    dists = numeric["trip_distance"]
    total_fare = math.fsum(fares)
    mean_fare = total_fare / n
    mean_trip_distance = math.fsum(dists) / n
    card_trips = sum(1 for p in pay if p == CREDIT_CARD_CODE)
    card_share = card_trips / n

    return {
        "status": "ok",
        "metrics": {
            "trip_count": n,
            "total_fare": total_fare,
            "mean_fare": mean_fare,
            "mean_trip_distance": mean_trip_distance,
            "card_share": card_share,
        },
        "reason": "",
    }
