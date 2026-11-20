import hashlib
import json
import math

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# TLC data dictionary: payment_type 1 = Credit card.
CREDIT_CARD_CODE = 1
REQUIRED_COLUMNS = ("record_id", "trip_distance", "fare_amount", "payment_type")
ROW_COUNT_KEYS = ("row_count", "num_rows", "record_count", "trip_count")
CHECKSUM_KEYS = ("sha256", "sha256_checksum")


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError("manifest root is not a JSON object")
    return manifest


def _manifest_value(manifest, keys):
    for key in keys:
        if key in manifest:
            return key, manifest[key]
    return None, None


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        manifest = _load_manifest(manifest_path)
    except Exception as exc:
        return _abstain(f"manifest could not be read: {type(exc).__name__}")

    # The manifest must provide something we can verify the snapshot against.
    checksum_key, expected_sha = _manifest_value(manifest, CHECKSUM_KEYS)
    count_key, expected_rows = _manifest_value(manifest, ROW_COUNT_KEYS)
    if checksum_key is None and count_key is None:
        return _abstain("manifest provides no checksum or row count to verify the data against")

    try:
        parquet_file = pq.ParquetFile(data_path)
    except Exception as exc:
        return _abstain(f"data file could not be opened as Parquet: {type(exc).__name__}")

    schema_names = set(parquet_file.schema_arrow.names)
    missing = [c for c in REQUIRED_COLUMNS if c not in schema_names]
    if missing:
        return _abstain(f"data is missing required columns: {', '.join(missing)}")

    if checksum_key is not None:
        try:
            actual_sha = _sha256_file(data_path)
        except Exception as exc:
            return _abstain(f"data file could not be hashed: {type(exc).__name__}")
        if not isinstance(expected_sha, str) or actual_sha != expected_sha.lower():
            return _abstain("data checksum does not match manifest")

    num_rows = parquet_file.metadata.num_rows
    if count_key is not None:
        if not isinstance(expected_rows, int) or isinstance(expected_rows, bool):
            return _abstain("manifest row count is not an integer")
        if num_rows != expected_rows:
            return _abstain("data row count does not match manifest")

    if num_rows == 0:
        return _abstain("data contains no rows")

    try:
        table = parquet_file.read(columns=list(REQUIRED_COLUMNS))
    except Exception as exc:
        return _abstain(f"data could not be read: {type(exc).__name__}")

    try:
        for name in REQUIRED_COLUMNS:
            if table.column(name).null_count != 0:
                return _abstain(f"column {name} contains nulls")

        record_ids = table.column("record_id")
        if pc.count_distinct(record_ids, mode="all").as_py() != num_rows:
            return _abstain("record_id is not unique; duplicate trips would bias the metrics")

        fares = pc.cast(table.column("fare_amount"), pa.float64())
        distances = pc.cast(table.column("trip_distance"), pa.float64())
        payment = pc.cast(table.column("payment_type"), pa.int64())

        if not (pc.all(pc.is_finite(fares)).as_py() and pc.all(pc.is_finite(distances)).as_py()):
            return _abstain("fare_amount or trip_distance contains non-finite values")

        trip_count = table.num_rows
        total_fare = pc.sum(fares).as_py()
        mean_fare = total_fare / trip_count
        mean_trip_distance = pc.sum(distances).as_py() / trip_count

        card_mask = pc.equal(payment, pa.scalar(CREDIT_CARD_CODE, pa.int64()))
        card_trips = pc.sum(pc.cast(card_mask, pa.int64())).as_py() or 0
        card_share = card_trips / trip_count
    except Exception as exc:
        return _abstain(f"metric computation failed: {type(exc).__name__}")

    if not all(math.isfinite(v) for v in (total_fare, mean_fare, mean_trip_distance, card_share)):
        return _abstain("computed metrics are not finite")

    return {
        "status": "ok",
        "metrics": {
            "trip_count": int(trip_count),
            "total_fare": float(total_fare),
            "mean_fare": float(mean_fare),
            "mean_trip_distance": float(mean_trip_distance),
            "card_share": float(card_share),
        },
        "reason": "",
    }
