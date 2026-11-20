import hashlib
import json
import math
import numbers

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Columns needed to compute the metrics.
REQUIRED_COLUMNS = (
    "record_id",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "fare_amount",
    "trip_distance",
    "payment_type",
)

# TLC data dictionary: payment_type 1 = credit card.
CREDIT_CARD_CODE = 1


def _abstain(reason: str) -> dict:
    return {"status": "abstain", "metrics": None, "reason": reason}


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_numeric_type(t: pa.DataType) -> bool:
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)


def _analyze(data_path: str, manifest_path: str) -> dict:
    # ---- 1. Manifest: must parse and carry at least one verifiable integrity field.
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        return _abstain("manifest is not a JSON object")

    expected_rows = manifest.get("row_count", manifest.get("record_count"))
    expected_sha = manifest.get("sha256")
    if expected_rows is None and expected_sha is None:
        return _abstain("manifest provides no verifiable integrity field (row_count or sha256)")

    if expected_sha is not None:
        if not isinstance(expected_sha, str):
            return _abstain("manifest sha256 is not a string")
        if _sha256_of_file(data_path).lower() != expected_sha.strip().lower():
            return _abstain("data file sha256 does not match manifest")

    # ---- 2. Parquet metadata and schema.
    pf = pq.ParquetFile(data_path)
    n_rows = pf.metadata.num_rows

    if expected_rows is not None:
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, numbers.Integral):
            return _abstain("manifest row count is not an integer")
        if int(expected_rows) != n_rows:
            return _abstain("row count in data file does not match manifest")

    if n_rows == 0:
        return _abstain("data file contains no trips")

    schema_names = set(pf.schema_arrow.names)
    missing = [c for c in REQUIRED_COLUMNS if c not in schema_names]
    if missing:
        return _abstain(f"missing required columns: {', '.join(missing)}")

    table = pf.read(columns=list(REQUIRED_COLUMNS))

    # ---- 3. Data quality checks. Abstain rather than guess.
    for col in REQUIRED_COLUMNS:
        if table.column(col).null_count > 0:
            return _abstain(f"column {col} contains null values")

    if pc.count_distinct(table.column("record_id")).as_py() != n_rows:
        return _abstain("record_id is not unique; trips would be double-counted")

    fare_col = table.column("fare_amount")
    dist_col = table.column("trip_distance")
    pay_col = table.column("payment_type")

    if not _is_numeric_type(fare_col.type):
        return _abstain("fare_amount is not numeric")
    if not _is_numeric_type(dist_col.type):
        return _abstain("trip_distance is not numeric")
    if not (pa.types.is_integer(pay_col.type) or pa.types.is_floating(pay_col.type)):
        return _abstain("payment_type is not numeric")

    fares = pc.cast(fare_col, pa.float64())
    distances = pc.cast(dist_col, pa.float64())

    if not pc.all(pc.is_finite(fares)).as_py():
        return _abstain("fare_amount contains non-finite values")
    if not pc.all(pc.is_finite(distances)).as_py():
        return _abstain("trip_distance contains non-finite values")

    # ---- 4. Metrics. Exact float summation via math.fsum.
    fare_values = fares.to_pylist()
    distance_values = distances.to_pylist()

    trip_count = n_rows
    total_fare = math.fsum(fare_values)
    mean_fare = total_fare / trip_count
    mean_trip_distance = math.fsum(distance_values) / trip_count

    credit_card_trips = int(
        pc.sum(pc.equal(pay_col, pa.scalar(CREDIT_CARD_CODE, type=pay_col.type))).as_py() or 0
    )
    card_share = credit_card_trips / trip_count

    return {
        "status": "ok",
        "metrics": {
            "trip_count": trip_count,
            "total_fare": total_fare,
            "mean_fare": mean_fare,
            "mean_trip_distance": mean_trip_distance,
            "card_share": card_share,
        },
        "reason": "",
    }


def analyze(data_path: str, manifest_path: str) -> dict:
    """Compute overall NYC yellow taxi trip metrics from a Parquet snapshot.

    Returns {"status": "ok", "metrics": {...}, "reason": ""} on success, or
    {"status": "abstain", "metrics": None, "reason": "..."} when the result
    cannot be trusted. Never prints, prompts, or writes files.
    """
    try:
        return _analyze(data_path, manifest_path)
    except Exception as exc:  # any read/parse failure means we cannot be sure
        return _abstain(f"could not process inputs: {type(exc).__name__}")
