import hashlib
import json
import math
import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REQUIRED_COLUMNS = (
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
)
METRIC_COLUMNS = ("record_id", "trip_distance", "payment_type", "fare_amount")
CREDIT_CARD_CODE = 1  # NYC TLC data dictionary: payment_type 1 = Credit card


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not a JSON object")
    return manifest


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_numeric(t):
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)


def _is_text(t):
    return pa.types.is_string(t) or pa.types.is_large_string(t)


def _check_manifest_integrity(manifest, data_path, num_rows):
    """Return an abstain reason string, or None if the data matches the manifest."""
    expected_sha = manifest.get("sha256")
    expected_rows = None
    for key in ("row_count", "record_count", "num_rows"):
        if key in manifest:
            expected_rows = manifest[key]
            break

    if expected_sha is None and expected_rows is None:
        return "manifest provides no integrity field (sha256 or row count) to verify data"

    if expected_sha is not None:
        if not isinstance(expected_sha, str):
            return "manifest sha256 field is not a string"
        if _sha256_file(data_path) != expected_sha.strip().lower():
            return "data file sha256 does not match manifest"

    if expected_rows is not None:
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
            return "manifest row count is not an integer"
        if num_rows != expected_rows:
            return "row count does not match manifest"

    return None


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        if not os.path.isfile(data_path):
            return _abstain("data file not found")
        if not os.path.isfile(manifest_path):
            return _abstain("manifest file not found")

        try:
            manifest = _load_manifest(manifest_path)
        except Exception:
            return _abstain("manifest is missing or not valid JSON")

        try:
            schema = pq.read_schema(data_path)
        except Exception:
            return _abstain("data file is not a readable Parquet file")

        present = set(schema.names)
        missing = [c for c in REQUIRED_COLUMNS if c not in present]
        if missing:
            return _abstain("missing required columns: " + ", ".join(missing))

        field_types = {f.name: f.type for f in schema}
        for col in ("fare_amount", "trip_distance"):
            if not _is_numeric(field_types[col]):
                return _abstain(f"column {col} is not numeric")
        if not (_is_numeric(field_types["payment_type"]) or _is_text(field_types["payment_type"])):
            return _abstain("column payment_type has an unsupported type")
        if not (_is_text(field_types["record_id"]) or _is_numeric(field_types["record_id"])):
            return _abstain("column record_id has an unsupported type")

        table = pq.read_table(data_path, columns=list(METRIC_COLUMNS))
        num_rows = table.num_rows

        reason = _check_manifest_integrity(manifest, data_path, num_rows)
        if reason is not None:
            return _abstain(reason)

        if num_rows == 0:
            return _abstain("no trip records in data")

        for col in METRIC_COLUMNS:
            if table.column(col).null_count > 0:
                return _abstain(f"column {col} contains null values")

        record_ids = table.column("record_id")
        if pc.count_distinct(record_ids).as_py() != num_rows:
            return _abstain("record_id is not unique; duplicate trips make counts untrustworthy")

        fares_arr = pc.cast(table.column("fare_amount"), pa.float64())
        dist_arr = pc.cast(table.column("trip_distance"), pa.float64())
        if not pc.all(pc.is_finite(fares_arr)).as_py():
            return _abstain("fare_amount contains non-finite values")
        if not pc.all(pc.is_finite(dist_arr)).as_py():
            return _abstain("trip_distance contains non-finite values")

        fares = fares_arr.to_pylist()
        distances = dist_arr.to_pylist()

        trip_count = num_rows
        total_fare = math.fsum(fares)
        mean_fare = total_fare / trip_count
        mean_trip_distance = math.fsum(distances) / trip_count

        payment = table.column("payment_type")
        if _is_numeric(payment.type):
            pay_num = pc.cast(payment, pa.float64())
            if not pc.all(pc.is_finite(pay_num)).as_py():
                return _abstain("payment_type contains non-finite values")
            mask = pc.equal(pay_num, float(CREDIT_CARD_CODE))
        else:
            lowered = pc.utf8_lower(pc.utf8_trim_whitespace(payment))
            mask = pc.or_(
                pc.equal(lowered, "credit card"),
                pc.equal(lowered, str(CREDIT_CARD_CODE)),
            )
        card_trips = pc.sum(pc.cast(mask, pa.int64())).as_py() or 0
        card_share = card_trips / trip_count

        if not all(math.isfinite(v) for v in (total_fare, mean_fare, mean_trip_distance, card_share)):
            return _abstain("computed metrics are not finite")

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
    except Exception as exc:
        return _abstain(f"unexpected failure while processing inputs ({type(exc).__name__})")
