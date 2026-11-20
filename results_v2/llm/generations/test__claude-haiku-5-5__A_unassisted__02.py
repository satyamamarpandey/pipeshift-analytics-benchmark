import hashlib
import json
import math
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Columns required by the NYC TLC yellow taxi record format used here.
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

# TLC data dictionary: payment_type 1 = credit card.
CREDIT_CARD_PAYMENT_TYPE = 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PERIOD_RE = re.compile(r"(\d{4})-(\d{2})")


class _Abstain(Exception):
    """Raised internally when a trustworthy result cannot be produced."""


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        raise _Abstain(f"manifest unreadable: {type(exc).__name__}")
    if not isinstance(manifest, dict):
        raise _Abstain("manifest is not a JSON object")
    return manifest


def _expected_sha256(manifest):
    value = manifest.get("sha256")
    if value is None and isinstance(manifest.get("data"), dict):
        value = manifest["data"].get("sha256")
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.lower()):
        raise _Abstain("manifest lacks a valid sha256 for the data file")
    return value.lower()


def _reporting_month(manifest):
    value = manifest.get("period", manifest.get("month"))
    if not isinstance(value, str):
        raise _Abstain("manifest does not specify the reporting month")
    match = _PERIOD_RE.fullmatch(value)
    if not match:
        raise _Abstain("manifest reporting month is not in YYYY-MM form")
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise _Abstain("manifest reporting month is out of range")
    return year, month


def _check_column_types(table):
    schema = table.schema
    for name in ("fare_amount", "trip_distance"):
        t = schema.field(name).type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)):
            raise _Abstain(f"column {name} is not numeric")
    if not pa.types.is_integer(schema.field("payment_type").type):
        raise _Abstain("column payment_type is not integer")
    if not pa.types.is_timestamp(schema.field("tpep_pickup_datetime").type):
        raise _Abstain("column tpep_pickup_datetime is not a timestamp")


def _run(data_path, manifest_path):
    manifest = _load_manifest(manifest_path)

    # Integrity: the snapshot must match the checksum published with it.
    expected = _expected_sha256(manifest)
    try:
        actual = _sha256_file(data_path)
    except OSError as exc:
        raise _Abstain(f"data unreadable: {type(exc).__name__}")
    if actual != expected:
        raise _Abstain("data checksum does not match manifest")

    year, month = _reporting_month(manifest)

    try:
        pf = pq.ParquetFile(data_path)
        names = set(pf.schema_arrow.names)
    except Exception as exc:
        raise _Abstain(f"data is not a readable Parquet file: {type(exc).__name__}")

    missing = [c for c in REQUIRED_COLUMNS if c not in names]
    if missing:
        raise _Abstain(f"missing required columns: {', '.join(missing)}")

    table = pq.read_table(data_path, columns=list(REQUIRED_COLUMNS))

    expected_rows = manifest.get("row_count")
    if expected_rows is not None and expected_rows != table.num_rows:
        raise _Abstain("row count does not match manifest")

    _check_column_types(table)

    for name in REQUIRED_COLUMNS:
        if table.column(name).null_count > 0:
            raise _Abstain(f"column {name} contains nulls")

    if pc.count_distinct(table.column("record_id")).as_py() != table.num_rows:
        raise _Abstain("duplicate record_id values present")

    # Restrict to trips picked up in the reporting month.
    pickups = table.column("tpep_pickup_datetime")
    in_month = pc.and_(
        pc.equal(pc.year(pickups), year),
        pc.equal(pc.month(pickups), month),
    )
    monthly = table.filter(in_month)
    n = monthly.num_rows
    if n == 0:
        raise _Abstain("no trips in the reporting month")

    fares = [float(v) for v in monthly.column("fare_amount").to_pylist()]
    distances = [float(v) for v in monthly.column("trip_distance").to_pylist()]

    total_fare = math.fsum(fares)
    mean_fare = total_fare / n
    mean_trip_distance = math.fsum(distances) / n

    card_count = pc.sum(
        pc.cast(
            pc.equal(monthly.column("payment_type"), CREDIT_CARD_PAYMENT_TYPE),
            pa.int64(),
        )
    ).as_py() or 0
    card_share = card_count / n

    values = (total_fare, mean_fare, mean_trip_distance, card_share)
    if not all(math.isfinite(v) for v in values):
        raise _Abstain("non-finite metric computed")

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


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        return _run(data_path, manifest_path)
    except _Abstain as exc:
        return _abstain(str(exc))
    except Exception as exc:
        return _abstain(f"unexpected error: {type(exc).__name__}")
