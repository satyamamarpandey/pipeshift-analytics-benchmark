import hashlib
import json
import math
import os

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

CREDIT_CARD_CODE = 1
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
# Columns that feed the metrics; nulls here make the metrics untrustworthy.
METRIC_COLUMNS = ("trip_distance", "payment_type", "fare_amount")


def _abstain(reason):
    return {"status": "abstain", "metrics": None, "reason": reason}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not a JSON object")
    return manifest


def _expectations(manifest, basename):
    """Return (expected_sha256, expected_row_count) for the data file.

    Either may be None if the manifest does not declare it.
    """
    if "files" in manifest:
        entries = manifest["files"]
        if not isinstance(entries, list):
            raise ValueError("manifest 'files' is not a list")
        matches = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name", entry.get("path", entry.get("filename")))
            if name is not None and os.path.basename(str(name)) == basename:
                matches.append(entry)
        if len(matches) != 1:
            raise ValueError("manifest does not describe the data file uniquely")
        scope = matches[0]
    else:
        scope = manifest

    sha = scope.get("sha256")
    rows = scope.get("row_count", scope.get("num_rows", scope.get("record_count")))
    if sha is not None and not isinstance(sha, str):
        raise ValueError("manifest sha256 is not a string")
    if rows is not None:
        if isinstance(rows, bool) or not isinstance(rows, (int, float)) or rows != int(rows):
            raise ValueError("manifest row count is not an integer")
        rows = int(rows)
    return (sha.lower() if sha else None), rows


def _analyze(data_path, manifest_path):
    manifest = _load_manifest(manifest_path)
    expected_sha, expected_rows = _expectations(manifest, os.path.basename(data_path))

    if expected_sha is None and expected_rows is None:
        return _abstain("manifest provides no checksum or row count to verify the snapshot")

    # Verify the snapshot against the manifest.
    if expected_sha is not None and _sha256(data_path) != expected_sha:
        return _abstain("data file checksum does not match manifest")

    parquet_file = pq.ParquetFile(data_path)
    n_rows = parquet_file.metadata.num_rows
    if expected_rows is not None and n_rows != expected_rows:
        return _abstain("row count in data file does not match manifest")

    schema_names = set(parquet_file.schema_arrow.names)
    missing = [c for c in REQUIRED_COLUMNS if c not in schema_names]
    if missing:
        return _abstain(f"missing required columns: {', '.join(missing)}")

    if n_rows == 0:
        return _abstain("snapshot contains no trips")

    table = pq.read_table(data_path, columns=list(REQUIRED_COLUMNS))

    # Each record must be a distinct trip; duplicates would double count.
    record_ids = table["record_id"]
    if record_ids.null_count != 0:
        return _abstain("record_id contains nulls")
    if pc.count_distinct(record_ids).as_py() != n_rows:
        return _abstain("record_id is not unique; trips may be duplicated")

    for name in METRIC_COLUMNS:
        if table[name].null_count != 0:
            return _abstain(f"{name} contains nulls")

    fares = np.asarray(table["fare_amount"].to_numpy(), dtype=np.float64)
    distances = np.asarray(table["trip_distance"].to_numpy(), dtype=np.float64)
    payments = np.asarray(table["payment_type"].to_numpy(), dtype=np.float64)

    if not (np.all(np.isfinite(fares)) and np.all(np.isfinite(distances))
            and np.all(np.isfinite(payments))):
        return _abstain("metric columns contain non-finite values")
    if not np.all(payments == np.floor(payments)) or np.any(payments < 0):
        return _abstain("payment_type contains non-integer or negative codes")

    trip_count = int(n_rows)
    total_fare = float(math.fsum(fares.tolist()))
    mean_fare = total_fare / trip_count
    mean_trip_distance = float(math.fsum(distances.tolist())) / trip_count
    card_trips = int(np.count_nonzero(payments == CREDIT_CARD_CODE))
    card_share = card_trips / trip_count

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
    try:
        return _analyze(data_path, manifest_path)
    except Exception as exc:  # any failure means we cannot vouch for the result
        return _abstain(f"cannot produce trustworthy result: {type(exc).__name__}")
