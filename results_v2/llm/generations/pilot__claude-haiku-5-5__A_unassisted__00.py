"""Compute a summary of NYC TLC yellow taxi trip records.

analyze() verifies the Parquet file against the published manifest before
computing metrics. It abstains instead of returning numbers when the input
cannot be verified or is malformed.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REQUIRED_COLUMNS = (
    "record_id",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "fare_amount",
    "trip_distance",
    "payment_type",
)
CREDIT_CARD_CODE = 1  # TLC data dictionary: 1 = credit card
_HEX_CHARS = frozenset("0123456789abcdef")
_NAME_KEYS = frozenset({"name", "path", "file", "filename", "file_name", "key", "uri"})
_ROW_COUNT_KEYS = frozenset({"row_count", "num_rows", "record_count", "n_rows", "total_rows"})


def _abstain(reason: str) -> dict:
    return {"status": "abstain", "metrics": None, "reason": reason}


def _iter_dicts(obj: Any) -> Iterator[dict]:
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            yield cur
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in _HEX_CHARS for c in value.lower())
    )


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_applies_to_file(entry: dict, fname: str) -> bool:
    names = [
        v for k, v in entry.items()
        if str(k).lower() in _NAME_KEYS and isinstance(v, str)
    ]
    if not names:
        return True
    return any(os.path.basename(n.replace("\\", "/")) == fname for n in names)


def _verify_against_manifest(manifest: Any, data_path: str) -> tuple[bool, str]:
    fname = os.path.basename(data_path)
    checksums: set[str] = set()
    row_counts: set[int] = set()

    for entry in _iter_dicts(manifest):
        if not _entry_applies_to_file(entry, fname):
            continue
        for key, value in entry.items():
            key_l = str(key).lower()
            if "sha256" in key_l and _is_sha256(value):
                checksums.add(value.lower())
            elif (
                key_l in _ROW_COUNT_KEYS
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                row_counts.add(value)

    if len(checksums) > 1 or len(row_counts) > 1:
        return False, "manifest lists conflicting checksums or row counts for this file"
    if not checksums and not row_counts:
        return False, "manifest provides no checksum or row count to verify the data against"

    if checksums:
        if _sha256_of_file(data_path) != next(iter(checksums)):
            return False, "data file sha256 does not match the manifest"

    if row_counts:
        with pq.ParquetFile(data_path) as pf:
            if pf.metadata.num_rows != next(iter(row_counts)):
                return False, "data file row count does not match the manifest"

    return True, ""


def _is_numeric(t: pa.DataType) -> bool:
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)


def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        if not os.path.isfile(data_path):
            return _abstain("data file not found")
        if not os.path.isfile(manifest_path):
            return _abstain("manifest file not found")

        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        if not isinstance(manifest, (dict, list)):
            return _abstain("manifest is not a JSON object")

        verified, why = _verify_against_manifest(manifest, data_path)
        if not verified:
            return _abstain(why)

        with pq.ParquetFile(data_path) as pf:
            present = set(pf.schema_arrow.names)
            missing = [c for c in REQUIRED_COLUMNS if c not in present]
            if missing:
                return _abstain(f"missing required columns: {', '.join(missing)}")
            table = pf.read(columns=list(REQUIRED_COLUMNS))

        n = table.num_rows
        if n == 0:
            return _abstain("no trip records in data file")

        for col in REQUIRED_COLUMNS:
            if table.column(col).null_count > 0:
                return _abstain(f"column {col} contains null values")

        record_ids = table.column("record_id")
        if pc.count_distinct(record_ids).as_py() != n:
            return _abstain("duplicate record_id values present")

        fare_raw = table.column("fare_amount")
        dist_raw = table.column("trip_distance")
        pay_raw = table.column("payment_type")

        if not _is_numeric(fare_raw.type) or not _is_numeric(dist_raw.type):
            return _abstain("fare_amount or trip_distance is not numeric")
        if not pa.types.is_integer(pay_raw.type):
            return _abstain("payment_type is not an integer code")

        fare = fare_raw.cast(pa.float64())
        dist = dist_raw.cast(pa.float64())

        if pc.all(pc.is_finite(fare)).as_py() is not True:
            return _abstain("fare_amount contains non-finite values")
        if pc.all(pc.is_finite(dist)).as_py() is not True:
            return _abstain("trip_distance contains non-finite values")

        total_fare = float(pc.sum(fare).as_py())
        mean_fare = total_fare / n
        mean_trip_distance = float(pc.sum(dist).as_py()) / n

        credit_mask = pc.cast(pc.equal(pay_raw, CREDIT_CARD_CODE), pa.int64())
        credit_trips = int(pc.sum(credit_mask).as_py() or 0)
        card_share = credit_trips / n

        return {
            "status": "ok",
            "metrics": {
                "trip_count": int(n),
                "total_fare": total_fare,
                "mean_fare": mean_fare,
                "mean_trip_distance": mean_trip_distance,
                "card_share": card_share,
            },
            "reason": "",
        }
    except Exception as exc:  # noqa: BLE001
        return _abstain(f"could not read or validate inputs: {type(exc).__name__}")
