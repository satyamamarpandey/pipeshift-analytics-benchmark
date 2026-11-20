"""Frozen prompt templates for the three AI-assisted workflow conditions.

A: unassisted   - task + brief dataset description + output interface.
B: schema-aware - A + documented data contract and validation instructions.
C: provenance   - B + manifest semantics and an explicit verification protocol.
No prompt contains episode data, fault names, file paths of episodes or truth values.
"""
from __future__ import annotations

import hashlib

SYSTEM_PROMPT = ("You are an expert data engineer. Answer with exactly one Python code block "
                 "containing the complete module and nothing else.")

INTERFACE = """\
Write a Python module that defines:

    def analyze(data_path: str, manifest_path: str) -> dict

It must return either
    {"status": "ok", "metrics": {"trip_count": ..., "total_fare": ..., "mean_fare": ...,
                                 "mean_trip_distance": ..., "card_share": ...}, "reason": ""}
or, if you cannot produce a trustworthy result,
    {"status": "abstain", "metrics": None, "reason": "<short explanation>"}

Metric definitions (do not round any value):
- trip_count: number of trips
- total_fare: sum of fare_amount in USD
- mean_fare: mean fare_amount in USD
- mean_trip_distance: mean trip_distance in miles
- card_share: fraction of trips paid by credit card

Environment: Python 3.13 with only the standard library, pandas 2.2, numpy 2.3 and pyarrow 21.
No network access and no files other than the two paths passed in. The function will be
called many times on different inputs and must not print, prompt, or write files."""

TASK = """\
Task: compute a monthly summary of New York City yellow taxi trips from a data snapshot.
data_path points to a Parquet file of NYC TLC yellow taxi trip records. manifest_path points
to a JSON metadata file published together with the data."""

DATASET_BRIEF = """\
The records follow the NYC TLC yellow taxi trip record format; columns include record_id,
VendorID, tpep_pickup_datetime, tpep_dropoff_datetime, PULocationID, DOLocationID,
trip_distance, payment_type, fare_amount, tip_amount and total_amount."""

CONTRACT = """\
Data contract (validate inputs against it):
| column                | type                 | meaning / rule                                       |
|-----------------------|----------------------|------------------------------------------------------|
| record_id             | int64                | trip identifier assigned by the publisher            |
| VendorID              | int32                | technology provider code                             |
| tpep_pickup_datetime  | timestamp (naive)    | pickup time, local New York time (America/New_York)  |
| tpep_dropoff_datetime | timestamp (naive)    | drop-off time, local New York time                   |
| PULocationID          | int32                | pickup taxi zone                                     |
| DOLocationID          | int32                | drop-off taxi zone                                   |
| trip_distance         | double               | miles; valid range 0 < x <= 100                      |
| payment_type          | int64                | 0 Flex Fare, 1 Credit card, 2 Cash, 3 No charge,     |
|                       |                      | 4 Dispute, 5 Unknown, 6 Voided trip; no other codes  |
| fare_amount           | double               | USD; valid range 0 <= x <= 1000                      |
| tip_amount            | double               | USD                                                  |
| total_amount          | double               | USD                                                  |
Rules:
- record_id, tpep_pickup_datetime, trip_distance, payment_type and fare_amount are required
  and must not be null.
- A record delivered more than once with identical values is a replay and counts once.
  The same record_id with different values is a conflict.
- All trips must have tpep_pickup_datetime within the manifest's reporting_period
  (start inclusive, end exclusive, local time).
- Producers may evolve the physical schema: columns may be reordered or added, numeric or
  datetime values may be delivered as strings or other compatible types, timestamps may be
  timezone-aware, and renamed columns are declared in the manifest's "schema_changes" list as
  {"type": "rename", "from": <original name>, "to": <new name>}.
- Card share counts payment_type == 1 only.
If the data violate the contract in a way that prevents a correct result, abstain rather than
returning numbers."""

MANIFEST_SPEC = """\
Manifest fields: dataset, snapshot_id, snapshot_version, reporting_period {start, end, timezone},
file, file_sha256 (SHA-256 hex digest of the exact bytes of the delivered data file),
row_count (number of rows in the delivered file), columns [{name, type}], units,
schema_changes, description.

Verification protocol (perform every step before computing metrics; abstain with a reason on
any failure):
1. The manifest must exist and be valid JSON with the fields above.
2. The SHA-256 of the data file's bytes must equal file_sha256.
3. The number of rows in the file must equal row_count.
4. The delivered columns must agree with the manifest's columns after applying schema_changes.
5. Every pickup timestamp must fall within reporting_period.
6. Apply all data-contract rules above (required non-null fields, documented codes, valid
   ranges, replay and conflict handling, units).
Only if all checks pass, normalize the schema and compute the metrics."""

CONDITIONS = {
    'A_unassisted': [TASK, DATASET_BRIEF, INTERFACE],
    'B_schema_aware': [TASK, CONTRACT, INTERFACE],
    'C_provenance_guided': [TASK, CONTRACT, MANIFEST_SPEC, INTERFACE],
}


def build_prompt(condition: str) -> str:
    return '\n\n'.join(CONDITIONS[condition]) + '\n'


def prompt_hashes() -> dict[str, str]:
    out = {c: hashlib.sha256(build_prompt(c).encode()).hexdigest() for c in CONDITIONS}
    out['system'] = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    return out


if __name__ == '__main__':
    for c in CONDITIONS:
        print(f'===== {c} ({len(build_prompt(c))} chars)\n{build_prompt(c)}')
    print(prompt_hashes())
