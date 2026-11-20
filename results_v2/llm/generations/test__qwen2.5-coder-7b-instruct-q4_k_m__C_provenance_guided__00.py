import json
import pandas as pd
import pyarrow as pa
from pyarrow.parquet import read_table
from pyarrow.json import read_json

def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # Step 1: Validate manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        if not manifest:
            return {"status": "abstain", "metrics": None, "reason": "Manifest is empty"}

        # Step 2: Verify data file integrity
        data = read_table(data_path)
        data_hash = pa.compute.sha256(data.to_batches()[0].to_arrow()).to_pybytes().hex()
        if data_hash != manifest['file_sha256']:
            return {"status": "abstain", "metrics": None, "reason": "Data file integrity check failed"}

        # Step 3: Verify row count
        if data.num_rows != manifest['row_count']:
            return {"status": "abstain", "metrics": None, "reason": "Row count mismatch"}

        # Step 4: Verify columns
        for col in manifest['columns']:
            if col['name'] not in data.column_names:
                return {"status": "abstain", "metrics": None, "reason": f"Column {col['name']} not found"}
            if col['type'] != data[col['name']].type.to_string():
                return {"status": "abstain", "metrics": None, "reason": f"Column {col['name']} type mismatch"}

        # Step 5: Apply schema changes
        if manifest.get('schema_changes'):
            schema = data.schema
            for change in manifest['schema_changes']:
                if change['type'] == 'rename':
                    schema = schema.remove(change['from']).add(pa.field(change['to'], schema.field(change['from']).type))
            data = data.select([schema.field(i).name for i in range(len(schema))])

        # Step 6: Validate data
        df = data.to_pandas()
        if not all(df['trip_distance'] > 0) or not all(df['trip_distance'] <= 100):
            return {"status": "abstain", "metrics": None, "reason": "Invalid trip distance"}
        if not all(df['payment_type'].isin([0, 1, 2, 3, 4, 5, 6])):
            return {"status": "abstain", "metrics": None, "reason": "Invalid payment type"}
        if not all(df['fare_amount'] >= 0) or not all(df['fare_amount'] <= 1000):
            return {"status": "abstain", "metrics": None, "reason": "Invalid fare amount"}
        if not all(df['tpep_pickup_datetime'] >= manifest['reporting_period']['start']) or not all(df['tpep_pickup_datetime'] < manifest['reporting_period']['end']):
            return {"status": "abstain", "metrics": None, "reason": "Pickup timestamp out of reporting period"}

        # Step 7: Compute metrics
        trip_count = df.shape[0]
        total_fare = df['fare_amount'].sum()
        mean_fare = df['fare_amount'].mean()
        mean_trip_distance = df['trip_distance'].mean()
        card_share = (df['payment_type'] == 1).mean()

        return {
            "status": "ok",
            "metrics": {
                "trip_count": trip_count,
                "total_fare": total_fare,
                "mean_fare": mean_fare,
                "mean_trip_distance": mean_trip_distance,
                "card_share": card_share
            },
            "reason": ""
        }
    except Exception as e:
        return {"status": "abstain", "metrics": None, "reason": str(e)}
