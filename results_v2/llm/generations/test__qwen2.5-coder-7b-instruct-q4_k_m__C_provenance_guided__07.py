import pandas as pd
import json
from pyarrow.parquet import read_parquet

def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # Step 1: Read manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Step 2: Verify manifest
        if not manifest or 'snapshot_id' not in manifest or 'columns' not in manifest:
            return {"status": "abstain", "metrics": None, "reason": "Invalid manifest"}
        
        # Step 3: Verify data file
        file_hash = read_parquet(data_path).to_pandas().to_json(orient='records')
        if file_hash != manifest['file_sha256']:
            return {"status": "abstain", "metrics": None, "reason": "File hash mismatch"}
        
        # Step 4: Verify row count
        if len(read_parquet(data_path).to_pandas()) != manifest['row_count']:
            return {"status": "abstain", "metrics": None, "reason": "Row count mismatch"}
        
        # Step 5: Verify columns
        expected_columns = {col['name'] for col in manifest['columns']}
        actual_columns = read_parquet(data_path).to_pandas().columns
        if expected_columns != set(actual_columns):
            return {"status": "abstain", "metrics": None, "reason": "Column mismatch"}
        
        # Step 6: Apply schema changes
        schema_changes = {change['from']: change['to'] for change in manifest.get('schema_changes', [])}
        df = read_parquet(data_path).to_pandas()
        df.rename(columns=schema_changes, inplace=True)
        
        # Step 7: Verify pickup timestamps
        if not df['tpep_pickup_datetime'].between(manifest['reporting_period']['start'], manifest['reporting_period']['end']):
            return {"status": "abstain", "metrics": None, "reason": "Pickup timestamp out of range"}
        
        # Step 8: Verify data contract
        required_columns = {'record_id', 'tpep_pickup_datetime', 'trip_distance', 'payment_type', 'fare_amount'}
        if not required_columns.issubset(df.columns):
            return {"status": "abstain", "metrics": None, "reason": "Missing required columns"}
        
        if df['trip_distance'].le(0).any() or df['trip_distance'].gt(100).any():
            return {"status": "abstain", "metrics": None, "reason": "Invalid trip distance"}
        
        if df['fare_amount'].lt(0).any() or df['fare_amount'].gt(1000).any():
            return {"status": "abstain", "metrics": None, "reason": "Invalid fare amount"}
        
        if df['payment_type'].isin([1, 2, 3, 4, 5, 6]).any():
            return {"status": "abstain", "metrics": None, "reason": "Invalid payment type"}
        
        # Step 9: Compute metrics
        trip_count = len(df)
        total_fare = df['fare_amount'].sum()
        mean_fare = df['fare_amount'].mean()
        mean_trip_distance = df['trip_distance'].mean()
        card_share = df[df['payment_type'] == 1]['fare_amount'].sum() / total_fare
        
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
