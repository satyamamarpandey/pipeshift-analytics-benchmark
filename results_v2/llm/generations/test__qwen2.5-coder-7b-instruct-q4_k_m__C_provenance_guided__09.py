import json
import pandas as pd
from pyarrow import parquet

def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # Step 1: Verify manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Step 2: Verify SHA-256
        with open(data_path, 'rb') as f:
            file_sha256 = hashlib.sha256(f.read()).hexdigest()
        if file_sha256 != manifest['file_sha256']:
            return {"status": "abstain", "metrics": None, "reason": "Incorrect SHA-256"}
        
        # Step 3: Verify row count
        with open(data_path, 'rb') as f:
            reader = parquet.ParquetFile(f)
            actual_row_count = reader.metadata.num_rows
        if actual_row_count != manifest['row_count']:
            return {"status": "abstain", "metrics": None, "reason": "Incorrect row count"}
        
        # Step 4: Verify columns
        delivered_columns = set([col['name'] for col in reader.schema])
        expected_columns = {change['to'] if change['type'] == 'rename' else change['from'] for change in manifest['schema_changes']}
        if delivered_columns != expected_columns:
            return {"status": "abstain", "metrics": None, "reason": "Incorrect columns"}
        
        # Step 5: Verify pickup timestamps
        df = reader.read().to_pandas()
        if not (manifest['reporting_period']['start'] <= df['tpep_pickup_datetime'] < manifest['reporting_period']['end']):
            return {"status": "abstain", "metrics": None, "reason": "Invalid pickup timestamps"}
        
        # Step 6: Apply data-contract rules
        required_columns = {'record_id', 'tpep_pickup_datetime', 'trip_distance', 'payment_type', 'fare_amount'}
        if not required_columns.issubset(df.columns):
            return {"status": "abstain", "metrics": None, "reason": "Missing required columns"}
        if df['fare_amount'].min() < 0 or df['fare_amount'].max() > 1000:
            return {"status": "abstain", "metrics": None, "reason": "Invalid fare_amount values"}
        if df['trip_distance'].min() <= 0 or df['trip_distance'].max() > 100:
            return {"status": "abstain", "metrics": None, "reason": "Invalid trip_distance values"}
        
        # Step 7: Normalize schema and compute metrics
        df = df[['record_id', 'tpep_pickup_datetime', 'trip_distance', 'payment_type', 'fare_amount']]
        df.drop_duplicates(inplace=True)
        
        metrics = {
            "trip_count": df.shape[0],
            "total_fare": df['fare_amount'].sum(),
            "mean_fare": df['fare_amount'].mean(),
            "mean_trip_distance": df['trip_distance'].mean(),
            "card_share": (df['payment_type'] == 1).mean()
        }
        
        return {"status": "ok", "metrics": metrics, "reason": ""}
    
    except Exception as e:
        return {"status": "abstain", "metrics": None, "reason": str(e)}
