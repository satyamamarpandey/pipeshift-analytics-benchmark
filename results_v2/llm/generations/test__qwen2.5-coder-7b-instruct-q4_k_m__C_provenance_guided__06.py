import json
import pandas as pd
from pyarrow.parquet import read_parquet

def analyze(data_path: str, manifest_path: str) -> dict:
    # Step 1: Load and validate the manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    if not manifest or 'file_sha256' not in manifest or 'row_count' not in manifest:
        return {"status": "abstain", "metrics": None, "reason": "Invalid manifest file"}
    
    # Step 2: Verify the SHA-256 and row count
    with open(data_path, 'rb') as f:
        file_sha256 = f.read()
        if file_sha256.hex() != manifest['file_sha256']:
            return {"status": "abstain", "metrics": None, "reason": "File SHA-256 mismatch"}
    
    # Step 3: Check if the number of rows matches
    try:
        df = read_parquet(data_path)
        if df.shape[0] != manifest['row_count']:
            return {"status": "abstain", "metrics": None, "reason": "Row count mismatch"}
    except Exception as e:
        return {"status": "abstain", "metrics": None, "reason": f"Error reading Parquet file: {e}"}
    
    # Step 4: Apply schema changes and verify columns
    for change in manifest.get('schema_changes', []):
        if change['type'] == 'rename':
            df.rename(columns={change['from']: change['to']}, inplace=True)
    
    required_columns = {'VendorID', 'tpep_pickup_datetime', 'trip_distance', 'payment_type', 'fare_amount'}
    if not required_columns.issubset(df.columns):
        return {"status": "abstain", "metrics": None, "reason": "Missing required columns"}
    
    # Step 5: Verify pickup timestamps
    pickup_timezone = manifest['reporting_period']['timezone']
    if pickup_timezone not in pd.Timestamp.now().tzinfo.tzname():
        return {"status": "abstain", "metrics": None, "reason": "Timezone not supported"}
    
    pickup_min = pd.Timestamp(manifest['reporting_period']['start'], tz=pickup_timezone)
    pickup_max = pd.Timestamp(manifest['reporting_period']['end'], tz=pickup_timezone)
    if not df['tpep_pickup_datetime'].between(pickup_min, pickup_max).all():
        return {"status": "abstain", "metrics": None, "reason": "Pickup timestamps out of range"}
    
    # Step 6: Apply data-contract rules
    if df.isnull().any(axis=1).sum() > 0:
        return {"status": "abstain", "metrics": None, "reason": "Null values found in required fields"}
    
    if not (0 < df['trip_distance'] <= 100).all():
        return {"status": "abstain", "metrics": None, "reason": "Invalid trip distance values"}
    
    if not (0 <= df['fare_amount'] <= 1000).all():
        return {"status": "abstain", "metrics": None, "reason": "Invalid fare amount values"}
    
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
