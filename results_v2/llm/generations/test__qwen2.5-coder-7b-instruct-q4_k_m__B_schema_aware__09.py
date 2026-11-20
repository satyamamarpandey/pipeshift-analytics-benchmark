import pandas as pd
import pyarrow.parquet as pq
import json

def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # Load the Parquet file
        data = pq.read_table(data_path).to_pandas()
        
        # Load the manifest file
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Validate required columns
        required_columns = ['record_id', 'tpep_pickup_datetime', 'trip_distance', 'payment_type', 'fare_amount']
        if not all(col in data.columns for col in required_columns):
            return {"status": "abstain", "metrics": None, "reason": "Missing required columns"}
        
        # Validate record_id uniqueness
        if data[data.duplicated('record_id')].shape[0] > 0:
            return {"status": "abstain", "metrics": None, "reason": "Duplicate record_id found"}
        
        # Validate pickup_datetime within reporting_period
        if 'reporting_period' in manifest and manifest['reporting_period']:
            start_time, end_time = pd.to_datetime(manifest['reporting_period'])
            data = data[(data['tpep_pickup_datetime'] >= start_time) & (data['tpep_pickup_datetime'] < end_time)]
        
        # Calculate metrics
        trip_count = data.shape[0]
        total_fare = data['fare_amount'].sum()
        mean_fare = data['fare_amount'].mean()
        mean_trip_distance = data['trip_distance'].mean()
        card_share = (data[data['payment_type'] == 1].shape[0] / trip_count) if trip_count > 0 else 0
        
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
