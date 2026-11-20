import pandas as pd
import pyarrow.parquet as pq
from pyarrow.json import read_json
import numpy as np

def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # Load data
        data = pq.read_table(data_path).to_pandas()
        manifest = read_json(manifest_path).to_pandas()
        
        # Validate required columns
        required_columns = ['record_id', 'tpep_pickup_datetime', 'trip_distance', 'payment_type', 'fare_amount']
        if not all(col in data.columns for col in required_columns):
            return {"status": "abstain", "metrics": None, "reason": "Missing required columns"}
        
        # Convert data types if necessary
        data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
        data['payment_type'] = data['payment_type'].astype(int)
        data['fare_amount'] = data['fare_amount'].astype(float)
        data['trip_distance'] = data['trip_distance'].astype(float)
        
        # Validate fare_amount range
        if data['fare_amount'].max() > 1000 or data['fare_amount'].min() < 0:
            return {"status": "abstain", "metrics": None, "reason": "Invalid fare_amount range"}
        
        # Validate trip_distance range
        if data['trip_distance'].max() > 100 or data['trip_distance'].min() <= 0:
            return {"status": "abstain", "metrics": None, "reason": "Invalid trip_distance range"}
        
        # Filter by reporting period
        reporting_period = pd.Interval(manifest['reporting_period'][0], manifest['reporting_period'][1], closed='left')
        data = data[data['tpep_pickup_datetime'].dt.tz_localize(None).dt.tz_localize('America/New_York')].query(f"tpep_pickup_datetime @ {reporting_period}")
        
        # Handle replayers and conflicts
        data = data.drop_duplicates(subset=required_columns)
        
        # Compute metrics
        trip_count = len(data)
        total_fare = data['fare_amount'].sum()
        mean_fare = data['fare_amount'].mean()
        mean_trip_distance = data['trip_distance'].mean()
        card_share = (data['payment_type'] == 1).mean()
        
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
