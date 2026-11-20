import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
from typing import Dict

def analyze(data_path: str, manifest_path: str) -> Dict[str, object]:
    try:
        with open(manifest_path, 'r') as f:
            manifest = pd.read_json(f)
        
        if manifest['dataset'] != 'nyc_taxi_yellow' or manifest['snapshot_id'] not in ['202301', '202302', '202303']:
            return {"status": "abstain", "metrics": None, "reason": "Invalid dataset or snapshot ID"}
        
        if manifest['snapshot_version'] != '1.0':
            return {"status": "abstain", "metrics": None, "reason": "Invalid snapshot version"}
        
        if manifest['file_sha256'] != hash_file(data_path):
            return {"status": "abstain", "metrics": None, "reason": "File integrity check failed"}
        
        if len(manifest['columns']) != 17 or not all(c['name'] in ['record_id', 'tpep_pickup_datetime', 'trip_distance', 'payment_type', 'fare_amount'] for c in manifest['columns']):
            return {"status": "abstain", "metrics": None, "reason": "Invalid column set"}
        
        df = pq.read_table(data_path).to_pandas()
        
        if not all(df['tpep_pickup_datetime'] >= manifest['reporting_period']['start'] and df['tpep_pickup_datetime'] < manifest['reporting_period']['end'] for _ in range(len(df))):
            return {"status": "abstain", "metrics": None, "reason": "Pickup time outside reporting period"}
        
        df = df.drop_duplicates(subset=['record_id', 'tpep_pickup_datetime', 'trip_distance', 'payment_type', 'fare_amount'])
        
        if not all(df['record_id'].notna() and df['tpep_pickup_datetime'].notna() and df['trip_distance'].notna() and df['payment_type'].notna() and df['fare_amount'].notna() for _ in range(len(df))):
            return {"status": "abstain", "metrics": None, "reason": "Required fields are null"}
        
        if not all(df['payment_type'].isin([0, 1, 2, 3, 4, 5, 6]) for _ in range(len(df))):
            return {"status": "abstain", "metrics": None, "reason": "Invalid payment_type values"}
        
        if not all(0 < df['trip_distance'] <= 100 for _ in range(len(df))):
            return {"status": "abstain", "metrics": None, "reason": "Invalid trip_distance values"}
        
        if not all(0 <= df['fare_amount'] <= 1000 for _ in range(len(df))):
            return {"status": "abstain", "metrics": None, "reason": "Invalid fare_amount values"}
        
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        
        metrics = {
            'trip_count': len(df),
            'total_fare': df['fare_amount'].sum(),
            'mean_fare': df['fare_amount'].mean(),
            'mean_trip_distance': df['trip_distance'].mean(),
            'card_share': df[df['payment_type'] == 1]['fare_amount'].sum() / df['fare_amount'].sum()
        }
        
        return {"status": "ok", "metrics": metrics, "reason": ""}
    
    except Exception as e:
        return {"status": "abstain", "metrics": None, "reason": str(e)}

def hash_file(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        return f.read().hex()
