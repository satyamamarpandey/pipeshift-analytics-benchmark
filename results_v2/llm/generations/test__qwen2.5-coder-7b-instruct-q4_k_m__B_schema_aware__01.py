import pandas as pd
import pyarrow.parquet as pq

def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # Load the data
        df = pq.read_table(data_path).to_pandas()
        
        # Load the manifest
        with open(manifest_path, 'r') as f:
            manifest = eval(f.read())
        
        # Check for required columns
        required_columns = {'record_id', 'tpep_pickup_datetime', 'trip_distance', 'payment_type', 'fare_amount'}
        if not required_columns.issubset(df.columns):
            return {"status": "abstain", "metrics": None, "reason": "Missing required columns"}
        
        # Convert tpep_pickup_datetime to datetime
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        
        # Check for replay and conflict records
        df.drop_duplicates(subset=['record_id'], inplace=True)
        
        # Filter trips by reporting period
        if 'schema_changes' in manifest and 'tpep_pickup_datetime' in manifest['schema_changes']:
            old_name = manifest['schema_changes'][0]['from']
            df.rename(columns={old_name: 'tpep_pickup_datetime'}, inplace=True)
        
        if 'reporting_period' in manifest:
            start_date, end_date = pd.to_datetime(manifest['reporting_period']['start']), pd.to_datetime(manifest['reporting_period']['end'])
            df = df[(df['tpep_pickup_datetime'] >= start_date) & (df['tpep_pickup_datetime'] < end_date)]
        
        # Validate data
        if df['trip_distance'].le(0).any() or df['fare_amount'].lt(0).any() or df['fare_amount'].gt(1000).any():
            return {"status": "abstain", "metrics": None, "reason": "Invalid data values"}
        
        # Calculate metrics
        trip_count = len(df)
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
