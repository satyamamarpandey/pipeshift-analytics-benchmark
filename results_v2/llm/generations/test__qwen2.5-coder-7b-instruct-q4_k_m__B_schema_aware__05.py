import pandas as pd

def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # Read the Parquet file
        df = pd.read_parquet(data_path)
        
        # Validate the required columns and their types
        required_columns = ['record_id', 'tpep_pickup_datetime', 'trip_distance', 'payment_type', 'fare_amount']
        if not all(col in df.columns for col in required_columns):
            return {"status": "abstain", "metrics": None, "reason": "Missing required columns"}
        
        # Convert tpep_pickup_datetime to datetime
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        
        # Validate fare_amount, trip_distance, and payment_type ranges
        if df['fare_amount'].min() < 0 or df['fare_amount'].max() > 1000:
            return {"status": "abstain", "metrics": None, "reason": "Invalid fare_amount range"}
        if df['trip_distance'].min() < 0 or df['trip_distance'].max() > 100:
            return {"status": "abstain", "metrics": None, "reason": "Invalid trip_distance range"}
        if df['payment_type'].min() < 0 or df['payment_type'].max() > 6:
            return {"status": "abstain", "metrics": None, "reason": "Invalid payment_type range"}
        
        # Check for replay and conflict
        df['record_id_hash'] = df['record_id'].astype(str) + df['tpep_pickup_datetime'].astype(str)
        unique_records = df['record_id_hash'].nunique()
        if unique_records != df.shape[0]:
            return {"status": "abstain", "metrics": None, "reason": "Replay or conflict detected"}
        
        # Load the manifest and validate reporting_period
        with open(manifest_path, 'r') as f:
            manifest = pd.read_json(f)
        reporting_period = pd.Interval(manifest['reporting_period']['start'], manifest['reporting_period']['end'], closed='left')
        if not (reporting_period.contains(df['tpep_pickup_datetime'].min()) and reporting_period.contains(df['tpep_pickup_datetime'].max())):
            return {"status": "abstain", "metrics": None, "reason": "Trip outside reporting period"}
        
        # Compute metrics
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
