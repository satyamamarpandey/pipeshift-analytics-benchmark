import pandas as pd
from typing import Dict

def analyze(data_path: str, manifest_path: str) -> Dict:
    try:
        # Load the dataset
        df = pd.read_parquet(data_path)

        # Calculate the monthly summary
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        monthly_summary = df.resample('M', on='tpep_pickup_datetime').agg({
            'trip_distance': 'sum',
            'fare_amount': ['sum', 'mean']
        }).reset_index()

        # Extract the metrics
        trip_count = monthly_summary['trip_distance'].sum()
        total_fare = monthly_summary['fare_amount']['sum'].sum()
        mean_fare = monthly_summary['fare_amount']['mean'].mean()
        mean_trip_distance = monthly_summary['trip_distance'].mean()
        card_share = (monthly_summary[monthly_summary['fare_amount']['sum'] > 0]['trip_distance'].sum() / trip_count)

        # Format the results
        metrics = {
            "trip_count": trip_count,
            "total_fare": total_fare,
            "mean_fare": mean_fare,
            "mean_trip_distance": mean_trip_distance,
            "card_share": card_share
        }

        return {"status": "ok", "metrics": metrics, "reason": ""}
    except Exception as e:
        return {"status": "abstain", "metrics": None, "reason": str(e)}
