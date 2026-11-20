import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # Load the Parquet file
        table = pq.read_table(data_path, engine='auto')
        df = table.to_pandas()

        # Calculate the monthly summary
        trip_count = len(df)
        total_fare = df['fare_amount'].sum()
        mean_fare = df['fare_amount'].mean()
        mean_trip_distance = df['trip_distance'].mean()
        card_share = (df['payment_type'] == 1).mean()

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
