import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # Load manifest and data
        manifest = pd.read_json(manifest_path)
        data = pq.read_table(data_path).to_pandas()

        # Validate data schema
        required_columns = {'record_id', 'tpep_pickup_datetime', 'trip_distance', 'payment_type', 'fare_amount'}
        if not required_columns.issubset(data.columns):
            return {"status": "abstain", "metrics": None, "reason": "Missing required columns in data"}

        # Check for replay and conflict records
        if data.duplicated(subset=['record_id']).any():
            return {"status": "abstain", "metrics": None, "reason": "Replay or conflict records found"}

        # Filter trips within reporting period
        if 'tpep_pickup_datetime' in data.columns and 'reporting_period' in manifest:
            reporting_start, reporting_end = manifest['reporting_period']
            data = data[(data['tpep_pickup_datetime'] >= reporting_start) & (data['tpep_pickup_datetime'] < reporting_end)]

        # Calculate metrics
        trip_count = len(data)
        total_fare = data['fare_amount'].sum()
        mean_fare = data['fare_amount'].mean()
        mean_trip_distance = data['trip_distance'].mean()
        card_share = data[data['payment_type'] == 1]['fare_amount'].sum() / total_fare if total_fare != 0 else 0

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
