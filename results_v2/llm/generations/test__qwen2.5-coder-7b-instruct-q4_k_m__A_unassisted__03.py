import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def analyze(data_path: str, manifest_path: str) -> dict:
    try:
        # Read the manifest file to get the schema
        with open(manifest_path, 'r') as f:
            manifest = pd.read_json(f)
            schema = pa.schema({col: manifest[col]['type'] for col in manifest['columns']})

        # Read the Parquet file
        table = pq.read_table(data_path, schema=schema)
        df = table.to_pandas()

        # Calculate the metrics
        trip_count = df['trip_distance'].count()
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
        return {
            "status": "abstain",
            "metrics": None,
            "reason": str(e)
        }
