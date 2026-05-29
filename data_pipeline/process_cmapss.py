# data_pipeline/process_cmapss.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import pandas as pd
from kafka import KafkaConsumer
from sqlalchemy import create_engine


def get_consumer(topic="cmapss-data"):
    """Create and return a Kafka consumer."""
    return KafkaConsumer(
        topic,
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="cmapss-consumer-group",
    )


def transform_row(row):
    """Apply transformations to each row (example: normalize values)."""
    # Example: scale sensor values (you can customize this)
    for k, v in row.items():
        if isinstance(v, (int, float)):
            row[k] = v / 100.0
    return row


def save_to_postgres(df, table="cmapss_processed"):
    """Save DataFrame to Postgres."""
    engine = create_engine("postgresql://user:password@localhost:5432/mydb")
    df.to_sql(table, engine, if_exists="append", index=False)
    print(f"[INFO] Saved {len(df)} rows to Postgres table '{table}'")


def save_to_csv(df, path="data/processed/cmapss_processed.csv"):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))
    print(f"[INFO] Saved {len(df)} rows to {path}")


def consume_and_process(topic="cmapss-data", limit=None, sink="csv"):
    consumer = get_consumer(topic)
    rows = []

    for i, msg in enumerate(consumer):
        row = transform_row(msg.value)
        rows.append(row)

        if limit is not None and i + 1 >= limit:
            break

    df = pd.DataFrame(rows)

    if sink == "postgres":
        save_to_postgres(df)
    else:
        save_to_csv(df)

    print("[INFO] Finished consuming and processing data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consume CMAPSS data from Kafka")
    parser.add_argument("--topic", type=str, default="cmapss-data", help="Kafka topic name")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to consume")
    parser.add_argument("--sink", type=str, choices=["csv", "postgres"], default="csv", help="Where to save the processed data")

    args = parser.parse_args()

    consume_and_process(topic=args.topic, limit=args.limit, sink=args.sink)
