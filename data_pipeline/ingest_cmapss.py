 # data_pipeline/ingest_cmapss.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import argparse
from kafka import KafkaProducer
from data_pipeline.cmapss_loader import load_cmapss, preprocess_cmapss


def get_producer():
    """Create and return a Kafka producer."""
    return KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def stream_cmapss(
    raw_file="data/raw/CMAPSS/train_FD001.txt",
    topic="cmapss-data",
    delay=0.5,
    limit=None,
):
    """Stream CMAPSS dataset rows into Kafka."""
    df = preprocess_cmapss(load_cmapss(raw_file))
    producer = get_producer()

    for i, (_, row) in enumerate(df.iterrows()):
        if limit is not None and i >= limit:
            break

        producer.send(topic, value=row.to_dict())
        print(f"[INFO] Sent row {i+1} to Kafka topic '{topic}'")
        time.sleep(delay)

    producer.flush()
    print("[INFO] Finished streaming CMAPSS data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream CMAPSS data to Kafka")
    parser.add_argument("--file", type=str, default="data/raw/CMAPSS/train_FD001.txt", help="Path to CMAPSS dataset")
    parser.add_argument("--topic", type=str, default="cmapss-data", help="Kafka topic name")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between messages in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to stream (default: all)")

    args = parser.parse_args()

    stream_cmapss(raw_file=args.file, topic=args.topic, delay=args.delay, limit=args.limit)
