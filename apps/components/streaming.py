from kafka import KafkaConsumer
import json
import pandas as pd

def consume(topic="cmapss-data", max_rows=30):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        consumer_timeout_ms=3000
    )

    rows = []
    for msg in consumer:
        rows.append(msg.value)
        if len(rows) >= max_rows:
            break

    consumer.close()
    return pd.DataFrame(rows)
