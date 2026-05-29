# scripts/generate_payload.py
import os
import json
import pandas as pd
import sys

CSV_PATH = "data/processed/train.csv"     # change if your processed CSV has a different name
FEATURES_PATH = "models/feature_list.json"
OUT_PATH = "payload.json"
SEQ_LEN = 30   # or read from models/feature_list.json if you prefer

# Validate paths
if not os.path.exists(CSV_PATH):
    print(f"ERROR: CSV not found at {CSV_PATH}")
    sys.exit(1)

# Load processed CSV
df = pd.read_csv(CSV_PATH).dropna(axis=1, how="all").fillna(0)

# Determine features (prefer feature_list.json if present)
if os.path.exists(FEATURES_PATH):
    try:
        feat_obj = json.load(open(FEATURES_PATH))
        features = feat_obj.get("features", list(df.columns))
        # keep only features that actually exist in df (and preserve order)
        features = [c for c in features if c in df.columns]
    except Exception as e:
        print("Warning: failed to load feature_list.json:", e)
        features = list(df.columns)
else:
    features = list(df.columns)

if len(features) == 0:
    print("ERROR: No usable features found in CSV.")
    sys.exit(1)

# Check available rows
if len(df) < SEQ_LEN:
    print(f"ERROR: Not enough rows in {CSV_PATH}. Need at least {SEQ_LEN}, got {len(df)}")
    sys.exit(1)

# Build payload: take last SEQ_LEN rows and keep only selected features
payload_df = df[features].tail(SEQ_LEN)
payload = payload_df.to_dict(orient="records")

# Save payload.json
with open(OUT_PATH, "w") as f:
    json.dump(payload, f, indent=2)

print(f"✅ Generated {OUT_PATH} with {len(payload)} rows and {len(features)} features.")
print("First row keys:", list(payload[0].keys())[:10])
