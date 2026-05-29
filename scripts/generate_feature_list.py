import pandas as pd
import json
import os

# ✅ Path to your processed training data
csv_path = "data/processed/train.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found. Please check the path.")

# ✅ Load CSV and remove empty columns
df = pd.read_csv(csv_path).dropna(axis=1, how="all")

# ✅ Extract feature names
features = list(df.columns)

# ✅ Define the feature list object
obj = {
    "model_version": "v1.0",
    "seq_len": 30,
    "features": features
}

# ✅ Ensure models directory exists
os.makedirs("models", exist_ok=True)

# ✅ Save JSON file
output_path = "models/feature_list.json"
with open(output_path, "w") as f:
    json.dump(obj, f, indent=2)

print(f"✅ Saved {output_path}")
print(f"Total features: {len(features)}")

print("Features:", features)
# scripts/generate_feature_list.py
# Usage: python scripts/generate_feature_list.py
# Generates models/feature_list.json from data/processed/train.csv
# Make sure data/processed/train.csv exists before running this script  
# (e.g., run scripts/preprocess_data.py first)
# The JSON file is used by the inference service to know which features to expect
# and the sequence length for time-series models.
# The JSON structure is:
# {
#   "model_version": "v1.0",
#   "seq_len": 30,
#   "features": ["time", "temperature", "vibration", "pressure", "RUL"]
# }
# Adjust "seq_len" and "model_version" as needed.
# Make sure to have pandas installed: pip install pandas
# Run this script from the root of the repository
# Example output:
# {
#   "model_version": "v1.0",
#   "seq_len": 30,
#   "features": ["time", "temperature", "vibration", "pressure", "RUL"]
# }
# scripts/generate_feature_list.py
# Usage: python scripts/generate_feature_list.py
# Generates models/feature_list.json from data/processed/train.csv
# Make sure data/processed/train.csv exists before running this script              