# scripts/fix_and_test_local.py
"""
Safe helper to prepare a payload and run local model prediction.
Overwrite this file in your repo.
"""
import os, sys, json, traceback
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np

from models.utils import get_ckpt_input_dim, trim_df_to_input_dim, create_sequences, load_feature_list
from models.lstm_model import LSTMRULModel

MODEL_PATH = "models/saved/lstm_fd001.pt"
CSV = "data/processed/train.csv"
OUT_PAYLOAD = "payload_fixed.json"
SEQ_LEN = 30

def main():
    try:
        if not os.path.exists(MODEL_PATH):
            raise SystemExit(f"Model not found: {MODEL_PATH}")
        if not os.path.exists(CSV):
            raise SystemExit(f"CSV not found: {CSV}")

        df = pd.read_csv(CSV).dropna(axis=1, how="all")
        # choose a unit with many rows if present
        if "unit" in df.columns:
            unit = int(df["unit"].value_counts().idxmax())
            df_unit = df[df["unit"] == unit].copy()
            print("Selected unit:", unit, "rows:", len(df_unit))
        else:
            df_unit = df.copy()
            print("No 'unit' column found. Using full dataset rows:", len(df_unit))

        input_dim = get_ckpt_input_dim(MODEL_PATH)
        print("Checkpoint expects input_dim:", input_dim)

        df_trim, used_cols = trim_df_to_input_dim(df_unit, input_dim)
        print("Trimmed to columns:", used_cols, "-> shape:", df_trim.shape)

        if df_trim.shape[0] < SEQ_LEN:
            raise SystemExit(f"Not enough rows for sequence (need {SEQ_LEN}, have {df_trim.shape[0]})")

        payload_df = df_trim.tail(SEQ_LEN).copy()
        payload = payload_df.to_dict(orient="records")
        with open(OUT_PAYLOAD, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print("Saved payload to", OUT_PAYLOAD)

        # Build sequences: ensure we do NOT drop any valid feature columns
        X, y = create_sequences(payload_df, sequence_length=SEQ_LEN, drop_columns=[])
        print("X shape:", X.shape, "y shape:", y.shape)

        # Load model with explicit input_dim
        model = LSTMRULModel.load(MODEL_PATH, input_dim=X.shape[2])
        print("Model loaded successfully with input_dim:", X.shape[2])

        preds = model.predict(X)
        print("Preds (first 5):", preds[:5])
        print("Preds (last 5):", preds[-5:])
        print("Done.")

    except Exception as e:
        print("ERROR in fix_and_test_local.py")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
