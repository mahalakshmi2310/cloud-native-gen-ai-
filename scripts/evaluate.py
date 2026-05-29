# scripts/evaluate.py
import argparse
import pandas as pd
import numpy as np
from models.lstm_model import LSTMRULModel
from models.utils import create_sequences
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--dataset", type=str, required=True, help="Validation dataset path")
    args = parser.parse_args()

    # 1) Load validation data
    df = pd.read_csv(args.dataset)

    # 2) Basic cleaning
    df = df.dropna(axis=1, how="all")  # drop all-NaN columns
    df = df.fillna(0)                  # replace partial NaNs
    print(f"[CLEAN] Validation data shape: {df.shape}")

    # 3) Create sequences (X: [N, seq_len, feat_dim], y: [N])
    X_val, y_val = create_sequences(df, sequence_length=30)

    if len(X_val) == 0:
        print("[EVALUATE] No valid sequences after cleaning! Check dataset preprocessing.")
        return

    feat_dim = X_val.shape[2]
    print(f"[EVALUATE] Feature dim from val data: {feat_dim}")

    # 4) Let the loader infer input_dim from checkpoint
    #    (do NOT pass input_dim here, so it uses the weights)
    model = LSTMRULModel.load(args.model, input_dim=None)

    # Just for debugging: show what input_dim the model actually uses
    print(f"[EVALUATE] Model expects input_dim={model.input_dim}")

    # 5) Optional safety check – warn if mismatch
    if model.input_dim != feat_dim:
        print("[WARN] Feature dimension mismatch!")
        print(f"       Val data has {feat_dim} features, model was trained with {model.input_dim}.")
        print("       You should retrain the model on data with the same preprocessing/columns.")
        # You *can* bail out here to avoid runtime errors:
        # return

    # 6) Predict
    preds = model.predict(X_val)

    # 7) Metrics
    mae = mean_absolute_error(y_val, preds)
    rmse = mean_squared_error(y_val, preds, squared=False)

    print(f"[EVALUATE] MAE: {mae:.4f}, RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()
