import argparse
import numpy as np
import pandas as pd
from models.lstm_model import LSTMRULModel
from models.utils import create_sequences
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.pt)")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset to test on (CSV)")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length for LSTM input")
    args = parser.parse_args()

    # You MUST update this to match the number of features after cleaning
    input_dim = 20  

    # Load model
    model = LSTMRULModel.load(args.model, input_dim=input_dim)
    print(f"[INFO] Model loaded from {args.model}")

    if args.dataset:
        # Run evaluation on provided dataset
        df = pd.read_csv(args.dataset).dropna(axis=1, how="any")  # drop NaN cols
        X, y = create_sequences(df, sequence_length=args.seq_len)

        preds = model.predict(X[:5])  # predict first 5 samples
        print("[INFO] Predictions on dataset (first 5):", preds)
    else:
        # Run dummy prediction
        X_dummy = np.random.rand(1, args.seq_len, input_dim)
        pred = model.predict(X_dummy)
        print("[INFO] Dummy prediction:", pred)


if __name__ == "__main__":
    main()
