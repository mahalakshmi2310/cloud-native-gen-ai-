import argparse
import pandas as pd
from models.lstm_model import LSTMRULModel
from models.utils import create_sequences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to processed training data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="models/saved/lstm_fd001.pt")
    args = parser.parse_args()

    # Load processed data
    df = pd.read_csv(args.dataset)

    # 🔧 Clean dataset
    df = df.dropna(axis=1, how="all")  # drop all-NaN columns
    df = df.fillna(0)  # replace any partial NaNs with 0
    print(f"[CLEAN] Training data shape: {df.shape}")

    # Create sequences
    X, y = create_sequences(df, sequence_length=30)

    # Train model
    model = LSTMRULModel(input_dim=X.shape[2])
    model.train(X, y, epochs=args.epochs, save_path=args.save_path)

    print(f"[INFO] Model trained and saved to {args.save_path}")

if __name__ == "__main__":
    main()
