import argparse
import pandas as pd
from models.lstm_model import LSTMRULModel
from models.utils import create_sequences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X, y = create_sequences(df, sequence_length=30)

    model = LSTMRULModel.load(args.model)
    preds = model.predict(X)

    for i, p in enumerate(preds[:10]):
        print(f"[PRED] Sample {i} → RUL: {p:.2f}")

if __name__ == "__main__":
    main()
