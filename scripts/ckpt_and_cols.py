# scripts/ckpt_and_cols.py
import torch, json, pandas as pd, os, sys
ckpt_path = sys.argv[1] if len(sys.argv)>1 else "models/saved/lstm_fd001.pt"
csv_path = sys.argv[2] if len(sys.argv)>2 else "data/processed/train.csv"

print("Checkpoint:", ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu")
# extract state_dict
if isinstance(ckpt, dict):
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
else:
    state = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt

# find LSTM weight key
lstm_keys = [k for k in state.keys() if "weight_ih_l0" in k]
if lstm_keys:
    k = lstm_keys[0]
    w = state[k]
    print(f"Found LSTM weight key: {k} shape: {tuple(w.shape)}")
    print("=> inferred input_dim from ckpt:", w.shape[1])
else:
    print("No lstm.weight_ih_l0 found in checkpoint keys. Sample keys:", list(state.keys())[:20])

print("\nProcessed CSV columns (first 200):")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("CSV shape:", df.shape)
    print(df.columns.tolist()[:200])
else:
    print("CSV not found:", csv_path)
print("\nTo generate feature_list.json, run:")
print(" python scripts/generate_feature_list.py")
print("To inspect checkpoint details, run:")
print(" python scripts/inspect_checkpoint.py", ckpt_path)
print("To test sequence creation, run:")
print(" python scripts/test_sequences.py")
print("\nDone.")