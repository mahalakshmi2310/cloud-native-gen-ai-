# scripts/inspect_checkpoint.py
import torch, sys, os
path = sys.argv[1] if len(sys.argv) > 1 else "models/saved/lstm_fd001.pt"
if not os.path.exists(path):
    print("Checkpoint not found:", path); sys.exit(1)

ckpt = torch.load(path, map_location="cpu")
print("Keys in checkpoint:", list(ckpt.keys())[:50])
# Common saved form: state_dict only, or dict with 'model_state'
if "state_dict" in ckpt:
    state = ckpt["state_dict"]
else:
    state = ckpt

for k in ["lstm.weight_ih_l0", "lstm.weight_hh_l0", "fc.0.weight"]:
    if k in state:
        tensor = state[k]
        print(f"{k}: shape {tuple(tensor.size())}")
    else:
        print(f"{k}: NOT FOUND in state_dict (available keys head): {list(state.keys())[:10]}")
