# scripts/inspect_ckpt.py
import torch, sys, os
path = "models/saved/lstm_fd001.pt"  # adjust if needed
if not os.path.exists(path):
    print("Checkpoint not found:", path); sys.exit(1)

ckpt = torch.load(path, map_location="cpu")
# try to get a state_dict object
if isinstance(ckpt, dict):
    state = None
    for key in ("model_state_dict","state_dict","model"):
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]; break
    if state is None:
        # maybe ckpt itself is a state_dict
        state = ckpt
else:
    # object with state_dict method
    if hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    else:
        print("Unknown checkpoint format"); sys.exit(1)

# find LSTM weight_ih_l0
keys = [k for k in state.keys() if "weight_ih_l0" in k]
if not keys:
    print("No weight_ih_l0 found in checkpoint keys. Keys sample:", list(state.keys())[:20])
else:
    key = keys[0]
    w = state[key]
    print("Found key:", key)
    print("weight_ih_l0 shape (out,input):", tuple(w.shape))
    print("=> checkpoint input_dim =", w.shape[1])
