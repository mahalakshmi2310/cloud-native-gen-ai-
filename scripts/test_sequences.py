# scripts/test_sequences.py
import sys, os
# ensure project root is on sys.path so "models" and other packages import correctly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import pandas as pd
from models.utils import create_sequences

P = os.path.join(ROOT, "payload.json")
if not os.path.exists(P):
    print("ERROR: payload.json not found at", P)
    raise SystemExit(1)

# load payload
try:
    payload = json.load(open(P))
except Exception as e:
    print("ERROR reading payload.json:", e)
    raise

print("payload rows:", len(payload))
if len(payload) == 0:
    print("payload is empty — regenerate with scripts/generate_payload.py")
    raise SystemExit(1)

print("sample keys:", list(payload[0].keys())[:50])

# convert to DataFrame like API does
df = pd.DataFrame(payload)
print("raw df shape:", df.shape)
df = df.dropna(axis=1, how="all")
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
print("clean df shape:", df.shape)
print("columns:", list(df.columns))

# run create_sequences
try:
    X, y = create_sequences(df, sequence_length=30)
    print("create_sequences -> X type:", type(X), "X.shape:", getattr(X, "shape", None))
    print("create_sequences -> y type:", type(y), "y.shape:", getattr(y, "shape", None))
except Exception as e:
    print("create_sequences raised:", e)
    raise
