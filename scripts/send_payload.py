# scripts/send_payload.py
import os, json, requests, sys

API_URL = os.environ.get("PRED_API", "http://127.0.0.1:8000/predict")
PAYLOAD_PATH = "payload.json"
SEQ_LEN = 30

if not os.path.exists(PAYLOAD_PATH):
    print(f"ERROR: {PAYLOAD_PATH} not found. Run scripts/generate_payload.py first.")
    sys.exit(1)

with open(PAYLOAD_PATH, "r") as f:
    payload = json.load(f)

print(f"Sending {len(payload)} rows to {API_URL}?seq_len={SEQ_LEN} ...")
try:
    resp = requests.post(API_URL, json=payload, params={"seq_len": SEQ_LEN}, timeout=60)
    print("status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)
except Exception as e:
    print("Request failed:", e)
print("Done.")