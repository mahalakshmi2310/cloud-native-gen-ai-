# apps/api.py
import time
import os
import json
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from models.lstm_model import LSTMRULModel
from models.utils import create_sequences  # must exist in models/utils.py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predict-api")

app = FastAPI(title="Predictive Maintenance API (With X.shape logging)")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/saved/lstm_fd001.pt")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0")
DEFAULT_SEQ_LEN = int(os.environ.get("SEQ_LEN", "30"))
FEATURES_PATH = os.environ.get("FEATURES_PATH", "models/feature_list.json")


# load feature list (optional)
_feature_list = None
if os.path.exists(FEATURES_PATH):
    try:
        with open(FEATURES_PATH, "r") as f:
            obj = json.load(f)
            _feature_list = obj.get("features")
            logger.info("Loaded feature list (len=%d) from %s", len(_feature_list), FEATURES_PATH)
    except Exception as e:
        logger.warning("Could not load features: %s", e)


@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running 🚀", "model_path": MODEL_PATH, "model_version": MODEL_VERSION}


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH, "model_version": MODEL_VERSION}


@app.get("/model-info")
def model_info():
    return {"model_path": MODEL_PATH, "model_version": MODEL_VERSION, "seq_len": DEFAULT_SEQ_LEN, "features": _feature_list}


@app.post("/predict")
def predict(payload: List[Dict[str, Any]], seq_len: int = Query(DEFAULT_SEQ_LEN, gt=0), model_path: str = Query(MODEL_PATH)):
    start = time.time()

    # Basic checks
    if not isinstance(payload, list) or len(payload) == 0:
        raise HTTPException(status_code=400, detail="Payload must be a non-empty JSON array of objects (rows).")

    # Convert to DataFrame
    try:
        df = pd.DataFrame(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to parse JSON payload into table: {e}")

    # Drop empty columns and coerce numeric columns
    df = df.dropna(axis=1, how="all")
    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No valid columns in payload after dropping all-NaN columns.")

    df_numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    if len(df_numeric) < seq_len:
        raise HTTPException(status_code=400, detail=f"Not enough rows to build a sequence: need seq_len={seq_len}, got {len(df_numeric)}")

    # Align to feature list if available
    if _feature_list:
        cols = [c for c in _feature_list if c in df_numeric.columns]
        if len(cols) == 0:
            raise HTTPException(status_code=400, detail="None of the saved feature columns are present in the payload.")
        df_for_model = df_numeric[cols]
    else:
        df_for_model = df_numeric

    # Build sequences (X, y) using helper that your repo should have
    try:
        X, y = create_sequences(df_for_model, sequence_length=seq_len)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create sequences for model input: {e}")

    # Log shapes BEFORE loading the model
    logger.info("Prepared sequences for model. X.shape=%s, y.shape=%s, model_path=%s", getattr(X, "shape", None), getattr(y, "shape", None), model_path)

    # Load model with explicit input_dim to avoid mismatch
    try:
        input_dim = int(X.shape[2])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot determine input_dim from X: {e}")

    try:
        model = LSTMRULModel.load(model_path, input_dim=input_dim)
    except Exception as e:
        logger.exception("Model load failed")
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    # Run prediction
    try:
        preds = model.predict(X)
    except Exception as e:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    latency_ms = int((time.time() - start) * 1000)
    result = {"predictions": preds.tolist() if hasattr(preds, "tolist") else list(preds), "windows_produced": len(preds), "latency_ms": latency_ms}
    logger.info("predict: rows=%d cols=%d windows=%d latency_ms=%d", len(df_for_model), df_for_model.shape[1], len(preds), latency_ms)
    return JSONResponse(status_code=200, content=result)
