import pandas as pd
import os

# -------------------------
# CMAPSS Loader + Preprocess
# -------------------------
def load_cmapss(filepath: str) -> pd.DataFrame:
    """Load CMAPSS engine dataset (FD001)."""
    # Define column names
    col_names = ["unit", "time",
                 "op1", "op2", "op3"] + [f"sensor_{i}" for i in range(1, 22)]
    df = pd.read_csv(filepath, sep=" ", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = col_names
    return df

def preprocess_cmapss(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize sensor data + add RUL estimation."""
    df["RUL"] = df.groupby("unit")["time"].transform(max) - df["time"]
    sensor_cols = [c for c in df.columns if "sensor_" in c]
    df[sensor_cols] = (df[sensor_cols] - df[sensor_cols].min()) / (df[sensor_cols].max() - df[sensor_cols].min())
    return df


# -------------------------
# AI4I Loader + Preprocess
# -------------------------
def load_ai4i(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def preprocess_ai4i(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["Air temperature [K]", "Process temperature [K]",
                    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    return df


# -------------------------
# Predictive Maintenance Loader + Preprocess
# -------------------------
def load_predictive(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def preprocess_predictive(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="number").columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    df = df.fillna(method="ffill")
    return df


# -------------------------
# Main Orchestrator
# -------------------------
def run_all_preprocessing():
    os.makedirs("data/processed", exist_ok=True)

    # CMAPSS
    cmapss_raw = "data/raw/CMAPSS/train_FD001.txt"
    df_cmapss = preprocess_cmapss(load_cmapss(cmapss_raw))
    df_cmapss.to_csv("data/processed/cmapss_processed.csv", index=False)
    print("[INFO] CMAPSS processed → data/processed/cmapss_processed.csv")

    # AI4I
    ai4i_raw = "data/raw/ai4i2020.csv"
    df_ai4i = preprocess_ai4i(load_ai4i(ai4i_raw))
    df_ai4i.to_csv("data/processed/ai4i2020_processed.csv", index=False)
    print("[INFO] AI4I processed → data/processed/ai4i2020_processed.csv")

    # Predictive Maintenance
    pred_raw = "data/raw/predictive_maintenance.csv"
    df_pred = preprocess_predictive(load_predictive(pred_raw))
    df_pred.to_csv("data/processed/predictive_processed.csv", index=False)
    print("[INFO] Predictive Maintenance processed → data/processed/predictive_processed.csv")

    print("\n✅ All datasets processed successfully!")

if __name__ == "__main__":
    run_all_preprocessing()
