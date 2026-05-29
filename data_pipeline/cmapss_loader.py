# data_pipeline/cmapss_loader.py
import pandas as pd

def load_cmapss(filepath: str) -> pd.DataFrame:
    """Load CMAPSS dataset into DataFrame."""
    col_names = [
        "unit_number", "time_in_cycles",
        "operational_setting_1", "operational_setting_2", "operational_setting_3"
    ] + [f"sensor_{i}" for i in range(1, 22)]  # 21 sensors

    df = pd.read_csv(filepath, sep=" ", header=None)
    df.dropna(axis=1, how="all", inplace=True)  # drop empty cols
    df.columns = col_names
    return df


def preprocess_cmapss(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess CMAPSS: add normalized sensors & RUL calculation."""
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycle - df["time_in_cycles"]

    # Normalize some sensors
    for sensor in ["sensor_2", "sensor_3", "sensor_4"]:
        df[f"{sensor}_norm"] = df[sensor] / df[sensor].max()

    return df


if __name__ == "__main__":
    raw_path = "data/raw/CMAPSS/train_FD001.txt"
    processed_path = "data/processed/train_FD001_processed.csv"

    df_raw = load_cmapss(raw_path)
    df_processed = preprocess_cmapss(df_raw)
    df_processed.to_csv(processed_path, index=False)

    print(f"[INFO] Saved processed data → {processed_path}")
