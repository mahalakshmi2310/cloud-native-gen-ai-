import streamlit as st
import pandas as pd
import numpy as np

from models.lstm_model import LSTMRULModel
from models.utils import create_sequences, get_ckpt_input_dim, trim_df_to_input_dim

st.set_page_config(layout="wide")

# ================= LOAD DATA =================
DATA_PATH = "data/processed/train.csv"
df = pd.read_csv(DATA_PATH)

# ================= MODEL =================
MODEL_PATH = "models/saved/lstm_fd001.pt"
model = LSTMRULModel.load(MODEL_PATH)

# ================= SENSOR COLUMNS =================
feature_cols = [col for col in df.columns if col.startswith("sensor_")]

TEMP_COL = "sensor_2"
VIB_COL = "sensor_3"
PRESS_COL = "sensor_7"

# ================= STATUS FUNCTIONS =================
def get_status(value, warn, crit):
    if value > crit:
        return "🔴 Critical"
    elif value > warn:
        return "🟡 Warning"
    return "🟢 Normal"

# ================= HEADER =================
st.markdown("## ⚙️ Engine Monitoring Dashboard")
st.caption("AI-powered predictive maintenance using real sensor data")

# ================= SELECT ENGINE =================
unit_id = st.selectbox("Select Engine ID", sorted(df["unit"].unique()))

data = df[df["unit"] == unit_id]
latest = data.tail(1)

# ================= SENSOR VALUES =================
temp = float(latest[TEMP_COL].values[0])
vib = float(latest[VIB_COL].values[0])
press = float(latest[PRESS_COL].values[0])

# ================= AI RUL FIXED =================
rul = 0  # default

try:
    # ONLY SENSOR DATA (IMPORTANT FIX)
    input_df = data[feature_cols].tail(30).copy()

    # remove nulls
    input_df = input_df.fillna(0).astype(float)

    # match model input shape
    ckpt_dim = get_ckpt_input_dim(MODEL_PATH)
    input_df, _ = trim_df_to_input_dim(input_df, ckpt_dim)

    # create sequences
    X, _ = create_sequences(input_df, 30)

    # predict
    preds = model.predict(X)

    rul = float(np.clip(preds[-1], 0, 300))

    # DEBUG (optional)
    # st.write("Pred sample:", preds[-5:])

except Exception as e:
    st.error(f"Prediction error: {e}")

# ================= METRICS =================
c1, c2, c3, c4 = st.columns(4)

c1.metric("🤖 AI RUL", f"{rul:.2f} hrs")
c2.metric("🌡 Temperature", f"{temp:.2f}")
c3.metric("📊 Vibration", f"{vib:.2f}")
c4.metric("📡 Pressure", f"{press:.2f}")

# ================= STATUS =================
st.markdown("### 🔧 Status")

st.write("Temperature:", get_status(temp, 0.5, 0.75))
st.write("Vibration:", get_status(vib, 0.45, 0.65))
st.write("Pressure:", get_status(press, 0.5, 0.7))

# ================= SMART ALERT =================
temp_flag = temp > 0.75
vib_flag = vib > 0.65
press_flag = press > 0.7

sensor_critical = temp_flag or vib_flag or press_flag

if sensor_critical:
    if rul < 30:
        st.error("🚨 Critical: Failure imminent!")
    else:
        st.warning("⚠️ Sensor anomaly detected")

else:
    if rul < 30:
        st.warning("⚠️ Low RUL but stable sensors")
    else:
        st.success("🟢 System operating normally")

# ================= TREND =================
st.markdown("### 📊 Sensor Trends (Last 50 Cycles)")
st.line_chart(data[[TEMP_COL, VIB_COL, PRESS_COL]].tail(50))