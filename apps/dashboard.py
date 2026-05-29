# ================= PATH FIX =================
import sys, os, time

from apps.components.alerts import send_email_alert, evaluate_alert
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ================= IMPORTS =================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
from apps.components.auth import require_login
from models.lstm_model import LSTMRULModel
from models.utils import create_sequences, get_ckpt_input_dim, trim_df_to_input_dim
from streamlit_lottie import st_lottie
from streamlit_autorefresh import st_autorefresh

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_49rdyysj.json"
lottie_anim = load_lottie(lottie_url)


# ================= CONFIG =================
MODEL_PATH = "models/saved/lstm_fd001.pt"
DATA_PATH = "data/processed/train.csv"

# ================= DATA LOADER (FIX) =================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

# ================= PAGE =================
st.set_page_config(page_title="AI Powered Predictive Maintenance", layout="wide")
require_login()
st_autorefresh(interval=1000000)

# ================= SESSION =================
if "alerts_log" not in st.session_state:
    st.session_state.alerts_log = []

if "preds" not in st.session_state:
    st.session_state.preds = None

# ================= SIDEBAR =================
st.sidebar.title("Control Panel")

role = st.session_state.get("role")  

if role == "admin":
    pages = ["Overview", "Predict", "Heatmap", "Live"]
elif role == "engineer":
    pages = ["Overview", "Predict", "Live"]
else:
    pages = ["Overview"]

page = st.sidebar.radio("", pages)

if page not in pages:
    st.error("Access Denied")
    st.stop()

# Display role safely
role_display = role.upper() if role else "NOT SET"

st.sidebar.info(f"""
User: {st.session_state['user']}
Role: {role_display}
""")

# ✅ LOGOUT BUTTON ADDED
st.sidebar.markdown("---")
if st.sidebar.button("Logout"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.success("Logged out successfully")
    st.rerun()

# ================= OVERVIEW =================
if page == "Overview":

    # ===== PREMIUM STYLE =====
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #cfd9df, #e2ebf0);
    }
    .card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 6px 25px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .kpi {
        padding: 15px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .kpi-blue { background: #4f46e5; }
    .kpi-green { background: #16a34a; }
    .kpi-orange { background: #f59e0b; }
    .kpi-red { background: #dc2626; }

    .title {
        font-size: 30px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    df = pd.read_csv(DATA_PATH)

    # ===== HEADER =====
    st.markdown("""
    <div class="card">
        <div class="title"> Welcome back!</div>
        <div>AI Predictive Maintenance Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    # ===== KPI VALUES =====
    total_machines = df["unit"].nunique()
    avg_rul = round(df["RUL"].mean(), 2)

    latest_df = df.groupby("unit").tail(1)

    # ✅ FIXED ALERT COUNT
    if "engine_history" in st.session_state and st.session_state.engine_history:
        history = pd.DataFrame(st.session_state.engine_history)
        alerts_count = len(history[history["status"] != "NORMAL"])
        critical_count = len(history[history["status"] == "CRITICAL"])
    else:
        alerts_count = 0
        critical_count = 0

    # ===== KPI CARDS =====
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f'<div class="kpi kpi-blue"><h4>Machines</h4><h2>{total_machines}</h2></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="kpi kpi-green"><h4>Alerts</h4><h2>{alerts_count}</h2></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="kpi kpi-orange"><h4>Avg RUL</h4><h2>{avg_rul} hrs</h2></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="kpi kpi-red"><h4>Critical</h4><h2>{critical_count}</h2></div>', unsafe_allow_html=True)

    st.markdown("###")

    if critical_count > 0:
        st.error(f"🚨 {critical_count} Machines Need Attention!")
    else:
        st.success("✅ All machines healthy")

    # ===== MAIN GRID =====
    left, right = st.columns([1.3, 1])
    trend = df.groupby("unit")["RUL"].mean().reset_index()

    fig3 = px.line(
        trend,
        x="unit",
        y="RUL",
        title="RUL Trend"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # ===== LEFT (BAR CHART) =====
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("RUL per Machine")

        sample = df.groupby("unit")["RUL"].last().head(20)

        sample = df.groupby("unit")["RUL"].last().reset_index().head(20)

        fig = px.bar(
            sample,
            x="unit",
            y="RUL",
        title="RUL per Machine"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ===== RIGHT (DONUT CHART) =====
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("Machine Health")

        critical = len(latest_df[latest_df["RUL"] < 30])
        safe = len(latest_df[latest_df["RUL"] >= 30])

        fig2 = go.Figure(data=[go.Pie(
            labels=["Critical", "Safe"],
            values=[critical, safe],
            hole=0.6
        )])

        fig2.update_layout(title="Machine Health")

        st.plotly_chart(fig2, use_container_width=True)

# ================= PREDICT =================
elif page == "Predict":

    st.markdown("## AI Prediction Engine")

    # ================= SESSION INIT =================
    if "engine_history" not in st.session_state:
        st.session_state.engine_history = []

    if "preds" not in st.session_state:
        st.session_state.preds = None

    df = pd.read_csv(DATA_PATH)

    unit_id = st.selectbox("Select Engine ID", sorted(df["unit"].unique()))
    engine_df = df[df["unit"] == unit_id]

    st.dataframe(engine_df.tail(10), height=200)

    role = st.session_state["role"]

    if role == "viewer":
        st.warning("Access restricted: Viewer cannot run predictions")

    else:
        if st.button("Run Prediction"):

            # ================= LOAD MODEL =================
            model = LSTMRULModel.load(MODEL_PATH)

            all_rul_values = []

            # ================= ALL ENGINE RUL =================
            for eng in df["unit"].unique():

                eng_df = df[df["unit"] == eng].copy()

                feature_cols = [c for c in eng_df.columns if c.startswith("sensor_")]

                input_df = eng_df[feature_cols].tail(30).copy()
                input_df = input_df.fillna(0).astype(float)

                ckpt_dim = get_ckpt_input_dim(MODEL_PATH)
                input_df, _ = trim_df_to_input_dim(input_df, ckpt_dim)

                X, _ = create_sequences(input_df, 30)

                preds_all = model.predict(X)
                rul_val = float(np.clip(preds_all[-1], 0, 300))

                all_rul_values.append(rul_val)

            # ================= CURRENT ENGINE =================
            feature_cols = [c for c in engine_df.columns if c.startswith("sensor_")]

            input_df = engine_df[feature_cols].tail(30).copy()
            input_df = input_df.fillna(0).astype(float)

            ckpt_dim = get_ckpt_input_dim(MODEL_PATH)
            input_df, _ = trim_df_to_input_dim(input_df, ckpt_dim)

            X, _ = create_sequences(input_df, 30)

            preds = model.predict(X)
            preds = np.clip(preds, 0, 300)

            st.session_state.preds = preds

            latest = float(preds[-1])

            # ================= METRIC =================
            st.metric("Latest RUL", f"{latest:.2f} hrs")

            # ================= SENSOR TRENDS =================
            st.markdown("### Sensor Trends")

            sensor_cols = [col for col in engine_df.columns if col.startswith("sensor_")]

            top_sensors = engine_df[sensor_cols].var().sort_values(ascending=False).head(5).index
            st.line_chart(engine_df[top_sensors].tail(50))

            with st.expander("Show All Sensor Trends (Advanced)"):
                fig, ax = plt.subplots(figsize=(10, 4))

                for col in sensor_cols:
                    ax.plot(engine_df[col].tail(50).values, linewidth=0.8)

                ax.set_title(f"All Sensor Trends - Engine {unit_id}")
                ax.grid(alpha=0.3)

                st.pyplot(fig)

            # ================= DYNAMIC ALERT =================
            level, alert_msg = evaluate_alert(latest, all_rul_values)

            st.markdown(f"<div class='alert'>{alert_msg}</div>", unsafe_allow_html=True)

            # ================= STORE HISTORY =================
            st.session_state.engine_history.append({
                "engine": f"Engine-{unit_id}",
                "rul": latest,
                "status": level,
                "time": pd.Timestamp.now()
            })

            # ================= EMAIL =================
            try:
                send_email_alert(
                    machine=f"Engine-{unit_id}",
                    level=level,
                    rul=latest,
                    action=alert_msg
                )
                st.success("📧 Email alert sent")
            except Exception as e:
                st.warning(str(e))

    # ================= RESULTS =================
    if st.session_state.preds is not None:

        preds = np.array(st.session_state.preds).flatten()

        df_plot = pd.DataFrame({
            "Step": range(len(preds)),
            "RUL": preds
        })

    # ================= HISTORY =================
    st.markdown("### Engine Prediction History")

    if st.session_state.engine_history:
        hist_df = pd.DataFrame(st.session_state.engine_history)
        st.dataframe(hist_df.sort_values("time", ascending=False))
# ================= HEATMAP =================
elif page == "Heatmap":

    st.title("Sensor Heatmap")

    # ===== Select Unit =====
    unit_id = st.selectbox("Select Machine / Engine", sorted(df["unit"].unique()))

    # Filter data
    filtered_df = df[df["unit"] == unit_id]

    # Select sensor columns
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

    # Take last 50 cycles (to avoid overcrowding)
    heatmap_data = filtered_df[sensor_cols].tail(50)

    # Normalize (better visualization)
    heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

    # ===== Plot =====
    fig, ax = plt.subplots(figsize=(9, 3))

    sns.heatmap(
        heatmap_data,
        cmap="RdYlGn_r",
        cbar=True,
        ax=ax
    )

    ax.set_title(f"Sensor Behavior (Last 50 Cycles) - Engine {unit_id}")
    ax.set_xlabel("Sensors")
    ax.set_ylabel("Time (Last Cycles)")

    st.pyplot(fig)

# ================= LIVE =================
elif page == "Live":

    st.title("Live Monitoring")

    placeholder = st.empty()

    if st.button("Start"):

        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["Temp","Vibration","Pressure"])

        for _ in range(50):
            new_row = pd.DataFrame(np.random.randn(1,3), columns=chart_data.columns)
            chart_data = pd.concat([chart_data, new_row]).tail(50)

            with placeholder.container():
                st.line_chart(chart_data)

            time.sleep(0.5)

