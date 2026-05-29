import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# KPI CARDS
# =====================================================
def show_kpis(preds, threshold=30.0):
    latest = float(preds[-1])
    prev = float(preds[-2]) if len(preds) > 1 else latest
    delta = latest - prev

    status = "CRITICAL ⚠️" if latest < threshold else "HEALTHY ✅"
    risk = "HIGH" if latest < threshold else "LOW"

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Remaining Useful Life (RUL)",
        f"{latest:.2f}",
        f"{delta:+.2f}",
    )
    col2.metric("System Status", status)
    col3.metric("Risk Level", risk)


# =====================================================
# RUL TREND
# =====================================================
def plot_rul_trend(preds):
    plt.figure(figsize=(10, 4))
    plt.plot(preds, marker="o", linewidth=2, color="#2563eb")
    plt.title("RUL Degradation Trend")
    plt.xlabel("Prediction Window")
    plt.ylabel("RUL")
    plt.grid(alpha=0.3)
    st.pyplot(plt.gcf())


# =====================================================
# CONFIDENCE / UNCERTAINTY BAND
# =====================================================
def plot_rul_with_confidence(preds, window=3):
    preds = np.array(preds)

    std = np.std(preds[-window:]) if len(preds) >= window else np.std(preds)
    upper = preds + std
    lower = preds - std

    plt.figure(figsize=(10, 4))
    plt.plot(preds, label="Predicted RUL", color="#2563eb", linewidth=2)
    plt.fill_between(
        range(len(preds)),
        lower,
        upper,
        color="#93c5fd",
        alpha=0.4,
        label="Confidence Band",
    )
    plt.title("RUL Prediction with Uncertainty")
    plt.xlabel("Prediction Window")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(alpha=0.3)
    st.pyplot(plt.gcf())


# =====================================================
# SENSOR HEATMAP (NO SEABORN)
# =====================================================
def plot_sensor_heatmap(df):
    st.subheader("🌡 Sensor Activity Heatmap")

    values = df.values.T

    plt.figure(figsize=(10, 4))
    plt.imshow(values, aspect="auto", cmap="viridis")
    plt.colorbar(label="Sensor Value")
    plt.xlabel("Time Step")
    plt.ylabel("Sensor Index")
    plt.title("Sensor Value Distribution (Recent Window)")
    st.pyplot(plt.gcf())


# =====================================================
# SENSOR TRENDS
# =====================================================
def plot_sensor_trends(df, top_n=5):
    st.subheader("📉 Sensor Trends (Top Influential Sensors)")

    variances = df.var().sort_values(ascending=False)
    top_sensors = variances.head(top_n).index.tolist()

    plt.figure(figsize=(10, 4))
    for col in top_sensors:
        plt.plot(df[col].values, label=col, linewidth=2)

    plt.title("Top Sensor Value Trends")
    plt.xlabel("Time Step")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(alpha=0.3)
    st.pyplot(plt.gcf())


# =====================================================
# PREDICTION ERROR ANALYSIS
# =====================================================
def plot_prediction_error(preds):
    diffs = np.diff(preds)

    plt.figure(figsize=(10, 4))
    plt.plot(diffs, marker="o", color="red", linewidth=2)
    plt.axhline(0, linestyle="--", color="black", alpha=0.6)
    plt.title("Change in RUL Between Predictions")
    plt.xlabel("Prediction Window")
    plt.ylabel("Δ RUL")
    plt.grid(alpha=0.3)
    st.pyplot(plt.gcf())


# =====================================================
# EXPLAINABILITY (WHY RUL CHANGED)
# =====================================================
def explain_rul_drop(df, top_n=5):
    st.subheader("🧠 Explainability: Why RUL Changed")

    deltas = df.diff().abs().mean().sort_values(ascending=False)
    top_features = deltas.head(top_n)

    for sensor, score in top_features.items():
        st.write(f"• **{sensor}** → impact score: {score:.3f}")

    plt.figure(figsize=(8, 3))
    plt.bar(top_features.index, top_features.values, color="#dc2626")
    plt.title("Top Sensor Contributions to RUL Change")
    plt.ylabel("Impact Score")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
