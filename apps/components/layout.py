import streamlit as st

def header():
    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#0f172a,#020617);
        padding:22px;
        border-radius:14px;
        color:white;
        margin-bottom:20px;
    ">
      <h1 style="margin:0;">
        🏭 Cloud-Native Generative AI for Industrial Maintenance
      </h1>
      <p style="margin-top:6px; opacity:0.85;">
        Real-time Remaining Useful Life (RUL) Prediction • Kafka • LSTM • Cloud-Native
      </p>
    </div>
    """, unsafe_allow_html=True)

def footer():
    st.markdown("""
    <hr>
    <p style="text-align:center;color:#777;font-size:13px">
    © 2025 Cloud-Native Generative AI Platform • Built with Streamlit, FastAPI, Kafka & PyTorch
    </p>
    """, unsafe_allow_html=True)
