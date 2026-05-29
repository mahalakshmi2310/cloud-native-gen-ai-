import streamlit as st   # ✅ MUST BE HERE

# ================= USERS =================
USERS = {
    "admin": {"password": "admin123", "role": "admin", "label": "System Administrator"},
    "engineer": {"password": "engineer123", "role": "engineer", "label": "Maintenance Engineer"},
    "viewer": {"password": "viewer123", "role": "viewer", "label": "Operations Viewer"},
}

def require_login():

    if st.session_state.get("user"):
        return

    # ===== PAGE TITLE =====
    st.markdown("""
    <h1 style='text-align:center; margin-top:30px;'>
    AI Predictive Maintenance System
    </h1>
    """, unsafe_allow_html=True)

    # ===== CENTER CARD =====
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        # ✅ LOGIN TITLE (no link icon)
        st.markdown("""
        <h2 style='text-align:center; margin-bottom:25px;'>
         Login
        </h2>
        """, unsafe_allow_html=True)

        # ✅ Inputs (automatically centered due to column)
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("Login", use_container_width=True):
            if username in USERS and USERS[username]["password"] == password:
                st.session_state["user"] = username
                st.session_state["role"] = USERS[username]["role"]
                st.session_state["label"] = USERS[username]["label"]
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.stop()

# ================= USER PANEL (SIDEBAR) =================
def show_user_panel():
    st.sidebar.markdown(f"""
    👤 **{st.session_state['user']}**  
    {st.session_state['label']}  
    **{st.session_state['role'].upper()}**
    """)

    # ✅ LOGOUT BUTTON
    if st.sidebar.button("Logout"):
        logout()


# ================= LOGOUT =================
def logout():
    st.session_state.clear()
    st.rerun()