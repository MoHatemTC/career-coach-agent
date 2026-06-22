import streamlit as st
from app.components.sidebar import render_sidebar
from app.db.connection import init_db
from app.db.users import seed_sample_user, get_all_users

st.set_page_config(page_title="AI Career Coach", layout="wide")

# 1. Initialize DB and inject the sample user
init_db()
seed_sample_user()

render_sidebar()

st.title("AI Career Coach")
st.write("Welcome! Please select an existing user or create a new one to continue.")

# 2. Session-State User Selection Flow
st.subheader("User Login")
existing_users = get_all_users()

col1, col2 = st.columns(2)
with col1:
    selected_user = st.selectbox("Select an existing user:", ["-- Select --"] + existing_users)
with col2:
    new_user = st.text_input("Or create a new username:")

if st.button("Access Profile"):
    if new_user:
        st.session_state['username'] = new_user.strip()
        st.success(f"Welcome, {st.session_state['username']}! Navigate to your Career Profile.")
    elif selected_user != "-- Select --":
        st.session_state['username'] = selected_user
        st.success(f"Logged in as {st.session_state['username']}. Navigate to your Career Profile.")
    else:
        st.error("Please select or enter a username to proceed.")

if 'username' in st.session_state:
    st.info(f"**Current Session:** {st.session_state['username']}")