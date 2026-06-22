import streamlit as st

def render_sidebar():
    # This CSS hides the default Streamlit auto-navigation
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none;}
        </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("Navigation")
        st.page_link("streamlit_app.py", label="Home", icon="🏠")
        st.page_link("pages/profile.py", label="Career Profile", icon="👤")
        st.divider()
        st.info("AI Career Coach - Sprint 1")