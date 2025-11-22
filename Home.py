
import streamlit as st

st.set_page_config(page_title="Donor Analytics", page_icon="🎯", layout="wide")
st.title("🎯 Donor Analytics")

from src.core.layout import sidebar_footer
sidebar_footer()

st.write(
    "Welcome! Use the left sidebar to navigate to **Segmentation**, **Churn**, or **LTV**.\n\n"
    "This is a scaffold. All features are wired but intentionally left as placeholders."
)
