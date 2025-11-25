import pandas as pd
import streamlit as st

from src.core.state import get_api_client

st.caption("This page will estimate the next-12-month LTV per donor once implemented.")

st.info("TODO: Implement things related to donor life time value (e.g. use RandomForestRegressor).")


# api usage example:
api = get_api_client()
donations = api.get_donations()
df = pd.DataFrame(donations)

# execute this command to run app: python -m streamlit run main.py