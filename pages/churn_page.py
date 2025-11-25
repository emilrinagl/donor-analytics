#Shanmukh
import streamlit as st
import pandas as pd

from src.core.state import get_api_client

st.caption("This page will train/predict churn probabilities once implemented.")

st.info("TODO: Implement things related to the churn metric (how likely it is that a donor stopped donating).")

# api usage example:
api = get_api_client()
donations = api.get_donations()
df = pd.DataFrame(donations)

# execute this command to run app: python -m streamlit run main.py