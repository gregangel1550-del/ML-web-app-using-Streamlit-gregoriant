import streamlit as st
from sqlalchemy import create_engine
import pandas as pd

def db_connect():
    # Streamlit Cloud looks for this in the "Secrets" settings
    db_url = st.secrets["DATABASE_URL"]
    engine = create_engine(db_url)
    return engine