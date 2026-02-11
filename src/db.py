import streamlit as st
from sqlalchemy import create_engine


def db_connect():
    db_url = st.secrets["DATABASE_URL"]
    return create_engine(db_url)