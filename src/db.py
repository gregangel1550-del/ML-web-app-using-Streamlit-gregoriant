import streamlit as st
from sqlalchemy import create_engine


def db_connect():
    """
    Crea conexión a la base de datos usando secrets de Streamlit
    """
    db_url = st.secrets["DATABASE_URL"]
    engine = create_engine(db_url)
    return engine