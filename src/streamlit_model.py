import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# -------------------------
# Config
# -------------------------
st.set_page_config(
    page_title="Population Forecast",
    layout="centered"
)

st.title("📈 Population Forecast by City")

# -------------------------
# Paths
# -------------------------
BASE_PATH = os.path.dirname(__file__)

MODELS = {
    "Alicante": "modelo_alicante.pkl",
    "Barcelona": "modelo_barcelona.pkl",
    "Castellón": "modelo_castellon.pkl",
    "Madrid": "modelo_madrid.pkl",
    "Valencia": "modelo_valencia.pkl",
}

DATA_FILE = "df_final_poblacion.csv"

# -------------------------
# Loaders
# -------------------------
@st.cache_data
def load_data():
    file_path = os.path.join(BASE_PATH, DATA_FILE)
    return pd.read_csv(file_path)

@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(BASE_PATH, model_name)
    return joblib.load(model_path)

# -------------------------
# UI
# -------------------------
city = st.selectbox("🏙️ Select a city", list(MODELS.keys()))
steps = st.slider("🔮 Forecast years", min_value=1, max_value=20, value=5)

# -------------------------
# Load model
# -------------------------
model = load_model(MODELS[city])

# -------------------------
# Forecast
# -------------------------
forecast = model.forecast(steps=steps)
forecast = pd.Series(forecast, name="Forecast")

# -------------------------
# Load historical data (optional but recommended)
# -------------------------
try:
    df = load_data()
    city_data = df[df["city"] == city]

    y_hist = city_data["population"].values
    x_hist = range(len(y_hist))
    x_forecast = range(len(y_hist), len(y_hist) + steps)

    fig, ax = plt.subplots()
    ax.plot(x_hist, y_hist, label="Historical", marker="o")
    ax.plot(x_forecast, forecast, label="Forecast", marker="o")

    ax.set_title(f"Population Forecast – {city}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.legend()

    st.pyplot(fig)

except Exception:
    st.warning("Historical data not found. Showing forecast only.")
    st.line_chart(forecast)

# -------------------------
# Show forecast table
# -------------------------
st.subheader("📊 Forecast values")
st.dataframe(forecast.reset_index(drop=True))