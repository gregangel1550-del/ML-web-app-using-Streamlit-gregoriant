import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import pydeck as pdk
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

@st.cache_data
def load_data():
    # Get the directory of the current script (src folder)
    base_path = os.path.dirname(__file__)
    # Combine it with the filename
    file_path = os.path.join(base_path, "df_final_poblacion.csv")
    
    return pd.read_csv(file_path)

# PAGE CONFIG

st.set_page_config(page_title="Series Temporales - Delitos",
                   page_icon="📈",
                   layout="wide")

st.title("📊 Análisis y Predicción de Delitos (Series Temporales)")
st.markdown(""" Esta aplicación permite analizar la evolución trimestral 
            de delitos y realizar **predicciones futuras**
            usando modelos de **series temporales**.
            """)

# LOAD DATA

df = load_data()


# SIDEBAR CONTROLS

st.sidebar.header("🔧 Filtros de análisis")

municipio = st.sidebar.selectbox("Seleccione Municipio",
                                 sorted(df["Municipio"].unique()))

delito = st.sidebar.selectbox("Seleccione el tipo de delito",
                              sorted(df["Delitos"].unique()))

forecast_steps = st.sidebar.slider("Trimestres de previsión",
                                   min_value=4,
                                   max_value=20,
                                   value=8)

normalize_pop = st.sidebar.checkbox("Normalizar por población (por cada 100.000 habitantes)")


# FILTER DATA

data = df[(df["Municipio"] == municipio) &
          (df["Delitos"] == delito)].copy()

# Create datetime index
data["Fecha"] = pd.to_datetime(data["Año"].astype(str) + "-" +
                               (data["Trimestre"] * 3).astype(str))

data = data.sort_values("Fecha")
data.set_index("Fecha", inplace=True)
serie = data["Valor trimestral"]



# Target variable
y = data["Valor trimestral"]
# 1. Convert Poblacion to numeric, turning errors into NaN
data["Poblacion"] = pd.to_numeric(data["Poblacion"], errors='coerce')

# 2. Drop rows where Poblacion is 0 or NaN to avoid division by zero/errors
data = data[data["Poblacion"] > 0]

if normalize_pop:
    y = (y / data["Poblacion"]) * 100000
    y.name = "Crimes per 100k inhabitants"


# SHOW RAW DATA

with st.expander("📄 View filtered data"):
    st.dataframe(data)


# TIME SERIES PLOT

st.subheader("📉 Serie temporal histórica")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(serie, marker="o")
ax.set_xlabel("Fecha")
ax.set_ylabel("Delitos trimestrales")
ax.grid(True)

st.pyplot(fig)


# MODEL TRAINING

st.subheader("🤖 Predicción con ARIMA")

if st.button("Entrenar modelo y predecir"):
    try:
        modelo = ARIMA(serie, order=(1, 1, 1))
        modelo_fit = modelo.fit()

        pasos = 4  # 1 año (4 trimestres)
        prediccion = modelo_fit.forecast(steps=pasos)

        fechas_futuras = pd.date_range(
            start=serie.index[-1],
            periods=pasos + 1,
            freq="Q")[1:]
  
# FORECAST PLOT
    
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(serie, label="Histórico")
        ax.plot(fechas_futuras, prediccion, label="Predicción", linestyle="--")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)


        st.subheader("📋 Valores predichos")

        df_pred = pd.DataFrame({"Fecha": fechas_futuras,
                                "Predicción de delitos": prediccion.values})

        st.dataframe(df_pred)

    except Exception as e:
        st.error(f"Error en el modelo: {e}")
    
# FOOTER

st.markdown("---")
st.markdown("Built with using **Streamlit**")