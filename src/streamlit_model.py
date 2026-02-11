# src/utils.py

import pickle
import pandas as pd
from pathlib import Path
import streamlit as st
from utils import cargar_modelo, predecir
from streamlit_model import vista_modelo

MODELOS_PATH = Path("src")


def cargar_modelo(ciudad: str):
    """
    Carga el modelo .pkl según la ciudad seleccionada
    """
    modelo_file = MODELOS_PATH / f"modelo_{ciudad}.pkl"

    if not modelo_file.exists():
        raise FileNotFoundError(f"No se encontró el modelo para {ciudad}")

    with open(modelo_file, "rb") as f:
        modelo = pickle.load(f)

    return modelo


def predecir(modelo, pasos: int):
    """
    Realiza predicciones usando el modelo de series temporales
    """
    # Compatibilidad con varios frameworks
    if hasattr(modelo, "forecast"):
        pred = modelo.forecast(steps=pasos)
    elif hasattr(modelo, "predict"):
        pred = modelo.predict(n_periods=pasos)
    else:
        raise AttributeError("El modelo no soporta forecast ni predict")

    return pd.Series(pred)


# src/streamlit_model.py



def vista_modelo():
    st.header("📈 Predicción con modelos de Series Temporales")

    st.markdown(
        """
        Esta aplicación utiliza modelos de **series temporales entrenados previamente**
        para realizar predicciones por ciudad.
        """
    )

    ciudad = st.selectbox(
        "Selecciona una ciudad",
        ["madrid", "barcelona", "valencia", "alicante", "castellon"]
    )

    pasos = st.slider(
        "Horizonte de predicción (periodos futuros)",
        min_value=1,
        max_value=60,
        value=12
    )

    if st.button("🔮 Generar predicción"):
        with st.spinner("Cargando modelo y calculando predicción..."):
            try:
                modelo = cargar_modelo(ciudad)
                predicciones = predecir(modelo, pasos)

                df_pred = pd.DataFrame({
                    "Periodo": range(1, pasos + 1),
                    "Predicción": predicciones.values
                })

                st.success("Predicción generada correctamente")

                st.subheader("📊 Resultados")
                st.dataframe(df_pred, use_container_width=True)

                st.subheader("📉 Visualización")
                st.line_chart(df_pred.set_index("Periodo"))

                st.download_button(
                    label="📥 Descargar predicciones (CSV)",
                    data=df_pred.to_csv(index=False),
                    file_name=f"prediccion_{ciudad}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error al generar la predicción: {e}")


# src/app.py



st.set_page_config(
    page_title="ML Time Series Forecast App",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Aplicación de Predicción con Series Temporales")
st.markdown(
    """
    Bienvenido a la aplicación de **Machine Learning para predicción temporal**.  
    Selecciona una ciudad y obtén predicciones basadas en modelos entrenados.
    """
)

menu = st.sidebar.radio(
    "Navegación",
    ["Predicción de Series Temporales"]
)

if menu == "Predicción de Series Temporales":
    vista_modelo()

st.sidebar.markdown("---")
st.sidebar.info(
    "Desarrollado con **Python + Streamlit**\n\n"
    "Modelos de Series Temporales entrenados previamente."
)