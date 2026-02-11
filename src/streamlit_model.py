import streamlit as st
import pandas as pd
import job pickle
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Series Temporales - Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- FUNCIONES DE CARGA ---
@st.cache_resource
def load_model(city_name):
    """
    Carga el modelo SARIMAX desde el archivo .pkl correspondiente.
    Utiliza cache_resource para mantener el modelo en memoria.
    """
    file_path = f"modelo_{city_name.lower()}.pkl"
    
    # Verificación de existencia del archivo
    if not os.path.exists(file_path):
        st.error(f"Error: El archivo {file_path} no se encuentra en el directorio raíz.")
        return None
    
    try:
        # Los archivos de statsmodels suelen requerir carga mediante su propio método
        # o vía pickle estándar si fueron guardados con el wrapper de resultados.
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo de {city_name}: {e}")
        return None

# --- INTERFAZ DE USUARIO (SIDEBAR) ---
st.sidebar.image("image_4929b4.png", use_container_width=True)
st.sidebar.title("Configuración de Predicción")

city = st.sidebar.selectbox(
    "Seleccione la Ciudad:",
    ["Madrid", "Barcelona", "Valencia", "Alicante", "Castellon"]
)

steps = st.sidebar.slider(
    "Horizonte de predicción (Pasos):",
    min_value=1,
    max_value=24,
    value=12,
    help="Número de periodos a futuro que desea predecir."
)

# --- CUERPO PRINCIPAL ---
st.title(f"Análisis Predictivo: {city}")
st.markdown(f"""
Este dashboard utiliza modelos **SARIMAX** entrenados específicamente para la ciudad de **{city}**. 
Seleccione los parámetros en el panel de la izquierda para generar proyecciones.
""")

# Carga del modelo seleccionado
model_results = load_model(city)

if model_results:
    # Crear pestañas para organizar la información
    tab1, tab2, tab3 = st.tabs(["📊 Predicción", "🔍 Diagnóstico del Modelo", "📄 Datos Técnicos"])

    with tab1:
        st.subheader(f"Pronóstico para los próximos {steps} periodos")
        
        # Generar predicción
        forecast = model_results.get_forecast(steps=steps)
        forecast_df = forecast.summary_frame()
        
        # Layout de columnas para métricas rápidas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de la predicción
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Intentar graficar los últimos datos históricos si están disponibles en el modelo
            try:
                model_results.fittedvalues.iloc[-20:].plot(ax=ax, label="Histórico Reciente", color="gray", linestyle="--")
            except:
                pass
                
            forecast_df['mean'].plot(ax=ax, label="Predicción", color="#1f77b4", marker='o')
            ax.fill_between(forecast_df.index, 
                            forecast_df['mean_ci_lower'], 
                            forecast_df['mean_ci_upper'], 
                            color='k', alpha=0.1, label="Intervalo de Confianza")
            
            ax.set_title(f"Proyección de Tendencia - {city}")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.write("Valores Predichos")
            st.dataframe(forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].style.format("{:.2f}"))

    with tab2:
        st.subheader("Análisis de Residuos")
        # Statsmodels provee una función de diagnóstico automática
        fig_diag = model_results.plot_diagnostics(figsize=(10, 8))
        st.pyplot(fig_diag)
        st.info("Estos gráficos permiten evaluar si el modelo ha capturado toda la información (ruido blanco).")

    with tab3:
        st.subheader("Resumen del Modelo")
        # Mostrar el resumen estadístico del modelo (AIC, BIC, Coeficientes)
        st.text(str(model_results.summary()))

else:
    st.warning("Por favor, asegúrese de que los archivos .pkl están en la carpeta correcta.")

# --- FOOTER ---
st.divider()
st.caption("Desarrollado con Python, Streamlit y Statsmodels.")