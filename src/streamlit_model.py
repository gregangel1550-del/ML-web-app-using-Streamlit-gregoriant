import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="📈 Population Forecast App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for professional look
# -----------------------------
st.markdown(
    """
    <style>
    .main {background-color: #f7f9fc;}
    h1, h2, h3 {color: #1f2c56;}
    .metric-box {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
st.title("📈 Population Forecast by City")
st.markdown("Forecast future population using trained **time series models**.")
st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Configuration")

CITY_MODELS = {
    "Madrid": "models/modelo_madrid.pkl",
    "Barcelona": "models/modelo_barcelona.pkl",
    "Valencia": "models/modelo_valencia.pkl",
    "Alicante": "models/modelo_alicante.pkl",
    "Castellón": "models/modelo_castellon.pkl",
}

city = st.sidebar.selectbox("🏙️ Select City", list(CITY_MODELS.keys()))

n_periods = st.sidebar.slider(
    "📅 Forecast horizon (years)",
    min_value=1,
    max_value=30,
    value=10,
    step=1
)

show_confidence = st.sidebar.checkbox("📉 Show confidence interval", value=True)

run_button = st.sidebar.button("🚀 Run Forecast")

st.sidebar.divider()
st.sidebar.info("Models trained with historical population data")

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# -----------------------------
# Main logic
# -----------------------------
if run_button:
    model_path = CITY_MODELS[city]

    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
    else:
        with st.spinner("Loading model and generating forecast..."):
            model = load_model(model_path)

            # Forecast (compatible with ARIMA / SARIMA)
            forecast = model.get_forecast(steps=n_periods)
            mean_forecast = forecast.predicted_mean
            conf_int = forecast.conf_int()

            years = np.arange(
                datetime.now().year + 1,
                datetime.now().year + n_periods + 1
            )

            df_forecast = pd.DataFrame({
                "Year": years,
                "Forecast": mean_forecast.values,
                "Lower": conf_int.iloc[:, 0].values,
                "Upper": conf_int.iloc[:, 1].values,
            })

        # -----------------------------
        # KPIs
        # -----------------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"<div class='metric-box'><h3>City</h3><h2>{city}</h2></div>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"<div class='metric-box'><h3>Years Forecasted</h3><h2>{n_periods}</h2></div>",
                unsafe_allow_html=True
            )

        with col3:
            growth = mean_forecast.values[-1] - mean_forecast.values[0]
            st.markdown(
                f"<div class='metric-box'><h3>Total Growth</h3><h2>{growth:,.0f}</h2></div>",
                unsafe_allow_html=True
            )

        st.divider()

        # -----------------------------
        # Plot
        # -----------------------------
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_forecast["Year"], df_forecast["Forecast"], label="Forecast", linewidth=2)

        if show_confidence:
            ax.fill_between(
                df_forecast["Year"],
                df_forecast["Lower"],
                df_forecast["Upper"],
                alpha=0.3,
                label="Confidence Interval"
            )

        ax.set_title(f"Population Forecast – {city}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Population")
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig)

        st.divider()

        # -----------------------------
        # Data table
        # -----------------------------
        st.subheader("📋 Forecast Data")
        st.dataframe(df_forecast, use_container_width=True)

        # -----------------------------
        # Download
        # -----------------------------
        csv = df_forecast.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"forecast_{city.lower()}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Configure the options on the sidebar and click **Run Forecast**")
