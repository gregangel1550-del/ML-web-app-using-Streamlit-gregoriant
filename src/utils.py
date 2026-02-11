import pickle
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def cargar_modelo(ciudad: str):
    modelo_path = BASE_DIR / f"modelo_{ciudad}.pkl"

    if not modelo_path.exists():
        raise FileNotFoundError(f"No existe el modelo para {ciudad}")

    with open(modelo_path, "rb") as f:
        modelo = pickle.load(f)

    return modelo


def predecir(modelo, pasos: int):
    if hasattr(modelo, "forecast"):
        pred = modelo.forecast(steps=pasos)
    elif hasattr(modelo, "predict"):
        pred = modelo.predict(n_periods=pasos)
    else:
        raise AttributeError("Modelo no compatible")

    return pd.Series(pred)