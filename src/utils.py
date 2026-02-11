import pickle
import pandas as pd
from pathlib import Path


def predecir(modelo, pasos: int):
    """
    Genera predicciones futuras.
    Si el modelo requiere variables exógenas,
    se generan automáticamente con valores 0.
    """
    import pandas as pd
    import numpy as np

    # Detectar si el modelo tiene exógenas
    if hasattr(modelo, "model") and hasattr(modelo.model, "exog_names"):
        exog_names = modelo.model.exog_names

        if exog_names is not None:
            # Crear DataFrame futuro con ceros
            exog_futuro = pd.DataFrame(
                np.zeros((pasos, len(exog_names))),
                columns=exog_names
            )

            pred = modelo.forecast(steps=pasos, exog=exog_futuro)
            return pd.Series(pred)

    # Si no tiene exógenas
    if hasattr(modelo, "forecast"):
        pred = modelo.forecast(steps=pasos)
    elif hasattr(modelo, "predict"):
        pred = modelo.predict(n_periods=pasos)
    else:
        raise AttributeError("Modelo no compatible")

    return pd.Series(pred)