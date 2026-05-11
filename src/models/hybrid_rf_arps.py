# -*- coding: utf-8 -*-
"""
Modelo Híbrido RF + Arps: HybridRFArpsModel.
===============================================

Combina un Random Forest (50%) con la ecuación de declive exponencial
de Arps (50%) para proyectar la producción petrolera diaria (Kb/d).

La ecuación de Arps modela el declive natural de los yacimientos:
    q(t) = q_i · exp(-λt)

donde:
    - q_i: Producción inicial (Kb/d)
    - λ (lambda): Tasa de declive exponencial (1/año)
    - t: Tiempo en años desde el punto inicial

Referencia:
    Vernaza Quiñonez, P.B. (2025). Sección 4.1 – "El ensemble híbrido
    supera al Random Forest puro al incorporar la restricción física del
    declive de yacimientos, reduciendo el MAE de 18.3 a 12.7 Kb/d."

    Arps, J.J. (1945). Analysis of Decline Curves. Transactions of the
    AIME, 160(01), 228-247.

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import ARPS_HYBRID_WEIGHT, RANDOM_SEED, RF_MAX_DEPTH, RF_N_ESTIMATORS
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class HybridRFArpsModel(BaseModel):
    """Modelo híbrido que combina Random Forest con declive exponencial de Arps.

    La combinación captura tanto los patrones no lineales aprendidos por ML
    (factores geopolíticos, inversión, precio) como la restricción física
    del agotamiento natural de yacimientos petroleros.

    Attributes:
        rf_model: RandomForestRegressor entrenado.
        arps_params: Tuple (q_i, lambda) ajustados por curve_fit.
        hybrid_weight: Peso de cada componente (default: 0.50).
        feature_importances: Importancia de features del Random Forest.

    Example:
        >>> model = HybridRFArpsModel()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, hybrid_weight: float = ARPS_HYBRID_WEIGHT) -> None:
        """Inicializa el modelo híbrido RF + Arps.

        Args:
            hybrid_weight: Peso del componente ML (1 - weight para Arps).
                Default: 0.50 (50/50).
        """
        super().__init__()
        self.rf_model: Optional[RandomForestRegressor] = None
        self.arps_params: Optional[Tuple[float, float]] = None
        self.hybrid_weight: float = hybrid_weight
        self.feature_importances: Optional[pd.Series] = None
        self._time_offset: float = 0.0
        logger.info(
            f"HybridRFArpsModel inicializado (peso ML: {hybrid_weight:.0%}, "
            f"peso Arps: {1 - hybrid_weight:.0%})"
        )

    @staticmethod
    def _arps_exponential(t: np.ndarray, q_i: float, lam: float) -> np.ndarray:
        """Ecuación de declive exponencial de Arps.

        Modela la producción como una exponencial decreciente:
            q(t) = q_i · exp(-λ·t)

        Args:
            t: Vector de tiempo (años).
            q_i: Producción inicial (Kb/d).
            lam: Tasa de declive (1/año). Valores típicos: 0.02–0.06.

        Returns:
            Array con producción estimada en cada punto temporal.
        """
        return q_i * np.exp(-lam * t)

    def _fit_arps(self, time_index: np.ndarray, production: np.ndarray) -> None:
        """Ajusta los parámetros de Arps por mínimos cuadrados no lineales.

        Usa scipy.optimize.curve_fit con bounds para garantizar parámetros
        físicamente significativos (q_i > 0, λ > 0).

        Args:
            time_index: Vector temporal normalizado.
            production: Producción observada (Kb/d).
        """
        try:
            # Bounds: q_i ∈ [100, 1000], λ ∈ [0.001, 0.2]
            popt, _ = curve_fit(
                self._arps_exponential,
                time_index,
                production,
                p0=[production[0], 0.03],
                bounds=([100, 0.001], [1000, 0.2]),
                maxfev=10000,
            )
            self.arps_params = (popt[0], popt[1])
            logger.info(
                f"Arps ajustado: q_i={popt[0]:.1f} Kb/d, λ={popt[1]:.4f}/año"
            )
        except (RuntimeError, ValueError) as e:
            # Fallback: usar valores razonables si el ajuste falla
            logger.warning(f"Ajuste Arps falló ({e}). Usando parámetros por defecto.")
            self.arps_params = (float(production[0]), 0.035)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HybridRFArpsModel":
        """Entrena ambos componentes del modelo híbrido.

        1. Entrena Random Forest con todas las features macroeconómicas.
        2. Ajusta Arps con la serie temporal de producción.

        Args:
            X: Features de entrenamiento (n_samples, n_features).
            y: Producción diaria objetivo (Kb/d).

        Returns:
            self: Instancia entrenada.
        """
        logger.info(f"Entrenando HybridRFArps con {len(X)} muestras...")

        # --- Componente 1: Random Forest ---
        self.rf_model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.rf_model.fit(X.values, y.values)

        # Almacenar importancia de features
        self.feature_importances = pd.Series(
            self.rf_model.feature_importances_,
            index=X.columns,
        ).sort_values(ascending=False)
        logger.info(f"RF entrenado. Top-3 features: {list(self.feature_importances.head(3).index)}")

        # --- Componente 2: Arps ---
        time_index = np.arange(len(y), dtype=float)
        self._time_offset = 0.0
        self._fit_arps(time_index, y.values)

        # --- Métricas de entrenamiento ---
        rf_pred = self.rf_model.predict(X.values)
        arps_pred = self._arps_exponential(time_index, *self.arps_params)
        hybrid_pred = self.hybrid_weight * rf_pred + (1 - self.hybrid_weight) * arps_pred

        self.training_metrics = {
            "mae": float(mean_absolute_error(y, hybrid_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, hybrid_pred))),
            "r2": float(r2_score(y, hybrid_pred)),
            "mae_rf_only": float(mean_absolute_error(y, rf_pred)),
            "mae_arps_only": float(mean_absolute_error(y, arps_pred)),
        }

        self.is_fitted = True
        logger.info(
            f"HybridRFArps entrenado. MAE={self.training_metrics['mae']:.2f}, "
            f"R²={self.training_metrics['r2']:.4f}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Genera predicciones híbridas (RF + Arps).

        Args:
            X: Features para predicción.

        Returns:
            Array con producción estimada (Kb/d).
        """
        self._check_fitted()

        # Predicción RF
        rf_pred = self.rf_model.predict(X.values)

        # Predicción Arps (extrapolada desde el final del entrenamiento)
        n_train = self.rf_model.n_features_in_  # proxy
        time_index = np.arange(len(X), dtype=float) + len(X)
        arps_pred = self._arps_exponential(time_index, *self.arps_params)

        # Combinación híbrida
        return self.hybrid_weight * rf_pred + (1 - self.hybrid_weight) * arps_pred

    def get_metrics(self) -> Dict[str, float]:
        """Retorna métricas de entrenamiento del modelo híbrido."""
        return self.training_metrics

    def get_model_name(self) -> str:
        """Retorna el nombre del modelo."""
        return "HybridRFArps_Produccion"
