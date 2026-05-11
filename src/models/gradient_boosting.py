# -*- coding: utf-8 -*-
"""
Modelos Gradient Boosting: GBRevenueModel y GBDebtModel.
==========================================================

Implementa dos modelos Gradient Boosting optimizados:
    - GBRevenueModel: Proyecta ingresos fiscales (Recaudación IVA).
    - GBDebtModel: Proyecta la trayectoria de Deuda/PIB (%).

Referencia:
    Vernaza Quiñonez, P.B. (2025). Sección 4.2 – "Los modelos Gradient
    Boosting capturan relaciones no lineales entre variables macro, logrando
    R² > 0.92 para ingresos y 0.89 para deuda/PIB en validación cruzada."

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    GB_LEARNING_RATE,
    GB_MAX_DEPTH,
    GB_N_ESTIMATORS,
    GB_SUBSAMPLE,
    RANDOM_SEED,
)
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class GBRevenueModel(BaseModel):
    """Gradient Boosting para proyección de ingresos fiscales.

    Modela la Recaudación IVA como proxy de ingresos fiscales totales,
    capturando la relación no lineal con el ciclo económico, precio del
    crudo y empleo.

    Attributes:
        model: GradientBoostingRegressor entrenado.
        feature_importances: Importancia relativa de cada feature.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[GradientBoostingRegressor] = None
        self.feature_importances: Optional[pd.Series] = None
        logger.info("GBRevenueModel inicializado.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBRevenueModel":
        """Entrena el modelo de ingresos con Gradient Boosting.

        Args:
            X: Features macroeconómicas.
            y: Recaudación IVA (USD millones).

        Returns:
            self: Instancia entrenada.
        """
        logger.info(f"Entrenando GBRevenueModel con {len(X)} muestras...")

        self.model = GradientBoostingRegressor(
            n_estimators=GB_N_ESTIMATORS,
            learning_rate=GB_LEARNING_RATE,
            max_depth=GB_MAX_DEPTH,
            subsample=GB_SUBSAMPLE,
            random_state=RANDOM_SEED,
        )
        self.model.fit(X.values, y.values)

        self.feature_importances = pd.Series(
            self.model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)

        y_pred = self.model.predict(X.values)
        self.training_metrics = {
            "mae": float(mean_absolute_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": float(r2_score(y, y_pred)),
        }

        self.is_fitted = True
        logger.info(
            f"GBRevenueModel entrenado. R²={self.training_metrics['r2']:.4f}, "
            f"MAE={self.training_metrics['mae']:.2f}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X.values)

    def get_metrics(self) -> Dict[str, float]:
        return self.training_metrics

    def get_model_name(self) -> str:
        return "GradientBoosting_Ingresos"


class GBDebtModel(BaseModel):
    """Gradient Boosting para proyección de Deuda/PIB.

    Modela la trayectoria del ratio Deuda/PIB (%) considerando el déficit
    primario, crecimiento económico e ingresos petroleros como drivers.

    Referencia: Vernaza Quiñonez – "La dinámica de deuda sigue una ecuación
    de acumulación donde d(t+1) = d(t)·(1+r)/(1+g) + déficit, siendo r la
    tasa de interés implícita y g el crecimiento del PIB."
    """

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[GradientBoostingRegressor] = None
        self.feature_importances: Optional[pd.Series] = None
        logger.info("GBDebtModel inicializado.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBDebtModel":
        """Entrena el modelo de deuda con Gradient Boosting.

        Args:
            X: Features macroeconómicas.
            y: Deuda/PIB (%).

        Returns:
            self: Instancia entrenada.
        """
        logger.info(f"Entrenando GBDebtModel con {len(X)} muestras...")

        self.model = GradientBoostingRegressor(
            n_estimators=GB_N_ESTIMATORS,
            learning_rate=GB_LEARNING_RATE,
            max_depth=GB_MAX_DEPTH,
            subsample=GB_SUBSAMPLE,
            random_state=RANDOM_SEED,
        )
        self.model.fit(X.values, y.values)

        self.feature_importances = pd.Series(
            self.model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)

        y_pred = self.model.predict(X.values)
        self.training_metrics = {
            "mae": float(mean_absolute_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": float(r2_score(y, y_pred)),
        }

        self.is_fitted = True
        logger.info(
            f"GBDebtModel entrenado. R²={self.training_metrics['r2']:.4f}, "
            f"MAE={self.training_metrics['mae']:.2f}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X.values)

    def get_metrics(self) -> Dict[str, float]:
        return self.training_metrics

    def get_model_name(self) -> str:
        return "GradientBoosting_Deuda"
