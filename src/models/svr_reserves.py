# -*- coding: utf-8 -*-
"""
Modelo SVR: SVRReservesModel.
================================

Support Vector Regression con kernel RBF para analizar la trayectoria
de agotamiento de reservas petroleras probadas (Mb).

Referencia:
    Vernaza Quiñonez, P.B. (2025). Sección 4.3 – "SVR con kernel RBF
    captura la tendencia decreciente no lineal de las reservas probadas,
    con un R² de 0.94 y MAE de 28.5 Mb en validación."

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.config import RANDOM_SEED, SVR_C_RANGE, SVR_GAMMA_RANGE, SVR_KERNEL
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class SVRReservesModel(BaseModel):
    """SVR con kernel RBF para proyección de reservas petroleras probadas.

    Utiliza Support Vector Regression para modelar la tendencia de
    agotamiento de reservas, capturando no linealidades que los
    modelos lineales no pueden representar.

    La selección de hiperparámetros (C, gamma) se realiza mediante
    GridSearchCV con validación temporal (3-fold).

    Attributes:
        pipeline: Pipeline con StandardScaler + SVR.
        best_params: Mejores hiperparámetros encontrados por GridSearch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pipeline: Optional[Pipeline] = None
        self.best_params: Optional[Dict] = None
        logger.info("SVRReservesModel inicializado.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SVRReservesModel":
        """Entrena SVR con optimización de hiperparámetros.

        Pipeline:
            1. StandardScaler (normalización Z-score, requerido por SVR).
            2. SVR con kernel RBF.
            3. GridSearchCV para selección de C y gamma.

        Args:
            X: Features macroeconómicas.
            y: Reservas probadas (Mb).

        Returns:
            self: Instancia entrenada.
        """
        logger.info(f"Entrenando SVRReservesModel con {len(X)} muestras...")

        # Pipeline: escalado + SVR
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel=SVR_KERNEL)),
        ])

        # Grid search para C y gamma
        param_grid = {
            "svr__C": SVR_C_RANGE,
            "svr__gamma": SVR_GAMMA_RANGE,
        }

        grid = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=min(3, len(X) // 2),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        grid.fit(X.values, y.values)

        self.pipeline = grid.best_estimator_
        self.best_params = grid.best_params_
        logger.info(f"SVR mejores parámetros: {self.best_params}")

        y_pred = self.pipeline.predict(X.values)
        self.training_metrics = {
            "mae": float(mean_absolute_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": float(r2_score(y, y_pred)),
        }

        self.is_fitted = True
        logger.info(
            f"SVRReservesModel entrenado. R²={self.training_metrics['r2']:.4f}, "
            f"MAE={self.training_metrics['mae']:.2f} Mb"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict(X.values)

    def get_metrics(self) -> Dict[str, float]:
        return self.training_metrics

    def get_model_name(self) -> str:
        return "SVR_Reservas"
