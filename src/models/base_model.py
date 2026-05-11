# -*- coding: utf-8 -*-
"""
Modelo Base Abstracto: BaseModel.
==================================

Define la interfaz que TODOS los modelos del ecosistema deben implementar,
garantizando intercambiabilidad (Principio de Sustitución de Liskov) y
extensibilidad (Principio Open/Closed).

Principio SOLID aplicado:
    L – Liskov Substitution: Cualquier subclase puede sustituir a BaseModel.
    I – Interface Segregation: Solo métodos esenciales en la interfaz.

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Interfaz abstracta para todos los modelos del ecosistema fiscal.

    Todo modelo predictivo del ecosistema DEBE heredar de esta clase
    e implementar los 4 métodos abstractos definidos aquí.

    Esta abstracción permite al ModelEcosystem orquestar el entrenamiento
    y predicción de cualquier modelo de forma uniforme, sin conocer
    los detalles de implementación de cada uno.

    Attributes:
        is_fitted: Indica si el modelo ha sido entrenado.
        training_metrics: Métricas calculadas durante el entrenamiento.
    """

    def __init__(self) -> None:
        """Inicializa el estado base del modelo."""
        self.is_fitted: bool = False
        self.training_metrics: Dict[str, float] = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Entrena el modelo con los datos proporcionados.

        Args:
            X: Features de entrenamiento.
            y: Variable objetivo.

        Returns:
            self: La instancia del modelo entrenado (para encadenamiento).
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Genera predicciones para nuevos datos.

        Args:
            X: Features para predicción.

        Returns:
            Array de predicciones.

        Raises:
            RuntimeError: Si el modelo no ha sido entrenado.
        """
        ...

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Retorna las métricas de rendimiento del modelo.

        Returns:
            Dict con métricas (e.g., MAE, R², RMSE).
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Retorna el nombre identificador del modelo.

        Returns:
            String con el nombre del modelo.
        """
        ...

    def _check_fitted(self) -> None:
        """Verifica que el modelo haya sido entrenado antes de predecir.

        Raises:
            RuntimeError: Si is_fitted es False.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"El modelo '{self.get_model_name()}' no ha sido entrenado. "
                f"Ejecute fit() primero."
            )
