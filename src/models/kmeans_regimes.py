# -*- coding: utf-8 -*-
"""
K-Means Clustering: EconomicRegimeClusterer.
===============================================

Identifica 7 regímenes económicos históricos de Ecuador (1990–2024)
usando K-Means sobre las componentes principales del PCA.

Los 7 regímenes identificados por la investigación:
    1. Pre-dolarización Crisis (1990–1999)
    2. Dolarización Temprana (2000–2003)
    3. Boom Petrolero I (2004–2008)
    4. Crisis Financiera Global (2009)
    5. Boom Petrolero II (2010–2014)
    6. Recesión Petrolera (2015–2017)
    7. Pandemia y Post-pandemia (2018–2024)

Referencia:
    Vernaza Quiñonez, P.B. (2025). Sección 4.4 – "K-Means revela 7
    clústeres temporales coherentes con los regímenes político-económicos
    documentados, validando la capacidad del modelo para detectar
    transiciones estructurales."

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import KMEANS_N_CLUSTERS, RANDOM_SEED

logger = logging.getLogger(__name__)

# Etiquetas interpretativas de los 7 regímenes (basadas en la investigación)
REGIME_LABELS: Dict[int, str] = {
    0: "Pre-dolarización/Crisis",
    1: "Dolarización Temprana",
    2: "Boom Petrolero I",
    3: "Crisis Global 2009",
    4: "Boom Petrolero II",
    5: "Recesión Petrolera",
    6: "Pandemia/Post-pandemia",
}


class EconomicRegimeClusterer:
    """Identificador de regímenes económicos históricos mediante K-Means.

    Aplica K-Means clustering sobre las componentes PCA para agrupar
    periodos históricos con características macroeconómicas similares.

    A diferencia de los modelos predictivos (BaseModel), este es un
    modelo descriptivo/no supervisado y NO implementa predict() para
    forecasting. Cumple con Interface Segregation (ISP).

    Attributes:
        model: KMeans entrenado.
        scaler: StandardScaler para normalización de inputs.
        labels: Etiquetas de clúster asignadas a cada observación.
        centroids: Centroides de cada clúster.
        silhouette: Coeficiente de silueta del clustering.
        regime_summary: Resumen estadístico por régimen.

    Example:
        >>> clusterer = EconomicRegimeClusterer()
        >>> labels = clusterer.fit_predict(df_pca[["PC1", "PC2", "PC3", "PC4"]])
        >>> print(clusterer.silhouette)
    """

    def __init__(self, n_clusters: int = KMEANS_N_CLUSTERS) -> None:
        """Inicializa el clusterer con el número de regímenes.

        Args:
            n_clusters: Número de clústeres (default: 7 regímenes).
        """
        self.n_clusters = n_clusters
        self.model: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.labels: Optional[np.ndarray] = None
        self.centroids: Optional[np.ndarray] = None
        self.silhouette: Optional[float] = None
        self.regime_summary: Optional[pd.DataFrame] = None
        logger.info(f"EconomicRegimeClusterer inicializado (k={n_clusters}).")

    def fit_predict(
        self,
        X: pd.DataFrame,
        original_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Entrena K-Means y asigna etiquetas de régimen.

        Args:
            X: Features para clustering (preferiblemente componentes PCA).
            original_df: DataFrame original con variables para el resumen
                estadístico por régimen (opcional).

        Returns:
            Array de etiquetas de clúster (0 a n_clusters-1).
        """
        logger.info(f"Ejecutando K-Means con k={self.n_clusters} sobre {X.shape}...")

        # Normalización
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X.values)

        # K-Means
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=RANDOM_SEED,
            n_init=20,
            max_iter=500,
        )
        self.labels = self.model.fit_predict(X_scaled)
        self.centroids = self.model.cluster_centers_

        # Silhouette score (calidad del clustering)
        if len(set(self.labels)) > 1:
            self.silhouette = float(silhouette_score(X_scaled, self.labels))
        else:
            self.silhouette = 0.0
        logger.info(f"Silhouette Score: {self.silhouette:.4f}")

        # Distribución de clústeres
        unique, counts = np.unique(self.labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            label = REGIME_LABELS.get(cluster_id, f"Clúster {cluster_id}")
            logger.info(f"  Régimen {cluster_id} ({label}): {count} observaciones")

        # Resumen estadístico por régimen
        if original_df is not None:
            self._compute_regime_summary(original_df)

        return self.labels

    def _compute_regime_summary(self, df: pd.DataFrame) -> None:
        """Calcula estadísticas descriptivas por régimen económico.

        Args:
            df: DataFrame con las variables macroeconómicas originales.
        """
        df_with_regime = df.copy()
        df_with_regime["regime"] = self.labels[:len(df)]
        df_with_regime["regime_label"] = df_with_regime["regime"].map(REGIME_LABELS)

        numeric_cols = df_with_regime.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != "regime"]

        self.regime_summary = df_with_regime.groupby("regime_label")[numeric_cols].agg(
            ["mean", "std"]
        ).round(2)

        logger.info("Resumen estadístico por régimen calculado.")

    def get_labels_with_names(self) -> pd.Series:
        """Retorna las etiquetas de clúster con nombres interpretativos.

        Returns:
            Series con el nombre del régimen para cada observación.
        """
        if self.labels is None:
            raise ValueError("Debe ejecutar fit_predict() primero.")
        return pd.Series(self.labels).map(REGIME_LABELS)

    def get_metrics(self) -> Dict[str, float]:
        """Retorna métricas del clustering."""
        return {
            "silhouette_score": self.silhouette or 0.0,
            "n_clusters": float(self.n_clusters),
            "inertia": float(self.model.inertia_) if self.model else 0.0,
        }

    def get_model_name(self) -> str:
        return "KMeans_Regimenes"
