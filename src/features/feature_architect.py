# -*- coding: utf-8 -*-
"""
Módulo de Ingeniería de Características: FeatureArchitect.
============================================================

Implementa la construcción del Índice de Riesgo Fiscal Compuesto (IRFC) y
el Análisis de Componentes Principales (PCA) para reducir la dimensionalidad
del espacio de features a los ejes estructurales de la economía ecuatoriana.

Referencia:
    Vernaza Quiñonez, P.B. (2025). Sección 4.5 – PCA y Sección 4.3 – IRFC.
    "El IRFC sintetiza cinco indicadores clave en un score normalizado [0-100]
    donde valores superiores a 75 indican riesgo fiscal crítico."
    "PCA revela dos ejes estructurales: el eje Petróleo-PIB (PC1) y el eje
    Deuda-Déficit (PC2), que juntos explican el 73.4% de la varianza."

Principio SOLID aplicado:
    S – Single Responsibility: Esta clase SOLO transforma features.
    D – Dependency Inversion: Depende de config.py (abstracciones), no de
        valores hardcoded.

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import (
    IRFC_WEIGHTS,
    NUMERIC_FEATURES,
    PCA_N_COMPONENTS,
    PCA_VARIANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class FeatureArchitect:
    """Constructor de features avanzadas para análisis de riesgo fiscal.

    Transforma el DataFrame maestro aplicando:
        1. Índice de Riesgo Fiscal Compuesto (IRFC): score ponderado [0-100].
        2. PCA: reducción de dimensionalidad a 4 componentes principales.

    El IRFC agrega 5 indicadores con ponderaciones basadas en su impacto
    relativo sobre la sostenibilidad fiscal, según la investigación de
    Vernaza Quiñonez (2025).

    Attributes:
        pca_model: Instancia de PCA ajustada tras llamar a apply_pca().
        scaler: StandardScaler usado para PCA.
        irfc_scaler: MinMaxScaler usado para normalizar componentes del IRFC.
        pca_loadings: Matriz de cargas factoriales (componentes × features).

    Example:
        >>> architect = FeatureArchitect()
        >>> df_augmented = architect.transform(df_master)
        >>> print("IRFC" in df_augmented.columns)
        True
    """

    def __init__(self) -> None:
        """Inicializa los transformadores internos."""
        self.pca_model: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.irfc_scaler: Optional[MinMaxScaler] = None
        self.pca_loadings: Optional[pd.DataFrame] = None
        logger.info("FeatureArchitect inicializado.")

    def compute_irfc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula el Índice de Riesgo Fiscal Compuesto (IRFC).

        El IRFC es un indicador sintético normalizado [0–100] que pondera
        cinco variables macroeconómicas clave:

            IRFC = 0.30 × norm(Deuda/PIB)
                 + 0.25 × norm(1 / Ratio_RP)
                 + 0.20 × norm(Déficit)
                 + 0.15 × norm(1 - Empleo/100)
                 + 0.10 × norm(1 - Ingresos_PGE/100)

        donde norm() aplica escalado MinMax [0,1] y las inversiones garantizan
        que mayor riesgo fiscal → mayor valor del componente.

        Referencia: Vernaza Quiñonez, Tabla 8 – "Los pesos del IRFC reflejan
        la importancia relativa de cada indicador en la literatura sobre
        sostenibilidad fiscal de economías petroleras."

        Args:
            df: DataFrame con las columnas requeridas del IRFC.

        Returns:
            DataFrame original con columna 'irfc' añadida (float, 0–100).

        Raises:
            KeyError: Si faltan columnas requeridas para el cálculo.
        """
        logger.info("Calculando Índice de Riesgo Fiscal Compuesto (IRFC)...")

        df_work = df.copy()

        # Verificar que existen las columnas necesarias
        required_cols = list(IRFC_WEIGHTS.keys())
        missing = [col for col in required_cols if col not in df_work.columns]
        if missing:
            raise KeyError(
                f"Columnas faltantes para IRFC: {missing}. "
                f"Disponibles: {list(df_work.columns)}"
            )

        # Construir componentes orientados al riesgo (mayor valor = mayor riesgo)
        risk_components = pd.DataFrame(index=df_work.index)

        # Deuda/PIB: mayor deuda → mayor riesgo (directo)
        risk_components["deuda_pib"] = df_work["deuda_pib"]

        # Ratio R/P invertido: menor ratio → mayor riesgo
        # Usar inverso para que valores bajos (pocas reservas) den riesgo alto
        risk_components["ratio_rp"] = 1.0 / df_work["ratio_rp"].clip(lower=0.1)

        # Déficit: mayor déficit → mayor riesgo (directo)
        risk_components["deficit_primario"] = df_work["deficit_primario"]

        # Empleo invertido: menor empleo → mayor riesgo
        risk_components["empleo_adecuado"] = 100.0 - df_work["empleo_adecuado"]

        # Ingresos PGE invertidos: menores ingresos → mayor riesgo
        risk_components["ingresos_pge"] = 100.0 - df_work["ingresos_pge"].clip(upper=100)

        # Normalización MinMax [0, 1] de cada componente
        self.irfc_scaler = MinMaxScaler()
        risk_normalized = pd.DataFrame(
            self.irfc_scaler.fit_transform(risk_components),
            columns=risk_components.columns,
            index=risk_components.index,
        )

        # Cálculo ponderado del IRFC
        irfc_score = sum(
            IRFC_WEIGHTS[col] * risk_normalized[col] for col in IRFC_WEIGHTS
        )

        # Escalar a [0, 100]
        df_work["irfc"] = (irfc_score * 100).round(2)

        logger.info(
            f"IRFC calculado: media={df_work['irfc'].mean():.2f}, "
            f"min={df_work['irfc'].min():.2f}, max={df_work['irfc'].max():.2f}"
        )

        return df_work

    def apply_pca(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Aplica PCA para reducir dimensionalidad a 4 componentes principales.

        Estandariza las features numéricas y aplica PCA, reteniendo las
        componentes necesarias para explicar >80% de la varianza total.

        Referencia: Vernaza Quiñonez, Sección 4.5 – "PCA revela que las
        12 variables originales se estructuran en torno a dos ejes
        fundamentales: PC1 (Petróleo-PIB, 45.2%) y PC2 (Deuda-Déficit, 28.2%)."

        Args:
            df: DataFrame con features numéricas.

        Returns:
            Tuple de:
                - DataFrame con columnas PC1, PC2, PC3, PC4 añadidas.
                - Dict con varianza explicada por componente y acumulada.

        Note:
            Las cargas factoriales se almacenan en self.pca_loadings para
            posterior interpretación de los ejes.
        """
        logger.info(f"Aplicando PCA (n_components={PCA_N_COMPONENTS})...")

        # Seleccionar features numéricas disponibles
        available_features = [f for f in NUMERIC_FEATURES if f in df.columns]
        logger.info(f"Features para PCA: {len(available_features)} variables")

        # Estandarización (media=0, std=1) – requisito para PCA
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(df[available_features].values)

        # Ajustar PCA
        n_components = min(PCA_N_COMPONENTS, len(available_features))
        self.pca_model = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca_model.fit_transform(X_scaled)

        # Agregar componentes al DataFrame
        df_result = df.copy()
        for i in range(n_components):
            df_result[f"PC{i + 1}"] = X_pca[:, i]

        # Varianza explicada
        variance_info = {}
        cumulative = 0.0
        for i, var in enumerate(self.pca_model.explained_variance_ratio_):
            cumulative += var
            variance_info[f"PC{i + 1}"] = round(var * 100, 2)
            logger.info(
                f"  PC{i + 1}: {var * 100:.2f}% varianza "
                f"(acumulada: {cumulative * 100:.2f}%)"
            )
        variance_info["cumulative"] = round(cumulative * 100, 2)

        # Almacenar cargas factoriales
        self.pca_loadings = pd.DataFrame(
            self.pca_model.components_,
            columns=available_features,
            index=[f"PC{i + 1}" for i in range(n_components)],
        )

        # Verificar umbral de varianza
        if cumulative >= PCA_VARIANCE_THRESHOLD:
            logger.info(
                f"✓ PCA cumple umbral: {cumulative * 100:.2f}% ≥ "
                f"{PCA_VARIANCE_THRESHOLD * 100:.0f}%"
            )
        else:
            logger.warning(
                f"⚠ PCA NO cumple umbral: {cumulative * 100:.2f}% < "
                f"{PCA_VARIANCE_THRESHOLD * 100:.0f}%"
            )

        return df_result, variance_info

    def get_pca_loadings(self) -> pd.DataFrame:
        """Retorna la matriz de cargas factoriales del PCA.

        Las cargas indican la correlación entre cada variable original y
        cada componente principal. Valores altos (abs > 0.4) indican
        contribuciones significativas.

        Returns:
            DataFrame con shape (n_components, n_features).

        Raises:
            ValueError: Si PCA no ha sido ajustado aún.
        """
        if self.pca_loadings is None:
            raise ValueError(
                "PCA no ha sido ajustado. Ejecute apply_pca() primero."
            )
        return self.pca_loadings

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Pipeline completo de ingeniería de características.

        Ejecuta secuencialmente:
            1. Cálculo del IRFC (Índice de Riesgo Fiscal Compuesto).
            2. PCA (Análisis de Componentes Principales).

        Args:
            df: DataFrame maestro del ETL.

        Returns:
            Tuple de:
                - DataFrame aumentado con IRFC + PC1..PC4.
                - Dict con metadatos (varianza PCA, estadísticas IRFC).
        """
        logger.info("=" * 70)
        logger.info("INICIO DE INGENIERÍA DE CARACTERÍSTICAS")
        logger.info("=" * 70)

        # Paso 1: IRFC
        df_featured = self.compute_irfc(df)

        # Paso 2: PCA
        df_featured, pca_info = self.apply_pca(df_featured)

        metadata = {
            "pca_variance": pca_info,
            "irfc_stats": {
                "mean": float(df_featured["irfc"].mean()),
                "std": float(df_featured["irfc"].std()),
                "min": float(df_featured["irfc"].min()),
                "max": float(df_featured["irfc"].max()),
            },
            "n_features_original": len(NUMERIC_FEATURES),
            "n_features_final": df_featured.shape[1],
        }

        logger.info("=" * 70)
        logger.info("INGENIERÍA DE CARACTERÍSTICAS COMPLETADA")
        logger.info("=" * 70)

        return df_featured, metadata
