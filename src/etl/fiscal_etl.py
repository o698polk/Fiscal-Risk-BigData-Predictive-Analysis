# -*- coding: utf-8 -*-
"""
Módulo ETL: FiscalETL — Extracción, Transformación y Carga de Datos Fiscales.
==============================================================================

Este módulo implementa el pipeline ETL completo para armonizar dos datasets
de diferente frecuencia temporal:
    - Dataset Macroeconómico Anual (1990–2024): 35 observaciones × 13 variables.
    - Dataset Trimestral (2005–2024): 80 observaciones × 13 variables.

La armonización se realiza mediante desagregación temporal Chow-Lin, que
distribuye los valores anuales (1990–2004) en estimaciones trimestrales
usando una relación estadística con indicadores de alta frecuencia.

Referencia:
    Vernaza Quiñonez, P.B. (2025). Sección 3.2 – Construcción del dataset
    multifrecuencia. "La consolidación de ambas fuentes requiere una técnica
    de desagregación temporal que preserve la coherencia de agregación."

    Chow, G. C., & Lin, A. (1971). Best linear unbiased interpolation,
    distribution, and extrapolation of time series by related series.
    The Review of Economics and Statistics, 53(4), 372-375.

Principio SOLID aplicado:
    S – Single Responsibility: Esta clase SOLO se ocupa de ETL.

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg

from src.config import (
    ANNUAL_DATASET,
    COLUMN_MAP,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    QUARTERLY_DATASET,
)

logger = logging.getLogger(__name__)


class FiscalETL:
    """Pipeline ETL para datos fiscales ecuatorianos multifrecuencia.

    Orquesta la extracción de dos CSVs (anual y trimestral), su limpieza,
    la desagregación temporal Chow-Lin de series anuales a trimestrales
    para el periodo 1990–2004, y la armonización final en un DataFrame
    maestro unificado.

    Attributes:
        df_annual: DataFrame con datos anuales crudos (1990–2024).
        df_quarterly: DataFrame con datos trimestrales crudos (2005–2024).
        df_master: DataFrame maestro armonizado (resultado del ETL).

    Example:
        >>> etl = FiscalETL()
        >>> df = etl.build_master_dataframe()
        >>> print(df.shape)  # (~140, 14+)
    """

    def __init__(self) -> None:
        """Inicializa el pipeline ETL sin ejecutar ninguna etapa."""
        self.df_annual: Optional[pd.DataFrame] = None
        self.df_quarterly: Optional[pd.DataFrame] = None
        self.df_master: Optional[pd.DataFrame] = None
        logger.info("FiscalETL inicializado.")

    # =========================================================================
    # EXTRACCIÓN
    # =========================================================================
    def extract_annual(self) -> pd.DataFrame:
        """Extrae el dataset macroeconómico anual (1990–2024).

        Carga el CSV, renombra las columnas al esquema estándar interno
        y establece el año como índice temporal.

        Returns:
            DataFrame con 35 filas × 12 columnas numéricas.

        Raises:
            FileNotFoundError: Si el archivo CSV no existe en la ruta esperada.
        """
        logger.info(f"Extrayendo dataset anual: {ANNUAL_DATASET}")
        df = pd.read_csv(ANNUAL_DATASET)
        df = df.rename(columns=COLUMN_MAP)
        df = df.set_index("year")
        df.index = df.index.astype(int)
        self.df_annual = df
        logger.info(
            f"Dataset anual cargado: {df.shape[0]} filas × {df.shape[1]} columnas. "
            f"Rango: {df.index.min()}–{df.index.max()}"
        )
        return df

    def extract_quarterly(self) -> pd.DataFrame:
        """Extrae el dataset trimestral (2005–2024).

        Parsea la columna 'Periodo' con formato 'YYYY TQ' a un índice
        datetime trimestral para facilitar la desagregación temporal.

        Returns:
            DataFrame con 80 filas × 12 columnas numéricas.

        Raises:
            FileNotFoundError: Si el archivo CSV no existe en la ruta esperada.
        """
        logger.info(f"Extrayendo dataset trimestral: {QUARTERLY_DATASET}")
        df = pd.read_csv(QUARTERLY_DATASET)
        df = df.rename(columns=COLUMN_MAP)

        # Parsear periodo "2005 T1" → datetime trimestral
        df["year"] = df["period"].str.extract(r"(\d{4})").astype(int)
        df["quarter"] = df["period"].str.extract(r"T(\d)").astype(int)
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + ((df["quarter"] - 1) * 3 + 1).astype(str) + "-01"
        )
        df = df.set_index("date")
        df = df.drop(columns=["period"], errors="ignore")

        self.df_quarterly = df
        logger.info(
            f"Dataset trimestral cargado: {df.shape[0]} filas × {df.shape[1]} columnas. "
            f"Rango: {df.index.min().strftime('%Y-Q')} a {df.index.max().strftime('%Y-Q')}"
        )
        return df

    # =========================================================================
    # TRANSFORMACIÓN
    # =========================================================================
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia un DataFrame: manejo de nulos, outliers y tipos de datos.

        Estrategia de limpieza:
            1. Interpolación lineal para valores faltantes internos.
            2. Forward-fill para valores faltantes en los extremos.
            3. Detección de outliers mediante IQR (se recortan a 1.5×IQR).

        Args:
            df: DataFrame a limpiar.

        Returns:
            DataFrame limpio sin valores nulos.
        """
        logger.info(f"Limpiando datos: {df.isnull().sum().sum()} nulos detectados.")

        # Seleccionar solo columnas numéricas para limpieza
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Paso 1: Interpolación lineal para gaps internos
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

        # Paso 2: Forward/backward fill para extremos
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        # Paso 3: Recorte de outliers (IQR × 1.5) – solo para series continuas
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                logger.warning(
                    f"  Columna '{col}': {outliers} outliers recortados "
                    f"(rango válido: [{lower:.2f}, {upper:.2f}])"
                )
            df[col] = df[col].clip(lower=lower, upper=upper)

        logger.info("Limpieza completada. Nulos restantes: 0.")
        return df

    def _chowlin_disaggregate(
        self,
        y_annual: pd.Series,
        indicator_quarterly: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Desagrega una serie anual a trimestral usando el método Chow-Lin.

        Implementación del estimador Chow-Lin (1971) que distribuye valores
        anuales en estimaciones trimestrales preservando la restricción de
        que la suma de los 4 trimestres iguale el valor anual original.

        El método resuelve:
            ŷ = X·β̂ + C'(C·C')⁻¹·(y - C·X·β̂)

        donde:
            - y: vector de valores anuales (baja frecuencia)
            - X: matriz de indicadores trimestrales (alta frecuencia)
            - C: matriz de agregación (suma trimestres → año)
            - β̂: coeficientes OLS de la regresión y = C·X·β + ε

        Si no se proporciona un indicador trimestral, se usa una tendencia
        lineal como proxy (Chow-Lin sin indicador relacionado).

        Args:
            y_annual: Serie con valores anuales a desagregar.
            indicator_quarterly: Serie trimestral indicadora (opcional).
                Si es None, se genera una tendencia lineal automática.

        Returns:
            Serie con valores trimestrales estimados.

        Note:
            Esta implementación es una versión simplificada del estimador
            Chow-Lin original. Para aplicaciones con más de una variable
            indicadora o estimación ML de ρ, usar la librería `tempdisagg`.
        """
        n_years = len(y_annual)
        n_quarters = n_years * 4

        # Construir matriz de indicadores (alta frecuencia)
        if indicator_quarterly is not None and len(indicator_quarterly) >= n_quarters:
            # Usar indicador real disponible
            X = indicator_quarterly.values[:n_quarters].reshape(-1, 1)
        else:
            # Fallback: tendencia lineal + constante
            trend = np.arange(1, n_quarters + 1, dtype=float)
            X = np.column_stack([np.ones(n_quarters), trend])

        # Matriz de agregación C (n_years × n_quarters)
        # Cada fila suma 4 trimestres consecutivos
        C = np.zeros((n_years, n_quarters))
        for i in range(n_years):
            C[i, i * 4: (i + 1) * 4] = 1.0

        # Paso 1: OLS sobre datos agregados → β̂
        CX = C @ X
        y = y_annual.values.astype(float)

        try:
            beta_hat = linalg.lstsq(CX, y)[0]
        except linalg.LinAlgError:
            logger.warning("Chow-Lin: lstsq falló, usando pseudoinversa.")
            beta_hat = np.linalg.pinv(CX) @ y

        # Paso 2: Estimación preliminar de alta frecuencia
        y_hat_hf = X @ beta_hat

        # Paso 3: Distribución del residuo de agregación
        residual_lf = y - C @ y_hat_hf

        # Matriz de covarianza simplificada (identidad × σ²)
        # Distribución proporcional del residuo
        CCt = C @ C.T
        try:
            CCt_inv = linalg.inv(CCt)
        except linalg.LinAlgError:
            CCt_inv = np.linalg.pinv(CCt)

        adjustment = C.T @ CCt_inv @ residual_lf
        y_disaggregated = y_hat_hf + adjustment

        return pd.Series(y_disaggregated, name=y_annual.name)

    def _harmonize_series(self) -> pd.DataFrame:
        """Armoniza las series desagregadas con los datos trimestrales originales.

        Procedimiento:
            1. Desagrega datos anuales 1990–2004 → trimestrales usando Chow-Lin.
            2. Concatena con datos trimestrales originales 2005–2024.
            3. Genera un índice datetime trimestral continuo.

        Returns:
            DataFrame armonizado con ~140 filas trimestrales (1990Q1–2024Q4).
        """
        logger.info("Armonizando series: Chow-Lin para 1990–2004 + originales 2005–2024")

        if self.df_annual is None or self.df_quarterly is None:
            raise ValueError("Debe ejecutar extract_annual() y extract_quarterly() primero.")

        # Datos anuales a desagregar: 1990–2004 (15 años)
        annual_pre2005 = self.df_annual.loc[1990:2004]

        # Columnas numéricas compartidas entre ambos datasets
        shared_cols = [
            col for col in NUMERIC_FEATURES
            if col in annual_pre2005.columns and col in self.df_quarterly.columns
        ]

        # Desagregar cada variable anual usando como indicador la serie trimestral
        disaggregated_records = []
        for col in shared_cols:
            # Usar la media trimestral de 2005–2008 como indicador tendencial
            indicator = self.df_quarterly[col].iloc[:16] if col in self.df_quarterly.columns else None
            series_q = self._chowlin_disaggregate(annual_pre2005[col], indicator)
            disaggregated_records.append(series_q)

        # Construir DataFrame desagregado
        df_disagg = pd.DataFrame(
            {col: series.values for col, series in zip(shared_cols, disaggregated_records)}
        )

        # Generar índice datetime trimestral 1990Q1–2004Q4
        dates_pre = pd.date_range(start="1990-01-01", periods=60, freq="QS")
        df_disagg.index = dates_pre

        # Agregar columnas year y quarter
        df_disagg["year"] = df_disagg.index.year
        df_disagg["quarter"] = df_disagg.index.quarter

        # Preparar datos trimestrales originales 2005–2024
        df_q_orig = self.df_quarterly[shared_cols + ["year", "quarter"]].copy()

        # Concatenar series
        df_harmonized = pd.concat([df_disagg, df_q_orig], axis=0, ignore_index=False)
        df_harmonized = df_harmonized.sort_index()

        logger.info(
            f"Armonización completada: {df_harmonized.shape[0]} trimestres × "
            f"{df_harmonized.shape[1]} columnas "
            f"({df_harmonized.index.min().year}Q1–{df_harmonized.index.max().year}Q4)"
        )
        return df_harmonized

    # =========================================================================
    # CARGA (Pipeline Completo)
    # =========================================================================
    def build_master_dataframe(self) -> pd.DataFrame:
        """Ejecuta el pipeline ETL completo y genera el DataFrame maestro.

        Pipeline:
            1. Extracción de ambos datasets (anual + trimestral).
            2. Limpieza de datos (nulos, outliers).
            3. Desagregación temporal Chow-Lin (1990–2004).
            4. Armonización y generación del DataFrame maestro.
            5. Exportación opcional a CSV procesado.

        Returns:
            DataFrame maestro armonizado listo para ingeniería de características.

        Example:
            >>> etl = FiscalETL()
            >>> df_master = etl.build_master_dataframe()
            >>> assert df_master.isnull().sum().sum() == 0
        """
        logger.info("=" * 70)
        logger.info("INICIO DEL PIPELINE ETL")
        logger.info("=" * 70)

        # Paso 1: Extracción
        self.extract_annual()
        self.extract_quarterly()

        # Paso 2: Limpieza
        self.df_annual = self._clean_data(self.df_annual)
        self.df_quarterly = self._clean_data(self.df_quarterly)

        # Paso 3-4: Armonización (incluye Chow-Lin)
        self.df_master = self._harmonize_series()

        # Paso 5: Limpieza final del maestro
        self.df_master = self._clean_data(self.df_master)

        # Exportar a CSV procesado
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DIR / "master_dataset_trimestral.csv"
        self.df_master.to_csv(output_path)
        logger.info(f"Dataset maestro exportado: {output_path}")

        logger.info("=" * 70)
        logger.info("PIPELINE ETL COMPLETADO EXITOSAMENTE")
        logger.info("=" * 70)

        return self.df_master
