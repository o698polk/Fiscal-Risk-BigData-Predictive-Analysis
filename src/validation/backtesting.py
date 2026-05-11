# -*- coding: utf-8 -*-
"""
Validación Retrospectiva: BacktestingValidator.
=================================================

Implementa backtesting temporal riguroso para validar la capacidad
predictiva de cada modelo del ecosistema, usando split temporal
Train (1990–2018) vs Test (2019–2024).

Referencia:
    Vernaza Quiñonez, P.B. (2025). Sección 4.8 – "El backtesting con datos
    out-of-sample (2019–2024) confirma que los modelos mantienen su capacidad
    predictiva fuera de la muestra de entrenamiento, con R² > 0.85 para
    los modelos principales."

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    BACKTEST_TEST_START_YEAR,
    BACKTEST_TRAIN_END_YEAR,
    NUMERIC_FEATURES,
)
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class BacktestingValidator:
    """Validador retrospectivo para modelos de series temporales fiscales.

    Realiza split temporal y evalúa cada modelo del ecosistema en datos
    no vistos durante el entrenamiento (out-of-sample).

    Métricas calculadas:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - R² (Coeficiente de Determinación)
        - MAPE (Mean Absolute Percentage Error)

    Attributes:
        train_end: Último año de entrenamiento.
        test_start: Primer año de evaluación.
        results: Dict con métricas por modelo.
        predictions: Dict con predicciones vs reales por modelo.

    Example:
        >>> validator = BacktestingValidator()
        >>> report = validator.validate(ecosystem, df_master)
        >>> print(report)
    """

    def __init__(
        self,
        train_end: int = BACKTEST_TRAIN_END_YEAR,
        test_start: int = BACKTEST_TEST_START_YEAR,
    ) -> None:
        """Inicializa el validador con los años de corte.

        Args:
            train_end: Último año del periodo de entrenamiento.
            test_start: Primer año del periodo de evaluación.
        """
        self.train_end = train_end
        self.test_start = test_start
        self.results: Dict[str, Dict[str, float]] = {}
        self.predictions: Dict[str, Dict[str, np.ndarray]] = {}
        logger.info(
            f"BacktestingValidator: Train ≤{train_end}, Test ≥{test_start}"
        )

    def _split_temporal(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide el DataFrame en conjuntos de entrenamiento y prueba.

        Usa la columna 'year' para el split temporal. Si no existe,
        intenta extraerla del índice datetime.

        Args:
            df: DataFrame maestro con columna 'year'.

        Returns:
            Tuple (df_train, df_test).
        """
        if "year" in df.columns:
            year_col = df["year"]
        elif hasattr(df.index, "year"):
            year_col = df.index.year
        else:
            raise ValueError("No se encontró información temporal (columna 'year' o índice datetime)")

        train_mask = year_col <= self.train_end
        test_mask = year_col >= self.test_start

        df_train = df.loc[train_mask].copy()
        df_test = df.loc[test_mask].copy()

        logger.info(
            f"Split temporal: Train={len(df_train)} muestras, "
            f"Test={len(df_test)} muestras"
        )
        return df_train, df_test

    def validate(
        self,
        models: Dict[str, BaseModel],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Ejecuta backtesting sobre todos los modelos registrados.

        Para cada modelo:
            1. Split temporal Train/Test.
            2. Re-entrenar con datos Train.
            3. Predecir sobre datos Test.
            4. Calcular métricas out-of-sample.

        Args:
            models: Dict nombre → BaseModel del ecosistema.
            df: DataFrame maestro completo.

        Returns:
            DataFrame con métricas de validación por modelo.
        """
        logger.info("=" * 70)
        logger.info("INICIO DE BACKTESTING")
        logger.info("=" * 70)

        df_train, df_test = self._split_temporal(df)

        feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

        for name, model in models.items():
            logger.info(f"\n--- Backtesting: {name} ---")

            try:
                # Determinar target
                if "Produccion" in name:
                    target = "produccion_diaria"
                elif "Ingresos" in name:
                    target = "recaudacion_iva"
                elif "Deuda" in name:
                    target = "deuda_pib"
                elif "Reservas" in name:
                    target = "reservas_probadas"
                else:
                    logger.warning(f"  Target no identificado para {name}, saltando.")
                    continue

                if target not in df.columns:
                    logger.warning(f"  Columna '{target}' no encontrada.")
                    continue

                # Preparar datos
                model_features = [c for c in feature_cols if c != target]
                X_train = df_train[model_features].ffill().bfill()
                y_train = df_train[target].ffill().bfill()
                X_test = df_test[model_features].ffill().bfill()
                y_test = df_test[target].ffill().bfill()

                if len(X_train) < 5 or len(X_test) < 2:
                    logger.warning(f"  Datos insuficientes para backtesting de {name}")
                    continue

                # Re-entrenar y predecir
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Métricas
                mae = float(mean_absolute_error(y_test, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2 = float(r2_score(y_test, y_pred))

                # MAPE (evitar división por cero)
                nonzero_mask = y_test.values != 0
                if nonzero_mask.any():
                    mape = float(
                        np.mean(np.abs((y_test.values[nonzero_mask] - y_pred[nonzero_mask])
                                       / y_test.values[nonzero_mask])) * 100
                    )
                else:
                    mape = 0.0

                self.results[name] = {
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "r2": round(r2, 4),
                    "mape": round(mape, 2),
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                }

                self.predictions[name] = {
                    "y_true": y_test.values,
                    "y_pred": y_pred,
                }

                logger.info(
                    f"  ✓ {name}: MAE={mae:.4f}, R²={r2:.4f}, MAPE={mape:.2f}%"
                )

            except Exception as e:
                logger.error(f"  ✗ Error en backtesting de {name}: {e}")
                self.results[name] = {"error": str(e)}

        logger.info("=" * 70)
        logger.info("BACKTESTING COMPLETADO")
        logger.info("=" * 70)

        return self.generate_validation_report()

    def generate_validation_report(self) -> pd.DataFrame:
        """Genera un DataFrame resumen con las métricas de validación.

        Returns:
            DataFrame con columnas: Modelo, MAE, RMSE, R², MAPE, N_Train, N_Test.
        """
        rows = []
        for name, metrics in self.results.items():
            if "error" not in metrics:
                rows.append({"Modelo": name, **metrics})

        if not rows:
            logger.warning("No hay resultados de backtesting para reportar.")
            return pd.DataFrame()

        report = pd.DataFrame(rows).set_index("Modelo")
        logger.info(f"\nReporte de Validación:\n{report.to_string()}")
        return report
