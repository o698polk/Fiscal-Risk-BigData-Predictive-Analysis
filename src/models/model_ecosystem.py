# -*- coding: utf-8 -*-
"""
Orquestador de Modelos: ModelEcosystem.
=========================================

Gestiona el ecosistema completo de 8 modelos, orquestando su entrenamiento,
predicción y serialización de forma unificada.

Principio SOLID aplicado:
    O – Open/Closed: Se pueden registrar nuevos modelos sin modificar
        esta clase, usando register_model().
    D – Dependency Inversion: Depende de la abstracción BaseModel,
        no de implementaciones concretas.

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, NUMERIC_FEATURES
from src.models.base_model import BaseModel
from src.models.gradient_boosting import GBDebtModel, GBRevenueModel
from src.models.hybrid_rf_arps import HybridRFArpsModel
from src.models.kmeans_regimes import EconomicRegimeClusterer
from src.models.svr_reserves import SVRReservesModel

logger = logging.getLogger(__name__)


class ModelEcosystem:
    """Orquestador del ecosistema de modelos de riesgo fiscal.

    Registra, entrena y gestiona todos los modelos del sistema,
    proporcionando una interfaz unificada para predicción y exportación.

    Attributes:
        models: Dict de modelos registrados (nombre → instancia).
        clusterer: Instancia del clustering K-Means.
        predictions: Dict con las últimas predicciones generadas.

    Example:
        >>> ecosystem = ModelEcosystem()
        >>> ecosystem.setup_default_models()
        >>> ecosystem.train_all(df, targets)
        >>> predictions = ecosystem.predict_all(X_new)
    """

    def __init__(self) -> None:
        """Inicializa el ecosistema vacío."""
        self.models: Dict[str, BaseModel] = {}
        self.clusterer: Optional[EconomicRegimeClusterer] = None
        self.predictions: Dict[str, np.ndarray] = {}
        logger.info("ModelEcosystem inicializado.")

    def register_model(self, model: BaseModel) -> None:
        """Registra un modelo en el ecosistema (Open/Closed Principle).

        Args:
            model: Instancia de BaseModel a registrar.
        """
        name = model.get_model_name()
        self.models[name] = model
        logger.info(f"Modelo registrado: {name}")

    def setup_default_models(self) -> None:
        """Configura el ecosistema con los 8 modelos por defecto.

        Modelos registrados:
            1. HybridRFArpsModel – Producción petrolera
            2. GBRevenueModel – Ingresos fiscales
            3. GBDebtModel – Deuda/PIB
            4. SVRReservesModel – Reservas probadas
            5. EconomicRegimeClusterer – Regímenes K-Means
        """
        logger.info("Configurando ecosistema con modelos por defecto...")

        self.register_model(HybridRFArpsModel())
        self.register_model(GBRevenueModel())
        self.register_model(GBDebtModel())
        self.register_model(SVRReservesModel())
        self.clusterer = EconomicRegimeClusterer()

        logger.info(f"Ecosistema configurado con {len(self.models)} modelos predictivos + K-Means")

    def train_all(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Entrena todos los modelos registrados con el DataFrame maestro.

        Cada modelo se entrena con las features apropiadas y su variable
        objetivo correspondiente.

        Args:
            df: DataFrame maestro con features e indicadores.

        Returns:
            Dict con métricas de entrenamiento por modelo.
        """
        logger.info("=" * 70)
        logger.info("INICIO DE ENTRENAMIENTO DEL ECOSISTEMA")
        logger.info("=" * 70)

        metrics_report: Dict[str, Dict[str, float]] = {}

        # Identificar features disponibles
        feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        X = df[feature_cols].copy()

        # Limpiar NaN en features (forward-fill + backward-fill)
        X = X.ffill().bfill()

        for name, model in self.models.items():
            logger.info(f"\n--- Entrenando: {name} ---")

            try:
                # Determinar variable objetivo según el modelo
                if "Produccion" in name:
                    y = df["produccion_diaria"].copy()
                    X_model = X.drop(columns=["produccion_diaria"], errors="ignore")
                elif "Ingresos" in name:
                    y = df["recaudacion_iva"].copy()
                    X_model = X.drop(columns=["recaudacion_iva"], errors="ignore")
                elif "Deuda" in name:
                    y = df["deuda_pib"].copy()
                    X_model = X.drop(columns=["deuda_pib"], errors="ignore")
                elif "Reservas" in name:
                    y = df["reservas_probadas"].copy()
                    X_model = X.drop(columns=["reservas_probadas"], errors="ignore")
                else:
                    logger.warning(f"No se pudo determinar target para {name}")
                    continue

                # Limpiar y alinear
                valid_mask = y.notna() & X_model.notna().all(axis=1)
                X_clean = X_model.loc[valid_mask]
                y_clean = y.loc[valid_mask]

                if len(X_clean) < 10:
                    logger.warning(f"  Datos insuficientes para {name}: {len(X_clean)} muestras")
                    continue

                model.fit(X_clean, y_clean)
                metrics = model.get_metrics()
                metrics_report[name] = metrics

                logger.info(
                    f"  ✓ {name} entrenado: MAE={metrics.get('mae', 'N/A')}, "
                    f"R²={metrics.get('r2', 'N/A')}"
                )

            except Exception as e:
                logger.error(f"  ✗ Error entrenando {name}: {e}")
                metrics_report[name] = {"error": str(e)}

        # K-Means (no supervisado)
        if self.clusterer is not None:
            pca_cols = [c for c in df.columns if c.startswith("PC")]
            if pca_cols:
                logger.info(f"\n--- Ejecutando K-Means sobre {pca_cols} ---")
                labels = self.clusterer.fit_predict(df[pca_cols], original_df=df)
                metrics_report["KMeans_Regimenes"] = self.clusterer.get_metrics()
                logger.info(
                    f"  ✓ K-Means: silhouette={self.clusterer.silhouette:.4f}"
                )

        logger.info("=" * 70)
        logger.info("ENTRENAMIENTO DEL ECOSISTEMA COMPLETADO")
        logger.info("=" * 70)

        return metrics_report

    def predict_all(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Genera predicciones con todos los modelos entrenados.

        Args:
            X: Features para predicción.

        Returns:
            Dict nombre_modelo → array de predicciones.
        """
        self.predictions = {}
        feature_cols = [c for c in NUMERIC_FEATURES if c in X.columns]

        for name, model in self.models.items():
            if not model.is_fitted:
                logger.warning(f"Modelo {name} no entrenado, saltando predicción.")
                continue

            try:
                if "Produccion" in name:
                    X_pred = X[feature_cols].drop(columns=["produccion_diaria"], errors="ignore")
                elif "Ingresos" in name:
                    X_pred = X[feature_cols].drop(columns=["recaudacion_iva"], errors="ignore")
                elif "Deuda" in name:
                    X_pred = X[feature_cols].drop(columns=["deuda_pib"], errors="ignore")
                elif "Reservas" in name:
                    X_pred = X[feature_cols].drop(columns=["reservas_probadas"], errors="ignore")
                else:
                    X_pred = X[feature_cols]

                X_pred = X_pred.ffill().bfill()
                self.predictions[name] = model.predict(X_pred)

            except Exception as e:
                logger.error(f"Error prediciendo con {name}: {e}")

        return self.predictions

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Retorna las métricas de todos los modelos."""
        report = {}
        for name, model in self.models.items():
            report[name] = model.get_metrics()
        if self.clusterer:
            report["KMeans_Regimenes"] = self.clusterer.get_metrics()
        return report

    def export_models(self, output_dir: Optional[Path] = None) -> None:
        """Serializa todos los modelos entrenados a disco con joblib.

        Args:
            output_dir: Directorio de salida (default: output/models/).
        """
        out = output_dir or MODELS_DIR
        out.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            if model.is_fitted:
                path = out / f"{name}.joblib"
                joblib.dump(model, path)
                logger.info(f"Modelo exportado: {path}")

        if self.clusterer and self.clusterer.model is not None:
            path = out / "KMeans_Regimenes.joblib"
            joblib.dump(self.clusterer, path)
            logger.info(f"Clusterer exportado: {path}")
