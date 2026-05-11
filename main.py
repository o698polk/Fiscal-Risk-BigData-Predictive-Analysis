# -*- coding: utf-8 -*-
"""
Pipeline Principal: main.py
==============================

Punto de entrada que orquesta el pipeline completo del sistema
Fiscal-Risk-BigData-Predictive-Analysis:

    1. ETL → DataFrame Maestro
    2. Feature Engineering → IRFC + PCA
    3. Model Training → Ecosistema de 8 modelos
    4. Backtesting → Validación retrospectiva
    5. Monte Carlo → Probabilidad de colapso (N=50,000)
    6. Scenarios → Proyecciones 2025–2032
    7. Visualization → Gráficos exportados a PNG

Uso:
    python main.py

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import logging
import sys
from pathlib import Path

# Agregar raíz del proyecto al PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LOG_DATE_FORMAT, LOG_FORMAT, OUTPUT_DIR, PLOTS_DIR
from src.etl.fiscal_etl import FiscalETL
from src.features.feature_architect import FeatureArchitect
from src.models.kmeans_regimes import REGIME_LABELS
from src.models.model_ecosystem import ModelEcosystem
from src.scenarios.scenario_generator import ScenarioGenerator
from src.simulation.monte_carlo import MonteCarloEngine
from src.validation.backtesting import BacktestingValidator
from src.visualization.fiscal_visualizer import FiscalVisualizer


def setup_logging() -> None:
    """Configura el sistema de logging con formato estándar."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_pipeline() -> dict:
    """Ejecuta el pipeline completo del sistema fiscal.

    Returns:
        Dict con todos los resultados del análisis.
    """
    setup_logging()
    logger = logging.getLogger("main")

    logger.info("=" * 80)
    logger.info("FISCAL-RISK-BIGDATA-PREDICTIVE-ANALYSIS")
    logger.info("Ecosistema de ML para Sostenibilidad Fiscal — Ecuador 2025–2032")
    logger.info("=" * 80)

    results = {}

    # =========================================================================
    # PASO 1: ETL
    # =========================================================================
    logger.info("\n[PASO 1] Pipeline ETL")
    etl = FiscalETL()
    df_master = etl.build_master_dataframe()
    results["etl"] = {
        "shape": list(df_master.shape),
        "columns": list(df_master.columns),
        "date_range": f"{df_master.index.min()} – {df_master.index.max()}",
    }

    # =========================================================================
    # PASO 2: Ingeniería de Características
    # =========================================================================
    logger.info("\n[PASO 2] Ingenieria de Caracteristicas")
    architect = FeatureArchitect()
    df_featured, feature_metadata = architect.transform(df_master)
    results["features"] = feature_metadata

    # =========================================================================
    # PASO 3: Entrenamiento del Ecosistema
    # =========================================================================
    logger.info("\n[PASO 3] Entrenamiento del Ecosistema de Modelos")
    ecosystem = ModelEcosystem()
    ecosystem.setup_default_models()
    training_metrics = ecosystem.train_all(df_featured)
    results["training_metrics"] = {
        k: v for k, v in training_metrics.items()
        if isinstance(v, dict) and "error" not in v
    }

    # =========================================================================
    # PASO 4: Backtesting
    # =========================================================================
    logger.info("\n[PASO 4] Validacion Retrospectiva (Backtesting)")
    validator = BacktestingValidator()
    backtest_report = validator.validate(ecosystem.models, df_featured)
    results["backtesting"] = validator.results

    # =========================================================================
    # PASO 5: Simulación Monte Carlo
    # =========================================================================
    logger.info("\n[PASO 5] Simulacion Monte Carlo (N=50,000)")

    # Condiciones iniciales (últimos datos de 2024)
    last_row = df_featured.iloc[-1]
    initial_conditions = {
        "produccion_diaria": float(last_row.get("produccion_diaria", 470)),
        "precio_crudo": float(last_row.get("precio_crudo", 72)),
        "deuda_pib": float(last_row.get("deuda_pib", 63)),
        "reservas_probadas": float(last_row.get("reservas_probadas", 1380)),
        "deficit_primario": float(last_row.get("deficit_primario", 4)),
        "empleo_adecuado": float(last_row.get("empleo_adecuado", 38)),
        "ingresos_pge": float(last_row.get("ingresos_pge", 11)),
        "ratio_rp": float(last_row.get("ratio_rp", 8)),
        "recaudacion_iva": float(last_row.get("recaudacion_iva", 8300)),
    }

    mc_engine = MonteCarloEngine()
    mc_engine.run_simulation(initial_conditions)
    results["monte_carlo"] = mc_engine.get_summary()

    # =========================================================================
    # PASO 6: Generación de Escenarios
    # =========================================================================
    logger.info("\n[PASO 6] Generacion de Escenarios 2025-2032")
    scenario_gen = ScenarioGenerator()
    projections = scenario_gen.generate_projections(initial_conditions)
    scenario_gen.export_to_csv()
    results["scenarios"] = {
        name: df.to_dict(orient="records")
        for name, df in projections.items()
    }
    results["scenario_comparison"] = scenario_gen.get_comparison_table().to_dict()
    results["collapse_probabilities"] = scenario_gen.collapse_probs

    # =========================================================================
    # PASO 7: Visualización
    # =========================================================================
    logger.info("\n[PASO 7] Generacion de Visualizaciones")
    visualizer = FiscalVisualizer()

    # Preparar DataFrame anual para gráficos históricos
    df_annual_plot = df_featured.copy()
    if "year" not in df_annual_plot.columns and hasattr(df_annual_plot.index, "year"):
        df_annual_plot["year"] = df_annual_plot.index.year

    # 1. Evolución IRFC
    if "irfc" in df_annual_plot.columns:
        visualizer.plot_irfc_evolution(df_annual_plot)

    # 2. Clústeres PCA
    if ecosystem.clusterer and ecosystem.clusterer.labels is not None:
        pca_cols = [c for c in df_annual_plot.columns if c.startswith("PC")]
        if pca_cols:
            visualizer.plot_pca_clusters(
                df_annual_plot, ecosystem.clusterer.labels, REGIME_LABELS
            )

    # 3. Producción petrolera
    visualizer.plot_production_forecast(df_annual_plot, projections)

    # 4. Monte Carlo
    visualizer.plot_monte_carlo_distribution(
        mc_engine.score_distribution, mc_engine.collapse_probability
    )

    # 5. Comparación de escenarios
    visualizer.plot_scenario_comparison(projections)

    # 6. Backtesting
    if validator.predictions:
        visualizer.plot_backtesting_results(validator.predictions)

    # 7. Trayectoria de deuda
    visualizer.plot_debt_trajectory(projections)

    results["plots"] = visualizer.get_all_plots()

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(">> PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info("=" * 80)
    logger.info(f"  DataFrame Maestro: {df_master.shape}")
    logger.info(f"  Modelos entrenados: {len(ecosystem.models)}")
    logger.info(f"  P(Colapso MC): {mc_engine.collapse_probability:.1%}")
    logger.info(f"  Semáforo Fiscal: {mc_engine.traffic_light}")
    logger.info(f"  Gráficos generados: {len(visualizer.generated_plots)}")
    logger.info(f"  Directorio de salida: {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    run_pipeline()
