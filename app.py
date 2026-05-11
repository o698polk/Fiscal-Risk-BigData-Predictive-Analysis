# -*- coding: utf-8 -*-
"""
Servidor Web Flask: app.py
============================

Proporciona una API REST y sirve la interfaz web para el sistema
Fiscal-Risk-BigData-Predictive-Analysis.

Endpoints:
    GET  /                  → Interfaz web principal
    POST /api/run-etl       → Ejecutar pipeline ETL
    POST /api/run-features  → Ejecutar ingeniería de características
    POST /api/run-training  → Entrenar ecosistema de modelos
    POST /api/run-backtest  → Ejecutar backtesting
    POST /api/run-montecarlo→ Ejecutar simulación Monte Carlo
    POST /api/run-scenarios → Generar escenarios
    POST /api/run-all       → Ejecutar pipeline completo
    GET  /api/plots/<name>  → Servir gráfico PNG generado
    GET  /api/status        → Estado actual del sistema

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import json
import logging
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from src.config import LOG_DATE_FORMAT, LOG_FORMAT, OUTPUT_DIR, PLOTS_DIR
from src.etl.fiscal_etl import FiscalETL
from src.features.feature_architect import FeatureArchitect
from src.models.kmeans_regimes import REGIME_LABELS
from src.models.model_ecosystem import ModelEcosystem
from src.scenarios.scenario_generator import ScenarioGenerator
from src.simulation.monte_carlo import MonteCarloEngine
from src.validation.backtesting import BacktestingValidator
from src.visualization.fiscal_visualizer import FiscalVisualizer

# =============================================================================
# Configuración Flask
# =============================================================================
app = Flask(
    __name__,
    static_folder=str(PROJECT_ROOT / "web" / "static"),
    template_folder=str(PROJECT_ROOT / "web"),
)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")

# =============================================================================
# Estado global del sistema (se actualiza con cada paso)
# =============================================================================
state = {
    "df_master": None,
    "df_featured": None,
    "feature_metadata": None,
    "ecosystem": None,
    "training_metrics": None,
    "backtest_results": None,
    "backtest_predictions": None,
    "mc_engine": None,
    "scenario_gen": None,
    "projections": None,
    "visualizer": None,
    "initial_conditions": None,
    "status": {
        "etl": False,
        "features": False,
        "training": False,
        "backtesting": False,
        "montecarlo": False,
        "scenarios": False,
        "plots": False,
    },
}


# =============================================================================
# Rutas de la Interfaz Web
# =============================================================================
@app.route("/")
def index():
    """Sirve la página principal de la interfaz web."""
    return send_from_directory(str(PROJECT_ROOT / "web"), "index.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Sirve archivos estáticos (CSS, JS)."""
    return send_from_directory(str(PROJECT_ROOT / "web" / "static"), filename)


# =============================================================================
# API: Estado del Sistema
# =============================================================================
@app.route("/api/status", methods=["GET"])
def get_status():
    """Retorna el estado actual de cada módulo del sistema."""
    return jsonify({"success": True, "status": state["status"]})


# =============================================================================
# API: Pipeline ETL
# =============================================================================
@app.route("/api/run-etl", methods=["POST"])
def run_etl():
    """Ejecuta el pipeline ETL completo."""
    try:
        logger.info("[ETL] API: Ejecutando ETL...")
        etl = FiscalETL()
        df = etl.build_master_dataframe()
        state["df_master"] = df
        state["status"]["etl"] = True

        return jsonify({
            "success": True,
            "message": "Pipeline ETL completado exitosamente.",
            "data": {
                "shape": list(df.shape),
                "columns": list(df.columns),
                "sample": df.head(5).to_dict(orient="records"),
                "date_range": f"{df.index.min()} – {df.index.max()}",
                "null_count": int(df.isnull().sum().sum()),
            },
        })
    except Exception as e:
        logger.error(f"Error en ETL: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# API: Ingeniería de Características
# =============================================================================
@app.route("/api/run-features", methods=["POST"])
def run_features():
    """Ejecuta la ingeniería de características (IRFC + PCA)."""
    try:
        if state["df_master"] is None:
            return jsonify({"success": False, "error": "Ejecute el ETL primero."}), 400

        logger.info("[FEATURES] API: Ejecutando Feature Engineering...")
        architect = FeatureArchitect()
        df_featured, metadata = architect.transform(state["df_master"])
        state["df_featured"] = df_featured
        state["feature_metadata"] = metadata
        state["status"]["features"] = True

        return jsonify({
            "success": True,
            "message": "Ingeniería de características completada.",
            "data": {
                "irfc_stats": metadata["irfc_stats"],
                "pca_variance": metadata["pca_variance"],
                "new_columns": [c for c in df_featured.columns if c.startswith("PC") or c == "irfc"],
                "shape": list(df_featured.shape),
            },
        })
    except Exception as e:
        logger.error(f"Error en Features: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# API: Entrenamiento de Modelos
# =============================================================================
@app.route("/api/run-training", methods=["POST"])
def run_training():
    """Entrena el ecosistema completo de modelos."""
    try:
        if state["df_featured"] is None:
            return jsonify({"success": False, "error": "Ejecute Features primero."}), 400

        logger.info("[TRAINING] API: Entrenando ecosistema...")
        ecosystem = ModelEcosystem()
        ecosystem.setup_default_models()
        metrics = ecosystem.train_all(state["df_featured"])
        state["ecosystem"] = ecosystem
        state["training_metrics"] = metrics
        state["status"]["training"] = True

        # Extraer condiciones iniciales
        last_row = state["df_featured"].iloc[-1]
        state["initial_conditions"] = {
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

        # Serializar métricas (convertir floats)
        safe_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                safe_metrics[k] = {
                    mk: round(float(mv), 4) if isinstance(mv, (int, float)) else str(mv)
                    for mk, mv in v.items()
                }

        return jsonify({
            "success": True,
            "message": f"Ecosistema entrenado: {len(ecosystem.models)} modelos.",
            "data": {
                "models_trained": list(ecosystem.models.keys()),
                "metrics": safe_metrics,
            },
        })
    except Exception as e:
        logger.error(f"Error en Training: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# API: Backtesting
# =============================================================================
@app.route("/api/run-backtest", methods=["POST"])
def run_backtest():
    """Ejecuta la validación retrospectiva (backtesting)."""
    try:
        if state["ecosystem"] is None:
            return jsonify({"success": False, "error": "Entrene modelos primero."}), 400

        logger.info("[BACKTEST] API: Ejecutando backtesting...")
        validator = BacktestingValidator()
        report = validator.validate(state["ecosystem"].models, state["df_featured"])
        state["backtest_results"] = validator.results
        state["backtest_predictions"] = validator.predictions
        state["status"]["backtesting"] = True

        # Generar gráfico de backtesting
        visualizer = FiscalVisualizer()
        plot_path = ""
        if validator.predictions:
            plot_path = visualizer.plot_backtesting_results(validator.predictions)

        # Serializar resultados
        safe_results = {}
        for k, v in validator.results.items():
            if isinstance(v, dict):
                safe_results[k] = {
                    mk: round(float(mv), 4) if isinstance(mv, (int, float)) else str(mv)
                    for mk, mv in v.items()
                }

        return jsonify({
            "success": True,
            "message": "Backtesting completado.",
            "data": {
                "metrics": safe_results,
                "plot": Path(plot_path).name if plot_path else None,
            },
        })
    except Exception as e:
        logger.error(f"Error en Backtesting: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# API: Monte Carlo
# =============================================================================
@app.route("/api/run-montecarlo", methods=["POST"])
def run_montecarlo():
    """Ejecuta la simulación Monte Carlo (N=50,000)."""
    try:
        if state["initial_conditions"] is None:
            return jsonify({"success": False, "error": "Entrene modelos primero."}), 400

        logger.info("[MONTECARLO] API: Ejecutando Monte Carlo (N=50,000)...")
        mc = MonteCarloEngine()
        mc.run_simulation(state["initial_conditions"])
        state["mc_engine"] = mc
        state["status"]["montecarlo"] = True

        # Generar gráfico
        visualizer = FiscalVisualizer()
        plot_path = visualizer.plot_monte_carlo_distribution(
            mc.score_distribution, mc.collapse_probability
        )

        return jsonify({
            "success": True,
            "message": f"Monte Carlo completado. P(colapso) = {mc.collapse_probability:.1%}",
            "data": {
                **mc.get_summary(),
                "plot": Path(plot_path).name if plot_path else None,
            },
        })
    except Exception as e:
        logger.error(f"Error en Monte Carlo: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# API: Escenarios
# =============================================================================
@app.route("/api/run-scenarios", methods=["POST"])
def run_scenarios():
    """Genera escenarios y visualizaciones finales."""
    try:
        if state["initial_conditions"] is None:
            return jsonify({"success": False, "error": "Entrene modelos primero."}), 400

        logger.info("[SCENARIOS] API: Generando escenarios 2025-2032...")
        gen = ScenarioGenerator()
        projections = gen.generate_projections(state["initial_conditions"])
        gen.export_to_csv()
        state["scenario_gen"] = gen
        state["projections"] = projections
        state["status"]["scenarios"] = True

        # Generar gráficos
        visualizer = FiscalVisualizer()

        # Preparar historical plot
        df_plot = state["df_featured"].copy()
        if "year" not in df_plot.columns and hasattr(df_plot.index, "year"):
            df_plot["year"] = df_plot.index.year

        plots = []

        if "irfc" in df_plot.columns:
            plots.append(Path(visualizer.plot_irfc_evolution(df_plot)).name)

        plots.append(Path(visualizer.plot_production_forecast(df_plot, projections)).name)
        plots.append(Path(visualizer.plot_scenario_comparison(projections)).name)
        plots.append(Path(visualizer.plot_debt_trajectory(projections)).name)

        # PCA clusters
        if state["ecosystem"] and state["ecosystem"].clusterer and state["ecosystem"].clusterer.labels is not None:
            pca_cols = [c for c in df_plot.columns if c.startswith("PC")]
            if pca_cols:
                plots.append(
                    Path(visualizer.plot_pca_clusters(
                        df_plot, state["ecosystem"].clusterer.labels, REGIME_LABELS
                    )).name
                )

        state["status"]["plots"] = True

        # Comparison table
        comparison = gen.get_comparison_table()
        comparison_data = comparison.reset_index().to_dict(orient="records")

        # Scenarios data
        scenario_data = {}
        for name, df_sc in projections.items():
            scenario_data[name] = df_sc.to_dict(orient="records")

        return jsonify({
            "success": True,
            "message": "Escenarios generados exitosamente.",
            "data": {
                "comparison": comparison_data,
                "collapse_probs": gen.collapse_probs,
                "scenarios": scenario_data,
                "plots": plots,
            },
        })
    except Exception as e:
        logger.error(f"Error en Scenarios: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# API: Pipeline Completo
# =============================================================================
@app.route("/api/run-all", methods=["POST"])
def run_all():
    """Ejecuta todos los pasos del pipeline en secuencia."""
    try:
        results = {}

        # ETL
        etl = FiscalETL()
        df = etl.build_master_dataframe()
        state["df_master"] = df
        state["status"]["etl"] = True

        # Features
        architect = FeatureArchitect()
        df_featured, metadata = architect.transform(df)
        state["df_featured"] = df_featured
        state["feature_metadata"] = metadata
        state["status"]["features"] = True

        # Training
        ecosystem = ModelEcosystem()
        ecosystem.setup_default_models()
        metrics = ecosystem.train_all(df_featured)
        state["ecosystem"] = ecosystem
        state["training_metrics"] = metrics
        state["status"]["training"] = True

        # Initial conditions
        last_row = df_featured.iloc[-1]
        state["initial_conditions"] = {
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

        # Backtesting
        validator = BacktestingValidator()
        validator.validate(ecosystem.models, df_featured)
        state["backtest_results"] = validator.results
        state["backtest_predictions"] = validator.predictions
        state["status"]["backtesting"] = True

        # Monte Carlo
        mc = MonteCarloEngine()
        mc.run_simulation(state["initial_conditions"])
        state["mc_engine"] = mc
        state["status"]["montecarlo"] = True

        # Scenarios
        gen = ScenarioGenerator()
        projections = gen.generate_projections(state["initial_conditions"])
        gen.export_to_csv()
        state["scenario_gen"] = gen
        state["projections"] = projections
        state["status"]["scenarios"] = True

        # Visualizations
        viz = FiscalVisualizer()
        df_plot = df_featured.copy()
        if "year" not in df_plot.columns and hasattr(df_plot.index, "year"):
            df_plot["year"] = df_plot.index.year

        plots = []
        if "irfc" in df_plot.columns:
            plots.append(Path(viz.plot_irfc_evolution(df_plot)).name)
        if ecosystem.clusterer and ecosystem.clusterer.labels is not None:
            pca_cols = [c for c in df_plot.columns if c.startswith("PC")]
            if pca_cols:
                plots.append(Path(viz.plot_pca_clusters(
                    df_plot, ecosystem.clusterer.labels, REGIME_LABELS
                )).name)
        plots.append(Path(viz.plot_production_forecast(df_plot, projections)).name)
        plots.append(Path(viz.plot_monte_carlo_distribution(
            mc.score_distribution, mc.collapse_probability
        )).name)
        plots.append(Path(viz.plot_scenario_comparison(projections)).name)
        if validator.predictions:
            plots.append(Path(viz.plot_backtesting_results(validator.predictions)).name)
        plots.append(Path(viz.plot_debt_trajectory(projections)).name)
        state["status"]["plots"] = True

        return jsonify({
            "success": True,
            "message": "Pipeline completo ejecutado exitosamente.",
            "data": {
                "etl_shape": list(df.shape),
                "irfc_mean": round(float(df_featured["irfc"].mean()), 2) if "irfc" in df_featured.columns else None,
                "collapse_probability": round(mc.collapse_probability * 100, 2),
                "traffic_light": mc.traffic_light,
                "plots": plots,
                "collapse_probs": gen.collapse_probs,
            },
        })
    except Exception as e:
        logger.error(f"Error en pipeline: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# API: Servir Gráficos
# =============================================================================
@app.route("/api/plots/<filename>")
def serve_plot(filename):
    """Sirve un gráfico PNG generado."""
    plot_path = PLOTS_DIR / filename
    if plot_path.exists():
        return send_file(str(plot_path), mimetype="image/png")
    return jsonify({"error": "Plot not found"}), 404


# =============================================================================
# Inicio
# =============================================================================
if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(">> Servidor Flask iniciando en http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
