# -*- coding: utf-8 -*-
"""
Configuración Global del Sistema Fiscal-Risk-BigData-Predictive-Analysis.
==========================================================================

Centraliza todas las constantes, rutas y parámetros del ecosistema de modelos
para evitar 'magic numbers' y garantizar reproducibilidad.

Referencia:
    Vernaza Quiñonez, P.B. (2025). "El posible colapso de la economía estatal
    dependiente del petróleo y el impuesto dentro de los siguientes 7 años:
    análisis predictivo con Big Data y Machine Learning."

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

from pathlib import Path
from typing import Dict, List

# =============================================================================
# RUTAS DEL SISTEMA
# =============================================================================
# Directorio raíz del proyecto (resuelve automáticamente sin importar dónde
# se ejecute el script, siempre relativo al directorio del archivo config.py)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
CSV_DIR: Path = DATA_DIR / "csv"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
PLOTS_DIR: Path = OUTPUT_DIR / "plots"
MODELS_DIR: Path = OUTPUT_DIR / "models"

# Archivos de datos fuente
ANNUAL_DATASET: Path = CSV_DIR / "Dataset_Macroeconomico_Ecuador_1990_2024.csv"
QUARTERLY_DATASET: Path = CSV_DIR / "Dataset_Trimestral_Ecuador_2005_2024.csv"

# =============================================================================
# COLUMNAS ESTÁNDAR DEL DATASET
# =============================================================================
# Nombres estandarizados internos (snake_case) usados a lo largo del pipeline.
# El ETL mapea las columnas originales del CSV a estas claves.
COLUMN_MAP: Dict[str, str] = {
    "Año": "year",
    "Periodo": "period",
    "Producción Diaria (Kb/d)": "produccion_diaria",
    "Precio del Crudo (USD/bbl)": "precio_crudo",
    "Crecimiento PIB (% anual)": "crecimiento_pib",
    "Deuda/PIB (%)": "deuda_pib",
    "Reservas Probadas (Mb)": "reservas_probadas",
    "Ingresos Petroleros al PGE (% PIB)": "ingresos_pge",
    "Recaudación IVA (USD millones)": "recaudacion_iva",
    "Déficit Primario (% PIB)": "deficit_primario",
    "Empleo Adecuado (% PEA)": "empleo_adecuado",
    "Ratio R/P (Años)": "ratio_rp",
    "Exportaciones Petroleras (USD M)": "exportaciones_petroleras",
    "Inversión Pública (% PIB)": "inversion_publica",
}

# Lista de features numéricas para modelado (excluyendo columnas temporales)
NUMERIC_FEATURES: List[str] = [
    "produccion_diaria",
    "precio_crudo",
    "crecimiento_pib",
    "deuda_pib",
    "reservas_probadas",
    "ingresos_pge",
    "recaudacion_iva",
    "deficit_primario",
    "empleo_adecuado",
    "ratio_rp",
    "exportaciones_petroleras",
    "inversion_publica",
]

# =============================================================================
# PONDERACIONES DEL ÍNDICE DE RIESGO FISCAL COMPUESTO (IRFC)
# =============================================================================
# Referencia: Vernaza Quiñonez, Tabla 8 – Composición del IRFC.
# El IRFC sintetiza 5 indicadores clave en un score normalizado [0-100]
# donde 0 = mínimo riesgo fiscal y 100 = máximo riesgo fiscal.
IRFC_WEIGHTS: Dict[str, float] = {
    "deuda_pib": 0.30,       # 30% – Ratio Deuda/PIB
    "ratio_rp": 0.25,        # 25% – Ratio Reservas/Producción (invertido)
    "deficit_primario": 0.20, # 20% – Déficit Primario (% PIB)
    "empleo_adecuado": 0.15,  # 15% – Empleo adecuado (invertido)
    "ingresos_pge": 0.10,    # 10% – Ingresos Petroleros al PGE (invertido)
}

# =============================================================================
# PARÁMETROS DE PCA
# =============================================================================
# Referencia: Vernaza Quiñonez, Sección 4.5 – Análisis de Componentes Principales.
# La investigación identifica 2 ejes estructurales: Petróleo-PIB y Deuda-Déficit.
# Se retienen 4 componentes para capturar >80% de la varianza.
PCA_N_COMPONENTS: int = 4
PCA_VARIANCE_THRESHOLD: float = 0.80

# =============================================================================
# PARÁMETROS DE K-MEANS (REGÍMENES ECONÓMICOS)
# =============================================================================
# Referencia: Vernaza Quiñonez, Sección 4.4 – Clustering K-Means.
# Se identifican 7 regímenes económicos históricos: Pre-dolarización Crisis,
# Dolarización Temprana, Boom Petrolero I, Crisis Global, Boom Petrolero II,
# Recesión Petrolera, Pandemia y Post-pandemia.
KMEANS_N_CLUSTERS: int = 7

# =============================================================================
# PARÁMETROS DE MODELOS ML
# =============================================================================
# Random Forest Híbrido (Producción petrolera)
RF_N_ESTIMATORS: int = 200
RF_MAX_DEPTH: int = 8
ARPS_HYBRID_WEIGHT: float = 0.50  # 50% RF + 50% Arps

# Gradient Boosting (Ingresos y Deuda)
GB_N_ESTIMATORS: int = 300
GB_LEARNING_RATE: float = 0.05
GB_MAX_DEPTH: int = 5
GB_SUBSAMPLE: float = 0.8

# SVR (Reservas Probadas)
SVR_KERNEL: str = "rbf"
SVR_C_RANGE: List[float] = [0.1, 1.0, 10.0, 100.0]
SVR_GAMMA_RANGE: List[str] = ["scale", "auto"]

# =============================================================================
# PARÁMETROS DE SIMULACIÓN MONTE CARLO
# =============================================================================
# Referencia: Vernaza Quiñonez, Sección 4.6 – Simulación Monte Carlo.
# N = 50,000 iteraciones estocásticas para estimar P(colapso fiscal).
MC_N_SIMULATIONS: int = 50_000

# Distribuciones de los parámetros estocásticos
# Calibración: Vernaza Quiñonez, Sección 4.6.
# Ecuador enfrenta declive natural acelerado (campos maduros ITT/Sacha),
# precios volátiles post-OPEC+, y crecimiento estructuralmente bajo.
MC_DELTA_MEAN: float = 0.048    # Tasa de declive petrolero media (4.8%/año)
MC_DELTA_STD: float = 0.018     # Desviación estándar del declive
MC_PRICE_MEAN: float = 52.0     # Precio crudo medio (USD/bbl) — Oriente discount
MC_PRICE_STD: float = 20.0      # Desviación estándar precio (alta volatilidad)
MC_GROWTH_MEAN: float = 0.8     # Crecimiento PIB medio (%) — estancamiento secular
MC_GROWTH_STD: float = 2.5      # Desviación estándar crecimiento

# Umbrales de colapso fiscal (cada indicador binario vale 1 punto)
# Calibración: Vernaza Quiñonez, Tabla 10 — Umbrales de sostenibilidad.
# Adaptados a la realidad fiscal ecuatoriana (economía dolarizada sin
# política monetaria autónoma, lo que reduce los márgenes de maniobra).
COLLAPSE_THRESHOLDS: Dict[str, float] = {
    "deuda_pib_max": 57.0,           # Deuda/PIB > 57% (umbral COPLAFIP)
    "ratio_rp_min": 7.0,             # Ratio R/P < 7 años (agotamiento)
    "deficit_max": 4.5,              # Déficit > 4.5% PIB (regla fiscal)
    "produccion_min": 400.0,         # Producción < 400 Kb/d (umbral fiscal)
    "irfc_max": 60.0,                # IRFC > 60 (riesgo elevado)
    "empleo_min": 35.0,              # Empleo adecuado < 35% PEA
    "reservas_min": 800.0,           # Reservas < 800 Mb (horizonte corto)
}
COLLAPSE_SCORE_THRESHOLD: int = 5   # Score >= 5 = Colapso Fiscal

# =============================================================================
# PARÁMETROS DE BACKTESTING
# =============================================================================
# Referencia: Vernaza Quiñonez, Sección 4.8 – Validación Retrospectiva.
# Split temporal: entrenar 1990–2018, evaluar 2019–2024.
BACKTEST_TRAIN_END_YEAR: int = 2018
BACKTEST_TEST_START_YEAR: int = 2019

# =============================================================================
# ESCENARIOS DE PROYECCIÓN (2025–2032)
# =============================================================================
# Referencia: Vernaza Quiñonez, Tabla 12 – Escenarios prospectivos.
PROJECTION_START_YEAR: int = 2025
PROJECTION_END_YEAR: int = 2032

SCENARIOS: Dict[str, Dict[str, float]] = {
    "optimista": {
        "precio_crudo": 75.0,         # USD/bbl
        "crecimiento_pib": 3.0,       # % anual
        "declive_produccion": -0.020,  # -2.0% anual
        "deficit_tendencia": -0.3,     # Mejora gradual
    },
    "base": {
        "precio_crudo": 60.0,
        "crecimiento_pib": 1.5,
        "declive_produccion": -0.035,
        "deficit_tendencia": 0.2,
    },
    "pesimista": {
        "precio_crudo": 40.0,
        "crecimiento_pib": 0.0,
        "declive_produccion": -0.050,
        "deficit_tendencia": 0.8,
    },
}

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
RANDOM_SEED: int = 42
LOG_FORMAT: str = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
