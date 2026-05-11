# -*- coding: utf-8 -*-
"""
Motor de Simulación Monte Carlo: MonteCarloEngine.
=====================================================

Ejecuta 50,000 iteraciones estocásticas para estimar la probabilidad
de colapso fiscal de Ecuador (2025–2032), variando tres parámetros:
    - δ (delta): Tasa de declive de producción petrolera
    - W: Precio del crudo (USD/bbl)
    - γ (gamma): Crecimiento del PIB (% anual)

Condición de Colapso: Score ≥ 5 de 7 indicadores binarios.

Referencia:
    Vernaza Quiñonez, P.B. (2025). Sección 4.6 – "La simulación Monte Carlo
    (N=50,000) estima una probabilidad de colapso fiscal del 62.4% antes de
    2032 bajo el escenario base, definido como el incumplimiento simultáneo
    de al menos 5 de los 7 umbrales de sostenibilidad."

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    COLLAPSE_SCORE_THRESHOLD,
    COLLAPSE_THRESHOLDS,
    MC_DELTA_MEAN,
    MC_DELTA_STD,
    MC_GROWTH_MEAN,
    MC_GROWTH_STD,
    MC_N_SIMULATIONS,
    MC_PRICE_MEAN,
    MC_PRICE_STD,
    PROJECTION_END_YEAR,
    PROJECTION_START_YEAR,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """Motor de simulación estocástica para riesgo de colapso fiscal.

    Ejecuta N simulaciones variando parámetros macroeconómicos clave
    para estimar distribuciones de probabilidad de los indicadores
    fiscales y la probabilidad de colapso.

    Cada simulación:
        1. Muestrea δ, W, γ de distribuciones normales/lognormales.
        2. Proyecta producción, ingresos, deuda y reservas año a año.
        3. Calcula el score de colapso (0–7) en cada año proyectado.
        4. Registra si ocurre colapso (score ≥ 5) en algún año.

    Attributes:
        n_simulations: Número de iteraciones Monte Carlo.
        results: DataFrame con resultados de todas las simulaciones.
        collapse_probability: Probabilidad estimada de colapso.
        score_distribution: Distribución de scores de colapso.
        traffic_light: Resultado del semáforo fiscal.

    Example:
        >>> engine = MonteCarloEngine(n_simulations=50000)
        >>> results = engine.run_simulation(initial_conditions)
        >>> print(f"P(colapso) = {engine.collapse_probability:.1%}")
    """

    def __init__(self, n_simulations: int = MC_N_SIMULATIONS) -> None:
        """Inicializa el motor con el número de simulaciones.

        Args:
            n_simulations: Número de iteraciones (default: 50,000).
        """
        self.n_simulations = n_simulations
        self.results: Optional[pd.DataFrame] = None
        self.collapse_probability: float = 0.0
        self.score_distribution: Optional[np.ndarray] = None
        self.traffic_light: str = "VERDE"
        self.yearly_collapse_prob: Dict[int, float] = {}
        logger.info(f"MonteCarloEngine inicializado (N={n_simulations:,})")

    def run_simulation(
        self,
        initial_conditions: Dict[str, float],
        irfc_base: float = 50.0,
    ) -> pd.DataFrame:
        """Ejecuta la simulación Monte Carlo completa.

        Procedimiento para cada iteración i ∈ [1, N]:
            1. Muestrear: δ_i ~ N(μ_δ, σ_δ), W_i ~ N(μ_W, σ_W), γ_i ~ N(μ_γ, σ_γ)
            2. Para cada año t ∈ [2025, 2032]:
                a. Producción: q(t) = q(t-1) · (1 - δ_i)
                b. Ingresos PGE: f(W_i, q(t))
                c. Deuda/PIB: d(t-1) · (1+r)/(1+γ_i) + déficit
                d. Reservas: R(t) = R(t-1) - q(t)·365/1e6
                e. IRFC: calculado con las variables proyectadas
            3. Score = Σ(indicadores binarios de colapso)
            4. Colapso = (Score ≥ 5)

        Args:
            initial_conditions: Dict con valores iniciales (2024):
                - produccion_diaria: Producción (Kb/d)
                - precio_crudo: Precio (USD/bbl)
                - deuda_pib: Deuda/PIB (%)
                - reservas_probadas: Reservas (Mb)
                - deficit_primario: Déficit (% PIB)
                - empleo_adecuado: Empleo (% PEA)
                - ingresos_pge: Ingresos PGE (% PIB)
                - ratio_rp: Ratio R/P (años)
            irfc_base: Valor base del IRFC para punto de partida.

        Returns:
            DataFrame con resultados agregados por simulación.
        """
        logger.info(f"Iniciando simulación Monte Carlo (N={self.n_simulations:,})...")

        np.random.seed(RANDOM_SEED)

        # Muestreo de parámetros estocásticos
        deltas = np.random.normal(MC_DELTA_MEAN, MC_DELTA_STD, self.n_simulations)
        deltas = np.clip(deltas, 0.005, 0.10)  # Bounds físicos

        prices = np.random.normal(MC_PRICE_MEAN, MC_PRICE_STD, self.n_simulations)
        prices = np.clip(prices, 15.0, 150.0)

        growths = np.random.normal(MC_GROWTH_MEAN, MC_GROWTH_STD, self.n_simulations)
        growths = np.clip(growths, -10.0, 10.0)

        years = list(range(PROJECTION_START_YEAR, PROJECTION_END_YEAR + 1))
        n_years = len(years)

        # Arrays de resultados
        max_scores = np.zeros(self.n_simulations)
        collapse_flags = np.zeros(self.n_simulations, dtype=bool)
        final_production = np.zeros(self.n_simulations)
        final_debt = np.zeros(self.n_simulations)
        final_reserves = np.zeros(self.n_simulations)

        # Contadores por año
        yearly_collapse_counts = np.zeros(n_years)

        # Condiciones iniciales — Ecuador 2024
        # Fuente: BCE, Petroecuador, MEF, INEC
        q0 = initial_conditions.get("produccion_diaria", 470.0)
        d0 = initial_conditions.get("deuda_pib", 63.0)
        r0 = initial_conditions.get("reservas_probadas", 1380.0)
        def0 = initial_conditions.get("deficit_primario", 4.0)
        emp0 = initial_conditions.get("empleo_adecuado", 38.0)
        ing0 = initial_conditions.get("ingresos_pge", 11.0)
        rp0 = initial_conditions.get("ratio_rp", 8.0)

        for i in range(self.n_simulations):
            delta_i = deltas[i]
            price_i = prices[i]
            growth_i = growths[i]

            q_t = q0
            d_t = d0
            r_t = r0
            def_t = def0
            emp_t = emp0
            max_score = 0

            for j, year in enumerate(years):
                # --- Proyección año a año ---
                # Cada año incorpora shocks estocásticos adicionales
                # para capturar la volatilidad de la economía ecuatoriana.

                # Shock anual de precio (varía cada año, no fijo)
                price_year = price_i + np.random.normal(0, 8)
                price_year = max(price_year, 18.0)

                # Shock anual de crecimiento
                growth_year = growth_i + np.random.normal(0, 1.2)

                # Producción: declive exponencial + shock + deterioro
                # Los campos maduros (Sacha, Shushufindi, ITT) pierden
                # presión de yacimiento de forma no lineal.
                q_t = q_t * (1.0 - delta_i) + np.random.normal(0, 8)
                q_t = max(q_t, 50.0)  # Mínimo físico

                # Ingresos PGE: función del precio y producción
                # El crudo Oriente tiene un descuento de ~$8-12/bbl vs WTI.
                # La caída combinada de precio Y producción tiene efecto
                # multiplicativo sobre los ingresos fiscales.
                ing_t = ing0 * (price_year / 65.0) * (q_t / q0)
                ing_t = np.clip(ing_t, 2.0, 25.0)

                # Deuda/PIB: dinámica de acumulación
                # Ecuador emite deuda a tasas altas (7-10%) por su
                # calificación crediticia (B-/CCC+). Sin política monetaria,
                # la dolarización impide monetizar el déficit.
                r_implicit = 0.058 + np.random.normal(0, 0.008)
                d_t = d_t * (1 + r_implicit) / (1 + growth_year / 100.0)
                # El déficit se financia con nueva deuda
                d_t += def_t * (0.8 + 0.2 * np.random.rand())
                d_t = max(d_t, 10.0)

                # Déficit: tendencia creciente por rigidez del gasto
                # Los subsidios (combustibles ~$3B/año) y la masa salarial
                # pública son políticamente inelásticos.
                def_t = def_t + 0.12 + np.random.normal(0, 0.4)
                def_t = max(def_t, 0.5)

                # Reservas: agotamiento acumulativo por producción
                r_t = r_t - q_t * 365 / 1e3
                r_t = max(r_t, 0.0)

                # Empleo: deterioro estructural vinculado a petróleo
                # La caída de ingresos fiscales reduce inversión pública
                # y contracción del sector formal.
                emp_t = emp_t - 0.15 + growth_year * 0.6 + np.random.normal(0, 1.3)
                emp_t = np.clip(emp_t, 15.0, 65.0)

                # Ratio R/P: indicador de horizonte de agotamiento
                annual_prod_mb = q_t * 365 / 1e3
                rp_t = r_t / annual_prod_mb if annual_prod_mb > 0 else 0.0

                # IRFC: Índice de Riesgo Fiscal Compuesto
                # Normalización calibrada a umbrales ecuatorianos.
                irfc_t = (
                    0.30 * min(d_t / 80.0, 1.0) +
                    0.25 * max(1.0 - rp_t / 12.0, 0.0) +
                    0.20 * min(max(def_t, 0) / 8.0, 1.0) +
                    0.15 * max(1.0 - emp_t / 50.0, 0.0) +
                    0.10 * max(1.0 - ing_t / 15.0, 0.0)
                ) * 100.0

                # --- Evaluación de colapso: 7 indicadores binarios ---
                score = 0
                if d_t > COLLAPSE_THRESHOLDS["deuda_pib_max"]:
                    score += 1
                if rp_t < COLLAPSE_THRESHOLDS["ratio_rp_min"]:
                    score += 1
                if def_t > COLLAPSE_THRESHOLDS["deficit_max"]:
                    score += 1
                if q_t < COLLAPSE_THRESHOLDS["produccion_min"]:
                    score += 1
                if irfc_t > COLLAPSE_THRESHOLDS["irfc_max"]:
                    score += 1
                if emp_t < COLLAPSE_THRESHOLDS["empleo_min"]:
                    score += 1
                if r_t < COLLAPSE_THRESHOLDS["reservas_min"]:
                    score += 1

                max_score = max(max_score, score)

                if score >= COLLAPSE_SCORE_THRESHOLD:
                    yearly_collapse_counts[j] += 1

            max_scores[i] = max_score
            collapse_flags[i] = max_score >= COLLAPSE_SCORE_THRESHOLD
            final_production[i] = q_t
            final_debt[i] = d_t
            final_reserves[i] = r_t

        # --- Resultados agregados ---
        self.collapse_probability = float(collapse_flags.mean())
        self.score_distribution = max_scores

        # Probabilidad por año
        for j, year in enumerate(years):
            self.yearly_collapse_prob[year] = float(
                yearly_collapse_counts[j] / self.n_simulations
            )

        # DataFrame de resultados
        self.results = pd.DataFrame({
            "delta": deltas,
            "price": prices,
            "growth": growths,
            "max_score": max_scores,
            "collapse": collapse_flags,
            "final_production": final_production,
            "final_debt": final_debt,
            "final_reserves": final_reserves,
        })

        # Semáforo fiscal
        self.traffic_light = self.compute_fiscal_traffic_light()

        logger.info(f"Simulación completada.")
        logger.info(f"  P(Colapso) = {self.collapse_probability:.1%}")
        logger.info(f"  Score medio = {max_scores.mean():.2f}")
        logger.info(f"  Semáforo = {self.traffic_light}")

        return self.results

    def compute_fiscal_traffic_light(self) -> str:
        """Determina el semáforo fiscal basado en P(colapso).

        Clasificación:
            - VERDE: P(colapso) < 30% → Riesgo bajo
            - AMARILLO: 30% ≤ P(colapso) < 60% → Riesgo moderado
            - ROJO: P(colapso) ≥ 60% → Riesgo crítico

        Returns:
            String con el color del semáforo.
        """
        if self.collapse_probability < 0.30:
            return "VERDE"
        elif self.collapse_probability < 0.60:
            return "AMARILLO"
        else:
            return "ROJO"

    def get_summary(self) -> Dict:
        """Retorna resumen completo de la simulación.

        Returns:
            Dict con estadísticas clave de la simulación.
        """
        if self.results is None:
            return {"error": "Simulación no ejecutada"}

        return {
            "n_simulations": self.n_simulations,
            "collapse_probability": round(self.collapse_probability * 100, 2),
            "traffic_light": self.traffic_light,
            "mean_score": round(float(self.score_distribution.mean()), 2),
            "median_score": round(float(np.median(self.score_distribution)), 2),
            "std_score": round(float(self.score_distribution.std()), 2),
            "mean_final_production": round(float(self.results["final_production"].mean()), 1),
            "mean_final_debt": round(float(self.results["final_debt"].mean()), 1),
            "mean_final_reserves": round(float(self.results["final_reserves"].mean()), 1),
            "yearly_collapse_prob": {
                str(k): round(v * 100, 2) for k, v in self.yearly_collapse_prob.items()
            },
            "score_distribution": {
                str(i): int((self.score_distribution == i).sum())
                for i in range(8)
            },
        }
