# -*- coding: utf-8 -*-
"""
Generador de Escenarios: ScenarioGenerator.
=============================================

Genera proyecciones fiscales para 2025–2032 bajo tres escenarios
(Optimista, Base, Pesimista) y calcula la probabilidad de colapso
para cada uno usando los modelos entrenados.

Referencia:
    Vernaza Quiñonez, P.B. (2025). Tabla 12 – "Los tres escenarios
    prospectivos permiten cuantificar la sensibilidad del sistema fiscal
    a variaciones en el precio del crudo, crecimiento económico y tasa
    de declive petrolero."

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import (
    PROJECTION_END_YEAR,
    PROJECTION_START_YEAR,
    SCENARIOS,
    COLLAPSE_THRESHOLDS,
    COLLAPSE_SCORE_THRESHOLD,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """Generador de proyecciones fiscales bajo múltiples escenarios.

    Proyecta las variables macroeconómicas clave desde 2025 hasta 2032
    usando los supuestos definidos en cada escenario (precio crudo,
    crecimiento PIB, declive de producción).

    Attributes:
        scenarios: Dict con definición de cada escenario.
        projections: Dict escenario → DataFrame de proyecciones.
        collapse_probs: Dict escenario → probabilidad de colapso.

    Example:
        >>> generator = ScenarioGenerator()
        >>> projections = generator.generate_projections(initial_conditions)
        >>> generator.export_to_csv()
    """

    def __init__(self, scenarios: Optional[Dict] = None) -> None:
        """Inicializa con los escenarios definidos en configuración.

        Args:
            scenarios: Dict de escenarios. Si None, usa los predeterminados.
        """
        self.scenarios = scenarios or SCENARIOS
        self.projections: Dict[str, pd.DataFrame] = {}
        self.collapse_probs: Dict[str, float] = {}
        logger.info(
            f"ScenarioGenerator inicializado con {len(self.scenarios)} escenarios: "
            f"{list(self.scenarios.keys())}"
        )

    def generate_projections(
        self,
        initial_conditions: Dict[str, float],
    ) -> Dict[str, pd.DataFrame]:
        """Genera proyecciones determinísticas para cada escenario.

        Para cada escenario, proyecta año a año usando las ecuaciones:
            - Producción: q(t+1) = q(t) · (1 + declive)
            - Deuda/PIB: d(t+1) = d(t) · (1+r)/(1+g) + déficit
            - Reservas: R(t+1) = R(t) - q(t)·365/1000
            - Ingresos: f(precio, producción)
            - IRFC: índice compuesto recalculado

        Args:
            initial_conditions: Dict con valores base 2024.

        Returns:
            Dict escenario → DataFrame con proyecciones año a año.
        """
        logger.info("Generando proyecciones 2025–2032...")

        years = list(range(PROJECTION_START_YEAR, PROJECTION_END_YEAR + 1))

        for scenario_name, params in self.scenarios.items():
            logger.info(f"\n--- Escenario: {scenario_name.upper()} ---")

            records = []
            q_t = initial_conditions.get("produccion_diaria", 470.0)
            d_t = initial_conditions.get("deuda_pib", 63.0)
            r_t = initial_conditions.get("reservas_probadas", 1380.0)
            def_t = initial_conditions.get("deficit_primario", 4.0)
            emp_t = initial_conditions.get("empleo_adecuado", 38.0)
            ing_t = initial_conditions.get("ingresos_pge", 11.0)
            iva_t = initial_conditions.get("recaudacion_iva", 8300.0)

            precio = params["precio_crudo"]
            growth = params["crecimiento_pib"]
            decline = params["declive_produccion"]
            def_trend = params["deficit_tendencia"]

            for year in years:
                # Producción petrolera: declive anual
                q_t = q_t * (1.0 + decline)
                q_t = max(q_t, 50.0)

                # Reservas: producción anual acumulada
                r_t = r_t - q_t * 365 / 1000
                r_t = max(r_t, 0.0)

                # Ratio R/P
                rp_t = r_t / (q_t * 365 / 1000) if q_t > 0 else 0.0

                # Deuda/PIB
                r_interest = 0.05
                d_t = d_t * (1 + r_interest) / (1 + growth / 100.0) + def_t * 0.5
                d_t = max(d_t, 10.0)

                # Déficit
                def_t = def_t + def_trend

                # Ingresos PGE
                ing_t = 11.0 * (precio / 65.0) * (q_t / 470.0) * 0.85
                ing_t = np.clip(ing_t, 2.0, 25.0)

                # Recaudación IVA
                iva_t = 8300 * (1 + growth / 100.0)

                # Empleo
                emp_t = 38.0 + growth * 2.0

                # Exportaciones petroleras
                exp_t = q_t * precio * 365 / 1e3 * 0.65

                # Inversión pública
                inv_t = 14.3 * (ing_t / 11.0)

                # IRFC simplificado
                irfc_t = (
                    0.30 * min(d_t / 100.0, 1.0)
                    + 0.25 * max(1.0 - rp_t / 15.0, 0.0)
                    + 0.20 * min(max(def_t, 0) / 10.0, 1.0)
                    + 0.15 * max(1.0 - emp_t / 60.0, 0.0)
                    + 0.10 * max(1.0 - ing_t / 20.0, 0.0)
                ) * 100.0

                # Score de colapso
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

                records.append({
                    "year": year,
                    "produccion_diaria": round(q_t, 2),
                    "precio_crudo": precio,
                    "crecimiento_pib": growth,
                    "deuda_pib": round(d_t, 2),
                    "reservas_probadas": round(r_t, 2),
                    "ingresos_pge": round(ing_t, 2),
                    "recaudacion_iva": round(iva_t, 2),
                    "deficit_primario": round(def_t, 2),
                    "empleo_adecuado": round(emp_t, 2),
                    "ratio_rp": round(rp_t, 2),
                    "exportaciones_petroleras": round(exp_t, 2),
                    "inversion_publica": round(inv_t, 2),
                    "irfc": round(irfc_t, 2),
                    "collapse_score": score,
                    "collapse": score >= COLLAPSE_SCORE_THRESHOLD,
                })

            df_scenario = pd.DataFrame(records)
            self.projections[scenario_name] = df_scenario

            # Probabilidad de colapso = % de años con score ≥ 5
            self.collapse_probs[scenario_name] = float(df_scenario["collapse"].mean())

            logger.info(
                f"  {scenario_name}: Prod 2032={df_scenario.iloc[-1]['produccion_diaria']:.0f} Kb/d, "
                f"Deuda/PIB={df_scenario.iloc[-1]['deuda_pib']:.1f}%, "
                f"IRFC={df_scenario.iloc[-1]['irfc']:.1f}, "
                f"P(colapso)={self.collapse_probs[scenario_name]:.0%}"
            )

        return self.projections

    def export_to_csv(self, output_dir=None) -> Dict[str, str]:
        """Exporta las proyecciones de cada escenario a archivos CSV.

        Args:
            output_dir: Directorio de salida (default: data/processed/).

        Returns:
            Dict escenario → ruta del archivo exportado.
        """
        out_dir = output_dir or PROCESSED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        for name, df in self.projections.items():
            path = out_dir / f"scenario_{name}_2025_2032.csv"
            df.to_csv(path, index=False)
            paths[name] = str(path)
            logger.info(f"Escenario '{name}' exportado: {path}")

        return paths

    def get_comparison_table(self) -> pd.DataFrame:
        """Genera tabla comparativa de los tres escenarios al año 2032.

        Returns:
            DataFrame con métricas clave por escenario en el año final.
        """
        rows = []
        for name, df in self.projections.items():
            last = df.iloc[-1]
            rows.append({
                "Escenario": name.capitalize(),
                "Producción 2032 (Kb/d)": last["produccion_diaria"],
                "Deuda/PIB 2032 (%)": last["deuda_pib"],
                "Reservas 2032 (Mb)": last["reservas_probadas"],
                "IRFC 2032": last["irfc"],
                "Score Colapso": last["collapse_score"],
                "¿Colapso?": "SÍ" if last["collapse"] else "NO",
            })
        return pd.DataFrame(rows).set_index("Escenario")
