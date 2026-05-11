# -*- coding: utf-8 -*-
"""
Módulo de Visualización: FiscalVisualizer.
============================================

Genera gráficos de alta calidad para todos los componentes del análisis,
separando estrictamente la lógica de presentación de la lógica de negocio
(Single Responsibility Principle).

Todos los gráficos se exportan como PNG de alta resolución (300 DPI)
al directorio output/plots/.

Referencia:
    Vernaza Quiñonez, P.B. (2025). Los gráficos reproducen las figuras
    principales del artículo: evolución IRFC, clústeres PCA, proyecciones
    de producción, distribución Monte Carlo y comparación de escenarios.

Autor: Ing. Polk Brando Vernaza Quiñonez
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Backend no-interactivo para servidor web
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PLOTS_DIR

logger = logging.getLogger(__name__)

# Estilo global profesional
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.figsize": (12, 7),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.family": "sans-serif",
})

# Paleta de colores consistente con la investigación
COLORS = {
    "primary": "#1a73e8",
    "danger": "#dc3545",
    "warning": "#ffc107",
    "success": "#28a745",
    "info": "#17a2b8",
    "dark": "#343a40",
    "scenarios": {
        "optimista": "#28a745",
        "base": "#ffc107",
        "pesimista": "#dc3545",
    },
}


class FiscalVisualizer:
    """Generador de visualizaciones para el análisis de riesgo fiscal.

    Cada método genera un gráfico específico, lo guarda como PNG y
    retorna la ruta del archivo generado.

    Attributes:
        output_dir: Directorio donde se guardan los gráficos.
        generated_plots: Lista de rutas de gráficos generados.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Inicializa el visualizador y crea el directorio de salida.

        Args:
            output_dir: Directorio de salida para gráficos.
        """
        self.output_dir = output_dir or PLOTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_plots: List[str] = []
        logger.info(f"FiscalVisualizer: output → {self.output_dir}")

    def _save_plot(self, fig: plt.Figure, filename: str) -> str:
        """Guarda un gráfico y registra la ruta.

        Args:
            fig: Figura de matplotlib.
            filename: Nombre del archivo (sin extensión).

        Returns:
            Ruta absoluta del archivo guardado.
        """
        path = self.output_dir / f"{filename}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.generated_plots.append(str(path))
        logger.info(f"Gráfico guardado: {path}")
        return str(path)

    def plot_irfc_evolution(self, df: pd.DataFrame) -> str:
        """Evolución temporal del IRFC con zonas de semáforo fiscal.

        Muestra la serie temporal del Índice de Riesgo Fiscal Compuesto
        con bandas de color indicando niveles de riesgo.

        Args:
            df: DataFrame con columnas 'year' e 'irfc'.

        Returns:
            Ruta del archivo PNG generado.
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Zonas de semáforo
        ax.axhspan(0, 35, alpha=0.1, color=COLORS["success"], label="Bajo Riesgo")
        ax.axhspan(35, 65, alpha=0.1, color=COLORS["warning"], label="Riesgo Moderado")
        ax.axhspan(65, 100, alpha=0.1, color=COLORS["danger"], label="Riesgo Crítico")

        # Serie IRFC
        if "year" in df.columns:
            x = df["year"]
        else:
            x = range(len(df))

        ax.plot(x, df["irfc"], color=COLORS["primary"], linewidth=2.5,
                marker="o", markersize=3, label="IRFC")

        # Línea de umbral de colapso
        ax.axhline(y=75, color=COLORS["danger"], linestyle="--",
                   linewidth=1.5, alpha=0.7, label="Umbral Colapso (75)")

        ax.set_xlabel("Año / Periodo")
        ax.set_ylabel("IRFC (0–100)")
        ax.set_title("Evolución del Índice de Riesgo Fiscal Compuesto (IRFC)\nEcuador — Serie Histórica",
                     fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        return self._save_plot(fig, "01_irfc_evolution")

    def plot_pca_clusters(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        regime_names: Dict[int, str],
    ) -> str:
        """Scatter plot de PCA 2D coloreado por régimen K-Means.

        Args:
            df: DataFrame con columnas PC1 y PC2.
            labels: Array de etiquetas de clúster.
            regime_names: Dict id → nombre del régimen.

        Returns:
            Ruta del archivo PNG.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        palette = sns.color_palette("husl", n_colors=len(set(labels)))

        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            name = regime_names.get(cluster_id, f"Clúster {cluster_id}")
            ax.scatter(
                df.loc[mask, "PC1"] if "PC1" in df.columns else range(mask.sum()),
                df.loc[mask, "PC2"] if "PC2" in df.columns else range(mask.sum()),
                c=[palette[cluster_id]],
                label=name,
                s=60,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xlabel("PC1 — Eje Petróleo-PIB")
        ax.set_ylabel("PC2 — Eje Deuda-Déficit")
        ax.set_title("Regímenes Económicos de Ecuador (K-Means sobre PCA)\n7 Clústeres Históricos 1990–2024",
                     fontweight="bold")
        ax.legend(loc="best", framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)

        return self._save_plot(fig, "02_pca_clusters")

    def plot_production_forecast(
        self,
        historical: pd.DataFrame,
        scenarios: Dict[str, pd.DataFrame],
    ) -> str:
        """Proyección de producción petrolera con escenarios.

        Args:
            historical: DataFrame histórico con 'year' y 'produccion_diaria'.
            scenarios: Dict escenario → DataFrame con proyecciones.

        Returns:
            Ruta del archivo PNG.
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Histórico
        if "year" in historical.columns:
            ax.plot(
                historical["year"], historical["produccion_diaria"],
                color=COLORS["dark"], linewidth=2, marker="s", markersize=3,
                label="Histórico (1990–2024)"
            )

        # Escenarios
        for name, df_sc in scenarios.items():
            color = COLORS["scenarios"].get(name, COLORS["info"])
            ax.plot(
                df_sc["year"], df_sc["produccion_diaria"],
                color=color, linewidth=2, linestyle="--",
                marker="o", markersize=4,
                label=f"Escenario {name.capitalize()}"
            )

        # Línea de umbral
        ax.axhline(y=300, color=COLORS["danger"], linestyle=":",
                   alpha=0.6, label="Umbral Crítico (300 Kb/d)")

        ax.set_xlabel("Año")
        ax.set_ylabel("Producción Diaria (Kb/d)")
        ax.set_title("Proyección de Producción Petrolera — Ecuador\nHistórico + Escenarios 2025–2032",
                     fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        return self._save_plot(fig, "03_production_forecast")

    def plot_monte_carlo_distribution(
        self,
        scores: np.ndarray,
        collapse_prob: float,
    ) -> str:
        """Distribución de scores de colapso fiscal (Monte Carlo).

        Args:
            scores: Array con scores máximos de cada simulación.
            collapse_prob: Probabilidad de colapso estimada.

        Returns:
            Ruta del archivo PNG.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Panel 1: Histograma de scores
        ax1 = axes[0]
        unique_scores = sorted(set(scores.astype(int)))
        counts = [int((scores.astype(int) == s).sum()) for s in unique_scores]
        colors = [
            COLORS["success"] if s < 3 else
            COLORS["warning"] if s < 5 else
            COLORS["danger"]
            for s in unique_scores
        ]
        ax1.bar(unique_scores, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax1.axvline(x=4.5, color=COLORS["danger"], linestyle="--",
                    linewidth=2, label="Umbral Colapso (≥5)")
        ax1.set_xlabel("Score de Colapso (0–7)")
        ax1.set_ylabel("Frecuencia")
        ax1.set_title("Distribución de Scores — Monte Carlo\n(N=50,000 Simulaciones)",
                      fontweight="bold")
        ax1.legend()

        # Panel 2: Gauge de probabilidad
        ax2 = axes[1]
        sizes = [collapse_prob * 100, (1 - collapse_prob) * 100]
        colors_pie = [COLORS["danger"], COLORS["success"]]
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=["Colapso", "Estable"], colors=colors_pie,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 14, "fontweight": "bold"},
        )
        ax2.set_title(f"Probabilidad de Colapso Fiscal\nP = {collapse_prob:.1%}",
                      fontweight="bold", fontsize=14)

        plt.tight_layout()
        return self._save_plot(fig, "04_monte_carlo_distribution")

    def plot_scenario_comparison(
        self,
        scenarios: Dict[str, pd.DataFrame],
    ) -> str:
        """Panel comparativo de escenarios: 4 indicadores clave.

        Args:
            scenarios: Dict escenario → DataFrame de proyecciones.

        Returns:
            Ruta del archivo PNG.
        """
        indicators = [
            ("produccion_diaria", "Producción (Kb/d)"),
            ("deuda_pib", "Deuda/PIB (%)"),
            ("reservas_probadas", "Reservas (Mb)"),
            ("irfc", "IRFC"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for idx, (col, title) in enumerate(indicators):
            ax = axes[idx // 2][idx % 2]

            for name, df_sc in scenarios.items():
                color = COLORS["scenarios"].get(name, COLORS["info"])
                ax.plot(
                    df_sc["year"], df_sc[col],
                    color=color, linewidth=2, marker="o", markersize=4,
                    label=name.capitalize()
                )

            ax.set_xlabel("Año")
            ax.set_ylabel(title)
            ax.set_title(title, fontweight="bold")
            ax.legend(framealpha=0.9)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Comparación de Escenarios Fiscales — Ecuador 2025–2032",
                     fontweight="bold", fontsize=15, y=1.02)
        plt.tight_layout()
        return self._save_plot(fig, "05_scenario_comparison")

    def plot_backtesting_results(
        self,
        predictions: Dict[str, Dict[str, np.ndarray]],
        years_test: Optional[np.ndarray] = None,
    ) -> str:
        """Resultados de backtesting: Predicho vs Real por modelo.

        Args:
            predictions: Dict modelo → {y_true, y_pred}.
            years_test: Array con los años del periodo de prueba.

        Returns:
            Ruta del archivo PNG.
        """
        n_models = len(predictions)
        if n_models == 0:
            logger.warning("No hay predicciones de backtesting para graficar.")
            return ""

        cols = min(n_models, 2)
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (name, data) in enumerate(predictions.items()):
            ax = axes[idx]
            y_true = data["y_true"]
            y_pred = data["y_pred"]
            x = years_test if years_test is not None else range(len(y_true))

            ax.plot(x, y_true, "o-", color=COLORS["primary"],
                    label="Real", linewidth=2, markersize=5)
            ax.plot(x, y_pred, "s--", color=COLORS["danger"],
                    label="Predicho", linewidth=2, markersize=5)

            # Banda de error
            ax.fill_between(
                x, y_true, y_pred, alpha=0.15, color=COLORS["danger"]
            )

            ax.set_title(name.replace("_", " "), fontweight="bold")
            ax.set_xlabel("Periodo")
            ax.legend(framealpha=0.9)
            ax.grid(True, alpha=0.3)

        # Ocultar ejes vacíos
        for idx in range(len(predictions), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Validación Retrospectiva (Backtesting)\nTrain ≤2018 vs Test ≥2019",
                     fontweight="bold", fontsize=14)
        plt.tight_layout()
        return self._save_plot(fig, "06_backtesting_results")

    def plot_debt_trajectory(self, scenarios: Dict[str, pd.DataFrame]) -> str:
        """Trayectoria de Deuda/PIB por escenario con umbrales.

        Args:
            scenarios: Dict escenario → DataFrame.

        Returns:
            Ruta del archivo PNG.
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        ax.axhspan(0, 50, alpha=0.05, color=COLORS["success"])
        ax.axhspan(50, 80, alpha=0.05, color=COLORS["warning"])
        ax.axhspan(80, 200, alpha=0.05, color=COLORS["danger"])
        ax.axhline(y=80, color=COLORS["danger"], linestyle="--",
                   alpha=0.6, label="Umbral Crítico (80%)")

        for name, df_sc in scenarios.items():
            color = COLORS["scenarios"].get(name, COLORS["info"])
            ax.plot(df_sc["year"], df_sc["deuda_pib"],
                    color=color, linewidth=2.5, marker="o", markersize=5,
                    label=f"{name.capitalize()}")

        ax.set_xlabel("Año")
        ax.set_ylabel("Deuda/PIB (%)")
        ax.set_title("Trayectoria de Deuda Pública / PIB — Ecuador\nProyección 2025–2032",
                     fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        return self._save_plot(fig, "07_debt_trajectory")

    def get_all_plots(self) -> List[str]:
        """Retorna todas las rutas de gráficos generados.

        Returns:
            Lista de paths absolutos de archivos PNG.
        """
        return self.generated_plots
