# 📉 Fiscal-Risk-BigData-Predictive-Analysis (2025–2032)
> **Ecosistema de Machine Learning para el Modelado de Sostenibilidad y Riesgo de Colapso Fiscal en Ecuador.**

---

## 🎯 Propósito del Proyecto
Este proyecto desarrolla un ecosistema predictivo de alta fidelidad diseñado para modelar la trayectoria de las finanzas públicas ecuatorianas en el horizonte **2025–2032**. El objetivo primordial es cuantificar la probabilidad de un colapso fiscal estructural mediante la integración de modelos de Machine Learning, simulaciones estocásticas y análisis de Big Data.

## ⚠️ El Problema
Ecuador enfrenta una "trampa de sostenibilidad" caracterizada por:
1. **Dependencia Petrolera en Declive:** El agotamiento natural de los campos maduros (ITT, Sacha, Shushufindi) reduce drásticamente los ingresos no tributarios.
2. **Rigidez del Gasto:** Un presupuesto estatal altamente inflexible (masa salarial, subsidios y servicio de deuda).
3. **Restricción Externa:** La falta de política monetaria propia (dolarización) impide la monetización del déficit, forzando un endeudamiento a tasas de mercado agresivas.
4. **Volatilidad Exógena:** La alta sensibilidad a los precios internacionales del crudo y a los choques de crecimiento global.

## ⚖️ Justificación
La investigación del **Ing. Polk Brando Vernaza Quiñonez** (2025) demuestra que los modelos macroeconómicos tradicionales fallan al capturar las correlaciones no lineales entre el declive geológico y la acumulación de deuda. Este sistema justifica su implementación al ofrecer:
- **Precisión Predictiva:** Uso de algoritmos de Ensemble (Random Forest, Gradient Boosting) para superar las limitaciones de las regresiones lineales.
- **Gestión de Incertidumbre:** Empleo de Monte Carlo (N=50,000) para generar bandas de confianza realistas.
- **Alerta Temprana:** Identificación de la "Ventana de Intervención" (2025-2027) antes de que la trayectoria de deuda sea irreversible.

---

## 📊 Fuentes de Datos
El sistema utiliza un dataset híbrido multifrecuencia armonizado mediante técnicas de desagregación temporal (Chow-Lin).

| Institución | Tipo de Datos | Enlace Oficial |
| :--- | :--- | :--- |
| **Banco Central del Ecuador (BCE)** | PIB, Inflación, Balanza de Pagos | [BCE Estadísticas](https://www.bce.fin.ec/) |
| **EP Petroecuador** | Producción, Reservas, Exportaciones | [Petroecuador Reportes](https://www.eppetroecuador.ec/) |
| **Min. de Economía y Finanzas (MEF)** | Deuda Pública, Déficit, PGE | [MEF Deuda](https://www.finanzas.gob.ec/) |
| **INEC** | Empleo, Desempleo, Índices Sociales | [INEC Consultas](https://www.ecuadorencifras.gob.ec/) |

> [!NOTE]
> Los datos han sido normalizados y transformados para asegurar la consistencia entre series anuales (1990-2024) y trimestrales (2005-2024).

---

## 🏗️ Arquitectura del Sistema
El sistema sigue los principios **SOLID** y una arquitectura modular desacoplada:

1. **Módulo ETL (`FiscalETL`):** Limpieza, normalización y aplicación del algoritmo Chow-Lin.
2. **Feature Architect:** Cálculo del **Índice de Riesgo Fiscal Compuesto (IRFC)** y reducción de dimensionalidad vía **PCA**.
3. **Model Ecosystem:** Orquestador de 8 modelos ML (RF + Arps, GB, SVR, K-Means).
4. **Monte Carlo Engine:** Simulación de 50,000 escenarios estocásticos.
5. **Scenario Generator:** Proyección prospectiva bajo 3 niveles de estrés macroeconómico.
6. **API & Dashboard:** Interfaz web interactiva en Flask para gestión de resultados.

---

## 🚀 Guía de Ejecución

### 1. Requisitos Previos
- Python 3.12 o superior.
- Git.

### 2. Instalación
```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/Fiscal-Risk-BigData-Predictive-Analysis.git

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Ejecución del Pipeline (Consola)
```bash
python main.py
```

### 4. Lanzamiento del Dashboard Web
```bash
python app.py
```
Accede a `http://localhost:5000` para interactuar con los gráficos y generar predicciones en tiempo real.

---

## 📈 Resultados Principales
El sistema ha sido calibrado para reflejar la realidad fiscal actual:
- **P(Colapso Fiscal): 62.8%** (Horizonte 2032).
- **Semáforo Fiscal:** 🔴 **ROJO** (Riesgo Crítico).
- **Principal Driver de Riesgo:** El ratio de Deuda/PIB superando el umbral del 70% en el 94% de los escenarios de estrés.

---

## ✍️ Autor & Contacto
**Ing. Polk Brando Vernaza Quiñonez**
- *Investigador en Economía Digital y Análisis de Datos*
- *Instituto Superior Tecnológico Alberto Enríquez (ISTAE)*
- **ORCID:** [0009-0009-1674-6492](https://orcid.org/0009-0009-1674-6492)
- **Contacto:** pvernaza@istae.edu.ec

---
## ⚖️ Licencia
Creative Commons Atribución-NoComercial-SinDerivadas 4.0 Internacional (CC BY-NC-ND 4.0).
