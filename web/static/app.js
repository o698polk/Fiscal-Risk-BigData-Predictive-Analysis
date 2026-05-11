/**
 * Fiscal Risk Analytics — Frontend Application
 * ================================================
 * Controla la interacción del dashboard: ejecuta pasos del pipeline,
 * actualiza el log, renderiza KPIs, tablas, gráficos y el semáforo fiscal.
 *
 * Autor: Ing. Polk Brando Vernaza Quiñonez
 */

const API_BASE = '';

// ─── STATE ─────────────────────────────────────────────────────
const appState = {
    running: false,
    currentStep: null,
    completedSteps: new Set(),
};

// ─── STEP CONFIGURATION ───────────────────────────────────────
const STEPS = {
    etl:        { endpoint: '/api/run-etl',        label: 'ETL',          btnId: 'btnETL',        statusId: 'statusETL' },
    features:   { endpoint: '/api/run-features',   label: 'Features',     btnId: 'btnFeatures',   statusId: 'statusFeatures' },
    training:   { endpoint: '/api/run-training',   label: 'Entrenamiento',btnId: 'btnTraining',   statusId: 'statusTraining' },
    backtest:   { endpoint: '/api/run-backtest',    label: 'Backtesting',  btnId: 'btnBacktest',   statusId: 'statusBacktesting' },
    montecarlo: { endpoint: '/api/run-montecarlo', label: 'Monte Carlo',  btnId: 'btnMonteCarlo', statusId: 'statusMontecarlo' },
    scenarios:  { endpoint: '/api/run-scenarios',  label: 'Escenarios',   btnId: 'btnScenarios',  statusId: 'statusScenarios' },
};

// ─── LOGGING ──────────────────────────────────────────────────
function addLog(message, type = 'info') {
    const container = document.getElementById('logContainer');
    const panel = document.getElementById('logPanel');
    panel.style.display = 'block';

    const entry = document.createElement('div');
    entry.className = `log-entry log-entry--${type}`;

    const time = new Date().toLocaleTimeString('es-EC');
    entry.textContent = `[${time}] ${message}`;
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
}

// ─── STATUS UPDATES ───────────────────────────────────────────
function setHeaderStatus(text, state) {
    const el = document.getElementById('headerStatus');
    const dot = el.querySelector('.status-dot');
    dot.className = `status-dot status-dot--${state}`;
    el.querySelector('span:last-child').textContent = text;
}

function setStepState(stepKey, state) {
    const config = STEPS[stepKey];
    if (!config) return;
    const btn = document.getElementById(config.btnId);
    btn.classList.remove('completed', 'running', 'error');
    if (state) btn.classList.add(state);
}

function setAllButtonsDisabled(disabled) {
    document.getElementById('btnRunAll').disabled = disabled;
    Object.keys(STEPS).forEach(key => {
        document.getElementById(STEPS[key].btnId).disabled = disabled;
    });
}

// ─── API CALL ─────────────────────────────────────────────────
async function callAPI(endpoint) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    });
    const data = await response.json();
    if (!response.ok || !data.success) {
        throw new Error(data.error || `HTTP ${response.status}`);
    }
    return data;
}

// ─── RUN INDIVIDUAL STEP ──────────────────────────────────────
async function runStep(stepKey) {
    if (appState.running) return;

    const config = STEPS[stepKey];
    if (!config) return;

    appState.running = true;
    appState.currentStep = stepKey;
    setAllButtonsDisabled(true);
    setStepState(stepKey, 'running');
    setHeaderStatus(`Ejecutando ${config.label}...`, 'running');
    addLog(`Iniciando: ${config.label}...`, 'info');

    try {
        const result = await callAPI(config.endpoint);

        setStepState(stepKey, 'completed');
        appState.completedSteps.add(stepKey);
        addLog(`✓ ${config.label} completado: ${result.message}`, 'success');

        // Process results
        processStepResult(stepKey, result);

        setHeaderStatus('Paso completado', 'success');
    } catch (error) {
        setStepState(stepKey, 'error');
        addLog(`✗ Error en ${config.label}: ${error.message}`, 'error');
        setHeaderStatus('Error', 'error');
    } finally {
        appState.running = false;
        appState.currentStep = null;
        setAllButtonsDisabled(false);
    }
}

// ─── RUN ALL PIPELINE ─────────────────────────────────────────
async function runAll() {
    if (appState.running) return;

    appState.running = true;
    setAllButtonsDisabled(true);
    setHeaderStatus('Ejecutando pipeline completo...', 'running');
    addLog('🚀 Iniciando pipeline completo...', 'info');

    // Mark all as running
    Object.keys(STEPS).forEach(key => setStepState(key, 'running'));

    try {
        const result = await callAPI('/api/run-all');

        // Mark all completed
        Object.keys(STEPS).forEach(key => {
            setStepState(key, 'completed');
            appState.completedSteps.add(key);
        });

        addLog('✅ Pipeline completo ejecutado exitosamente.', 'success');
        setHeaderStatus('Pipeline completado', 'success');

        // Process all results
        processFullResults(result);

    } catch (error) {
        Object.keys(STEPS).forEach(key => {
            if (!appState.completedSteps.has(key)) setStepState(key, 'error');
        });
        addLog(`✗ Error en pipeline: ${error.message}`, 'error');
        setHeaderStatus('Error en pipeline', 'error');
    } finally {
        appState.running = false;
        setAllButtonsDisabled(false);
    }
}

// ─── PROCESS RESULTS ──────────────────────────────────────────
function processStepResult(stepKey, result) {
    const data = result.data;
    if (!data) return;

    switch (stepKey) {
        case 'etl':
            addLog(`  Dataset: ${data.shape[0]} filas × ${data.shape[1]} columnas`, 'info');
            addLog(`  Nulos: ${data.null_count}`, 'info');
            break;

        case 'features':
            renderKPI('irfc_mean', 'IRFC Medio', data.irfc_stats.mean.toFixed(1), 'Índice de Riesgo Fiscal', 'yellow');
            renderKPI('pca_var', 'Varianza PCA', data.pca_variance.cumulative + '%', `${data.new_columns.length} componentes`, 'purple');
            break;

        case 'training':
            renderMetricsTable(data.metrics);
            break;

        case 'backtest':
            renderMetricsTable(data.metrics, 'Backtesting');
            if (data.plot) renderPlots([data.plot]);
            break;

        case 'montecarlo':
            renderMonteCarlo(data);
            if (data.plot) renderPlots([data.plot]);
            break;

        case 'scenarios':
            renderScenarioTable(data.comparison);
            if (data.plots) renderPlots(data.plots);
            renderCollapseProbs(data.collapse_probs);
            break;
    }
}

function processFullResults(result) {
    const data = result.data;
    if (!data) return;

    // KPI Cards
    const kpiPanel = document.getElementById('kpiPanel');
    kpiPanel.style.display = 'block';
    document.getElementById('kpiGrid').innerHTML = '';

    renderKPI('etl_shape', 'Dataset Maestro', `${data.etl_shape[0]}×${data.etl_shape[1]}`, 'Trimestres × Variables', 'blue');
    if (data.irfc_mean != null) {
        renderKPI('irfc', 'IRFC Medio', data.irfc_mean.toFixed(1), 'Índice de Riesgo Fiscal', data.irfc_mean > 65 ? 'red' : data.irfc_mean > 40 ? 'yellow' : 'green');
    }
    renderKPI('collapse', 'P(Colapso)', data.collapse_probability + '%', 'Simulación Monte Carlo (N=50K)', data.collapse_probability > 60 ? 'red' : data.collapse_probability > 30 ? 'yellow' : 'green');
    renderKPI('light', 'Semáforo', data.traffic_light, 'Evaluación Fiscal', data.traffic_light === 'ROJO' ? 'red' : data.traffic_light === 'AMARILLO' ? 'yellow' : 'green');

    // Plots
    if (data.plots && data.plots.length > 0) {
        renderPlots(data.plots);
    }

    // Traffic light
    if (data.traffic_light) {
        renderTrafficLight(data.traffic_light, data.collapse_probability);
    }
}

// ─── RENDER FUNCTIONS ─────────────────────────────────────────
function renderKPI(id, label, value, sub, color = 'blue') {
    const panel = document.getElementById('kpiPanel');
    panel.style.display = 'block';
    const grid = document.getElementById('kpiGrid');

    // Remove existing KPI with same id if present
    const existing = document.getElementById(`kpi-${id}`);
    if (existing) existing.remove();

    const card = document.createElement('div');
    card.className = `kpi-card kpi-card--${color}`;
    card.id = `kpi-${id}`;
    card.innerHTML = `
        <span class="kpi-card__label">${label}</span>
        <span class="kpi-card__value">${value}</span>
        <span class="kpi-card__sub">${sub}</span>
    `;
    grid.appendChild(card);
}

function renderMetricsTable(metrics, title = 'Entrenamiento') {
    const panel = document.getElementById('metricsPanel');
    panel.style.display = 'block';
    const container = document.getElementById('metricsTableContainer');

    let html = `<table class="data-table">
        <thead><tr>
            <th>Modelo</th><th>MAE</th><th>RMSE</th><th>R²</th><th>Estado</th>
        </tr></thead><tbody>`;

    for (const [name, m] of Object.entries(metrics)) {
        if (m.error) continue;
        const r2 = m.r2 !== undefined ? parseFloat(m.r2) : null;
        const badge = r2 !== null
            ? (r2 > 0.9 ? 'green' : r2 > 0.7 ? 'yellow' : 'red')
            : 'yellow';
        const badgeText = r2 !== null
            ? (r2 > 0.9 ? 'Excelente' : r2 > 0.7 ? 'Bueno' : 'Mejorable')
            : 'N/A';

        html += `<tr>
            <td style="font-family:var(--font-sans);font-weight:600;">${name.replace(/_/g, ' ')}</td>
            <td>${m.mae !== undefined ? parseFloat(m.mae).toFixed(4) : '—'}</td>
            <td>${m.rmse !== undefined ? parseFloat(m.rmse).toFixed(4) : '—'}</td>
            <td>${r2 !== null ? r2.toFixed(4) : '—'}</td>
            <td><span class="badge badge--${badge}">${badgeText}</span></td>
        </tr>`;
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}

function renderScenarioTable(comparison) {
    if (!comparison || comparison.length === 0) return;

    const panel = document.getElementById('scenarioPanel');
    panel.style.display = 'block';
    const container = document.getElementById('scenarioTableContainer');

    const keys = Object.keys(comparison[0]).filter(k => k !== 'Escenario');

    let html = `<table class="data-table"><thead><tr><th>Escenario</th>`;
    keys.forEach(k => { html += `<th>${k}</th>`; });
    html += `</tr></thead><tbody>`;

    comparison.forEach(row => {
        const color = row['¿Colapso?'] === 'SÍ' ? 'color:var(--accent-red)' : 'color:var(--accent-green)';
        html += `<tr><td style="font-family:var(--font-sans);font-weight:600;">${row.Escenario}</td>`;
        keys.forEach(k => {
            const val = row[k];
            const style = k === '¿Colapso?' ? color : '';
            html += `<td style="${style}">${typeof val === 'number' ? val.toFixed(2) : val}</td>`;
        });
        html += `</tr>`;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

function renderMonteCarlo(data) {
    renderKPI('mc_prob', 'P(Colapso)', data.collapse_probability + '%', 'Monte Carlo N=' + data.n_simulations.toLocaleString(),
        data.collapse_probability > 60 ? 'red' : data.collapse_probability > 30 ? 'yellow' : 'green');
    renderKPI('mc_score', 'Score Medio', data.mean_score.toFixed(2), 'De 0 a 7 indicadores', 'yellow');
    renderKPI('mc_light', 'Semáforo', data.traffic_light, 'Evaluación Global',
        data.traffic_light === 'ROJO' ? 'red' : data.traffic_light === 'AMARILLO' ? 'yellow' : 'green');

    renderTrafficLight(data.traffic_light, data.collapse_probability);
}

function renderCollapseProbs(probs) {
    if (!probs) return;
    Object.entries(probs).forEach(([scenario, prob]) => {
        const pct = (prob * 100).toFixed(0);
        const color = prob > 0.6 ? 'red' : prob > 0.3 ? 'yellow' : 'green';
        renderKPI(`collapse_${scenario}`, `P(Col.) ${scenario.charAt(0).toUpperCase() + scenario.slice(1)}`, pct + '%', 'Escenario ' + scenario, color);
    });
}

function renderPlots(plotNames) {
    const panel = document.getElementById('plotsPanel');
    panel.style.display = 'block';
    const grid = document.getElementById('plotsGrid');
    const btnDl = document.getElementById('btnDownloadAll');
    btnDl.style.display = 'inline-flex';

    plotNames.forEach(name => {
        // Skip if already rendered
        if (document.getElementById(`plot-${name}`)) return;

        const card = document.createElement('div');
        card.className = 'plot-card';
        card.id = `plot-${name}`;

        const prettyName = name.replace('.png', '').replace(/_/g, ' ').replace(/^\d+\s*/, '');

        card.innerHTML = `
            <img src="/api/plots/${name}" alt="${prettyName}" class="plot-card__image"
                 loading="lazy" onclick="openLightbox(this.src)">
            <div class="plot-card__footer">
                <span class="plot-card__name">${prettyName}</span>
                <button class="plot-card__download" onclick="downloadPlot('${name}')">⬇ PNG</button>
            </div>
        `;
        grid.appendChild(card);
    });
}

function renderTrafficLight(light, probability) {
    const panel = document.getElementById('trafficPanel');
    panel.style.display = 'block';
    const container = document.getElementById('trafficLightContainer');

    const isRed = light === 'ROJO';
    const isYellow = light === 'AMARILLO';
    const isGreen = light === 'VERDE';

    const descriptions = {
        ROJO: 'Riesgo fiscal CRÍTICO. La probabilidad de colapso supera el 60%. Se requieren reformas estructurales urgentes para evitar la insolvencia del Estado antes de 2032.',
        AMARILLO: 'Riesgo fiscal MODERADO. La probabilidad de colapso se sitúa entre 30% y 60%. Se recomienda implementar medidas preventivas de diversificación fiscal.',
        VERDE: 'Riesgo fiscal BAJO. La probabilidad de colapso es inferior al 30%. Las finanzas públicas se mantienen dentro de márgenes sostenibles.',
    };

    const colorClass = light.toLowerCase();

    container.innerHTML = `
        <div class="traffic-light">
            <div class="traffic-bulb traffic-bulb--red ${isRed ? 'active' : ''}"></div>
            <div class="traffic-bulb traffic-bulb--yellow ${isYellow ? 'active' : ''}"></div>
            <div class="traffic-bulb traffic-bulb--green ${isGreen ? 'active' : ''}"></div>
        </div>
        <div class="traffic-label traffic-label--${colorClass}">${light}</div>
        <p class="traffic-description">${descriptions[light] || ''}</p>
        <p class="traffic-description" style="font-weight:600; color:var(--text-heading);">
            Probabilidad de Colapso Fiscal: ${probability}%
        </p>
    `;
}

// ─── LIGHTBOX ─────────────────────────────────────────────────
function openLightbox(src) {
    const lb = document.createElement('div');
    lb.className = 'lightbox';
    lb.onclick = () => lb.remove();
    lb.innerHTML = `<img src="${src}" alt="Gráfico ampliado">`;
    document.body.appendChild(lb);
}

// ─── DOWNLOAD HELPERS ─────────────────────────────────────────
function downloadPlot(filename) {
    const a = document.createElement('a');
    a.href = `/api/plots/${filename}`;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function downloadAllPlots() {
    const images = document.querySelectorAll('.plot-card__image');
    images.forEach(img => {
        const filename = img.src.split('/').pop();
        downloadPlot(filename);
    });
}

// ─── INITIAL STATUS CHECK ─────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.success) {
            Object.entries(data.status).forEach(([key, done]) => {
                if (done) {
                    setStepState(key, 'completed');
                    appState.completedSteps.add(key);
                }
            });
        }
    } catch (e) {
        // Server not running, ignore
    }
});
