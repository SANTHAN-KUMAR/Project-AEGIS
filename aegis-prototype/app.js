/**
 * AEGIS 3.0 - Main Application Controller
 * Production-grade clinical decision support interface
 */

// ============================================
// Application State
// ============================================

const AppState = {
    currentSection: 'dashboard',
    simulation: {
        running: false,
        results: null
    },
    chart: null,
    config: {}
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    initializeChart();
    initializeSimulationConfig();
    initializeEventListeners();
    updateDashboardMetrics();
});

// ============================================
// Navigation
// ============================================

function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.dataset.section;
            navigateToSection(section);
        });
    });
}

function navigateToSection(sectionId) {
    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.section === sectionId);
    });

    // Update sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });

    const targetSection = document.getElementById(`${sectionId}-section`);
    if (targetSection) {
        targetSection.classList.add('active');
    }

    // Update header
    const titles = {
        dashboard: 'System Dashboard',
        simulation: 'Run Simulation',
        architecture: '5-Layer Architecture',
        validation: 'Validation Results'
    };

    document.getElementById('page-title').textContent = titles[sectionId] || 'Dashboard';
    AppState.currentSection = sectionId;
}

// ============================================
// Chart Initialization
// ============================================

function initializeChart() {
    const ctx = document.getElementById('glucose-chart');
    if (!ctx) return;

    AppState.chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Glucose (mg/dL)',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(15, 20, 25, 0.95)',
                    titleColor: '#f4f4f5',
                    bodyColor: '#9ca3af',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function (context) {
                            return `${context.parsed.y.toFixed(1)} mg/dL`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time (hours)',
                        color: '#6b7280',
                        font: { size: 11, weight: 500 }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7280',
                        font: { size: 10 },
                        maxTicksLimit: 12
                    }
                },
                y: {
                    display: true,
                    min: 40,
                    max: 300,
                    title: {
                        display: true,
                        text: 'Glucose (mg/dL)',
                        color: '#6b7280',
                        font: { size: 11, weight: 500 }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7280',
                        font: { size: 10 }
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        },
        plugins: [{
            id: 'targetRange',
            beforeDraw: (chart) => {
                const { ctx, chartArea, scales } = chart;
                if (!chartArea) return;

                // Draw target range (70-180)
                const yTop = scales.y.getPixelForValue(180);
                const yBottom = scales.y.getPixelForValue(70);

                ctx.save();
                ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
                ctx.fillRect(chartArea.left, yTop, chartArea.right - chartArea.left, yBottom - yTop);

                // Draw danger zone (<54)
                const yDanger = scales.y.getPixelForValue(54);
                ctx.fillStyle = 'rgba(239, 68, 68, 0.08)';
                ctx.fillRect(chartArea.left, yDanger, chartArea.right - chartArea.left, chartArea.bottom - yDanger);

                ctx.restore();
            }
        }]
    });
}

function updateChart(glucoseData) {
    if (!AppState.chart || !glucoseData) return;

    const labels = glucoseData.map(d => d.time.toFixed(1));
    const values = glucoseData.map(d => d.value);

    AppState.chart.data.labels = labels;
    AppState.chart.data.datasets[0].data = values;
    AppState.chart.update('none');
}

// ============================================
// Simulation Configuration
// ============================================

function initializeSimulationConfig() {
    // Meal protocol change handler
    const mealProtocol = document.getElementById('meal-protocol');
    const customMeals = document.getElementById('custom-meals');

    if (mealProtocol) {
        mealProtocol.addEventListener('change', (e) => {
            if (customMeals) {
                customMeals.classList.toggle('hidden', e.target.value !== 'custom');
            }
        });
    }

    // Range input handlers
    const rangeInputs = [
        { id: 'initial-glucose', valueId: 'initial-glucose-value', format: (v) => `${v} mg/dL` },
        { id: 'isf', valueId: 'isf-value', format: (v) => `${v} mg/dL per U` },
        { id: 'cr', valueId: 'cr-value', format: (v) => `1:${v} (g/U)` }
    ];

    rangeInputs.forEach(({ id, valueId, format }) => {
        const input = document.getElementById(id);
        const valueDisplay = document.getElementById(valueId);

        if (input && valueDisplay) {
            input.addEventListener('input', (e) => {
                valueDisplay.textContent = format(e.target.value);
            });
        }
    });
}

function getSimulationConfig() {
    return {
        patientProfile: document.getElementById('patient-profile')?.value || 'adult_avg',
        duration: parseInt(document.getElementById('sim-duration')?.value || 24),
        samplingInterval: parseInt(document.getElementById('sampling-interval')?.value || 5),
        mealProtocol: document.getElementById('meal-protocol')?.value || 'standard',
        initialGlucose: parseInt(document.getElementById('initial-glucose')?.value || 120),
        isf: parseInt(document.getElementById('isf')?.value || 50),
        cr: parseInt(document.getElementById('cr')?.value || 12),
        activityScenario: document.getElementById('activity-scenario')?.value || 'none',
        noiseModel: document.getElementById('noise-model')?.value || 'low',
        enableL1: document.getElementById('enable-l1')?.checked ?? true,
        enableL2: document.getElementById('enable-l2')?.checked ?? true,
        enableL3: document.getElementById('enable-l3')?.checked ?? true,
        enableL4: document.getElementById('enable-l4')?.checked ?? true,
        // Custom meals
        breakfastCarbs: parseInt(document.getElementById('breakfast-carbs')?.value || 45),
        breakfastTime: parseTimeToMinutes(document.getElementById('breakfast-time')?.value || '07:00'),
        lunchCarbs: parseInt(document.getElementById('lunch-carbs')?.value || 70),
        lunchTime: parseTimeToMinutes(document.getElementById('lunch-time')?.value || '12:00'),
        dinnerCarbs: parseInt(document.getElementById('dinner-carbs')?.value || 80),
        dinnerTime: parseTimeToMinutes(document.getElementById('dinner-time')?.value || '19:00')
    };
}

function parseTimeToMinutes(timeStr) {
    const [hours, minutes] = timeStr.split(':').map(Number);
    return hours * 60 + minutes;
}

// ============================================
// Event Listeners
// ============================================

function initializeEventListeners() {
    // Run simulation button
    const runBtn = document.getElementById('run-simulation');
    if (runBtn) {
        runBtn.addEventListener('click', runSimulation);
    }

    // Emergency stop
    const emergencyBtn = document.getElementById('emergency-stop');
    if (emergencyBtn) {
        emergencyBtn.addEventListener('click', emergencyStop);
    }
}

// ============================================
// Simulation Execution
// ============================================

async function runSimulation() {
    if (AppState.simulation.running) {
        showToast('Simulation already running', 'warning');
        return;
    }

    const config = getSimulationConfig();
    const simulator = new AEGISSimulator(config);

    // Update UI state
    AppState.simulation.running = true;
    const runBtn = document.getElementById('run-simulation');
    const statusEl = document.getElementById('sim-status');
    const resultsContent = document.getElementById('results-content');
    const logContainer = document.getElementById('execution-log');
    const logContent = document.getElementById('log-content');

    runBtn.disabled = true;
    runBtn.textContent = 'Running...';
    statusEl.textContent = 'Running';
    statusEl.className = 'sim-status running';
    logContainer.classList.remove('hidden');
    logContent.innerHTML = '';

    // Clear previous results
    resultsContent.innerHTML = '<div class="loading-state"><p>Executing simulation...</p></div>';

    try {
        // Progress callback
        const onProgress = (progress, layerStates) => {
            // Update layer status indicators on dashboard
            Object.entries(layerStates).forEach(([layer, state]) => {
                const statusEl = document.getElementById(`l${layer.slice(1)}-status`);
                if (statusEl) {
                    statusEl.textContent = state.status;
                    statusEl.className = `layer-status ${state.status === 'Verified' || state.status === 'Ready' ? 'safe' : 'active'}`;
                }
            });
        };

        // Log callback
        const onLog = (entry) => {
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry ${entry.type}`;

            const timeStr = entry.time > 0 ? `[${(entry.time / 60).toFixed(1)}h]` : '[0.0h]';
            logEntry.innerHTML = `<span class="log-time">${timeStr}</span> <span class="log-message">${entry.message}</span>`;

            logContent.appendChild(logEntry);
            logContent.scrollTop = logContent.scrollHeight;
        };

        // Run simulation
        const results = await simulator.run(onProgress, onLog);
        AppState.simulation.results = results;

        // Update dashboard
        updateDashboardFromResults(results);

        // Display results
        displayResults(results);

        statusEl.textContent = 'Complete';
        statusEl.className = 'sim-status complete';

        showToast('Simulation completed successfully', 'success');

    } catch (error) {
        console.error('Simulation error:', error);
        showToast('Simulation failed: ' + error.message, 'error');
        statusEl.textContent = 'Error';
        statusEl.className = 'sim-status';
    } finally {
        AppState.simulation.running = false;
        runBtn.disabled = false;
        runBtn.textContent = 'Run Simulation';
    }
}

function displayResults(results) {
    const content = document.getElementById('results-content');
    const metrics = results.metrics;

    content.innerHTML = `
        <div class="results-summary">
            <h4>Glycemic Outcomes</h4>
            <div class="results-grid">
                <div class="result-item">
                    <span class="result-label">Time in Range (70-180)</span>
                    <span class="result-value">${metrics.TIR}%</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Time Below 70</span>
                    <span class="result-value ${parseFloat(metrics.TBR_70) > 4 ? 'warning' : ''}">${metrics.TBR_70}%</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Time Below 54 (Severe)</span>
                    <span class="result-value ${parseFloat(metrics.TBR_54) > 0 ? 'danger' : 'success'}">${metrics.TBR_54}%</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Time Above 180</span>
                    <span class="result-value">${metrics.TAR_180}%</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Mean Glucose</span>
                    <span class="result-value">${metrics.mean} mg/dL</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Coefficient of Variation</span>
                    <span class="result-value">${metrics.CV}%</span>
                </div>
                <div class="result-item">
                    <span class="result-label">GMI (Est. A1C)</span>
                    <span class="result-value">${metrics.GMI}%</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Range (Min - Max)</span>
                    <span class="result-value">${metrics.min} - ${metrics.max}</span>
                </div>
            </div>
            
            <h4>Safety Metrics</h4>
            <div class="results-grid">
                <div class="result-item">
                    <span class="result-label">Severe Hypoglycemia Events</span>
                    <span class="result-value ${parseFloat(metrics.TBR_54) > 0 ? 'danger' : 'success'}">${parseFloat(metrics.TBR_54) > 0 ? 'Detected' : 'None'}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Seldonian Compliance</span>
                    <span class="result-value success">${metrics.seldonianCompliance}%</span>
                </div>
            </div>
        </div>
    `;

    // Add styles for results
    const style = document.createElement('style');
    style.textContent = `
        .results-summary h4 {
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 1rem;
            margin-top: 1.5rem;
        }
        .results-summary h4:first-child {
            margin-top: 0;
        }
        .results-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }
        .result-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background-color: var(--color-bg-tertiary);
            border-radius: var(--radius-md);
        }
        .result-label {
            font-size: 0.75rem;
            color: var(--color-text-secondary);
        }
        .result-value {
            font-size: 0.875rem;
            font-weight: 600;
            font-family: var(--font-mono);
        }
        .result-value.success {
            color: var(--color-success);
        }
        .result-value.warning {
            color: var(--color-warning);
        }
        .result-value.danger {
            color: var(--color-danger);
        }
        .loading-state {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 3rem;
            color: var(--color-text-muted);
        }
    `;

    if (!document.getElementById('results-styles')) {
        style.id = 'results-styles';
        document.head.appendChild(style);
    }
}

function updateDashboardFromResults(results) {
    // Update main metrics
    document.getElementById('current-glucose').textContent =
        results.glucoseTrace.length > 0
            ? results.glucoseTrace[results.glucoseTrace.length - 1].value.toFixed(0)
            : '--';

    document.getElementById('tir-value').textContent = results.metrics.TIR;
    document.getElementById('safety-score').textContent = results.metrics.seldonianCompliance;
    document.getElementById('hypo-value').textContent = results.metrics.TBR_54;

    // Update trend
    const trend = document.getElementById('glucose-trend');
    if (results.glucoseTrace.length >= 2) {
        const last = results.glucoseTrace[results.glucoseTrace.length - 1].value;
        const prev = results.glucoseTrace[results.glucoseTrace.length - 2].value;
        const change = last - prev;

        if (change > 2) {
            trend.querySelector('.trend-direction').textContent = 'Rising';
        } else if (change < -2) {
            trend.querySelector('.trend-direction').textContent = 'Falling';
        } else {
            trend.querySelector('.trend-direction').textContent = 'Stable';
        }
    }

    // Update chart
    updateChart(results.glucoseTrace);
}

function updateDashboardMetrics() {
    // Set initial placeholder values
    document.getElementById('current-glucose').textContent = '--';
    document.getElementById('tir-value').textContent = '--';
    document.getElementById('safety-score').textContent = '--';
    document.getElementById('hypo-value').textContent = '--';
}

// ============================================
// Emergency Stop
// ============================================

function emergencyStop() {
    if (AppState.simulation.running) {
        AppState.simulation.running = false;
        showToast('Emergency stop activated - simulation halted', 'warning');
    } else {
        showToast('Emergency stop activated - all insulin suspended', 'warning');
    }

    // Update layer 5 status
    const l5Status = document.getElementById('l5-status');
    if (l5Status) {
        l5Status.textContent = 'EMERGENCY';
        l5Status.className = 'layer-status';
        l5Status.style.backgroundColor = 'var(--color-danger-subtle)';
        l5Status.style.color = 'var(--color-danger)';
    }
}

// ============================================
// Toast Notifications
// ============================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span class="toast-message">${message}</span>`;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 200);
    }, 4000);
}
