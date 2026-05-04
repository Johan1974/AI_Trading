window.ChartUtils = {
    registry: {},
    CHART_GRID_DARK: "#222222",
    CHART_AXIS_WHITE: "#FFFFFF",

    initDefaults() {
        if (typeof Chart === "undefined" || !Chart.defaults) return;
        Chart.defaults.color = this.CHART_AXIS_WHITE;
        Chart.defaults.font = Chart.defaults.font || {};
        Chart.defaults.font.family = "'JetBrains Mono', ui-monospace, monospace";
        Chart.defaults.font.size = 14;
        Chart.defaults.font.weight = "700";
    },

    mergeContrastScale(scale = {}) {
        const prevTicks = scale.ticks || {};
        return {
            ...scale,
            ticks: {
                ...prevTicks,
                color: this.CHART_AXIS_WHITE,
                font: { size: 14, family: "'JetBrains Mono', ui-monospace, monospace", weight: "700" },
            },
            grid: { color: scale.grid?.color || this.CHART_GRID_DARK },
            border: { color: scale.border?.color || this.CHART_AXIS_WHITE, display: true },
        };
    },

    highContrastChartOptions(base = {}) {
        const options = { ...base };
        const existingPlugins = options.plugins || {};
        options.plugins = {
            ...existingPlugins,
            legend: {
                ...(existingPlugins.legend || {}),
                labels: { color: this.CHART_AXIS_WHITE, font: { size: 14, weight: "700", family: "'JetBrains Mono', ui-monospace, monospace" } }
            },
            tooltip: {
                ...(existingPlugins.tooltip || {}),
                backgroundColor: "#000000",
                titleColor: this.CHART_AXIS_WHITE,
                bodyColor: this.CHART_AXIS_WHITE,
                borderColor: this.CHART_AXIS_WHITE,
                borderWidth: 1,
            },
        };

        options.scales = options.scales ? { ...options.scales } : {};
        const scaleKeys = Object.keys(options.scales);
        if (!scaleKeys.length) {
            options.scales.x = this.mergeContrastScale({ ticks: { maxRotation: 0 } });
            options.scales.y = this.mergeContrastScale({});
        } else {
            for (const key of scaleKeys) {
                const prev = options.scales[key] || {};
                options.scales[key] = this.mergeContrastScale({ ...prev });
            }
        }
        return options;
    },

    upsertChart(canvasId, config) {
        const el = document.getElementById(canvasId);
        if (!el || el.offsetParent === null) return this.registry[canvasId] || null;

        const normalizedConfig = {
            ...config,
            options: this.highContrastChartOptions(config.options || {}),
        };

        let current = this.registry[canvasId];
        if (!current) {
            current = new Chart(el.getContext("2d"), normalizedConfig);
            this.registry[canvasId] = current;
            return current;
        }
        current.data = normalizedConfig.data;
        current.options = normalizedConfig.options;
        current.update();
        return current;
    },

    clearAllCharts() {
        Object.values(this.registry).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') chart.destroy();
        });
        this.registry = {};
    }
};
window.ChartUtils.initDefaults();