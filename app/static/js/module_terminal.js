
window.ModuleTerminal = {
    chart: null,
    series: null,
    /** LightweightCharts-lijn: AI Prediction (Chart.js-equivalent: borderDash [5,5] ≈ lineStyle Dashed). */
    predictionSeries: null,
    _lastBarTime: null,
    _lastBarStepSec: 300,

    onActivate() {
        if (typeof window.runSequencedStartup === 'function' || typeof window.switchTab === 'function') {
            console.log("[Legacy ModuleTerminal] Bypassed because monolithic terminal.js is active.");
            return;
        }
        this.refresh();
    },

    onDeactivate() {
        if (this.chart) {
            try { this.chart.remove(); } catch (_) {}
            this.chart = null;
            this.series = null;
            this.predictionSeries = null;
            this._lastBarTime = null;
        }
    },

    chartOptions: {
        autoSize: true,
        layout: { background: { color: "#000000" }, textColor: "#888888" },
        grid: { vertLines: { color: "#222" }, horzLines: { color: "#222" } },
        timeScale: { 
            timeVisible: true, 
            secondsVisible: false, 
            barSpacing: 15,
            tickMarkFormatter: (time) => {
                // Tijd is al handmatig met +7200 verhoogd, parse als UTC om dubbele browser-offset te voorkomen
                const dt = new Date(Number(time) * 1000);
                const hh = String(dt.getUTCHours()).padStart(2, "0");
                const mm = String(dt.getUTCMinutes()).padStart(2, "0");
                return `${hh}:${mm}`;
            }
        },
        localization: { locale: 'nl-NL' }
    },

    seriesOptions: {
        upColor: "#39ff14", downColor: "#ff3131",
        borderVisible: true, wickUpColor: "#39ff14", wickDownColor: "#ff3131"
    },

    initChart(containerId = "priceChart") {
        const container = document.getElementById(containerId);
        if (!container || !window.LightweightCharts) return;
        if (this.chart) return;

        this.chart = LightweightCharts.createChart(container, this.chartOptions);
        
        if (typeof this.chart.addCandlestickSeries === "function") {
            this.series = this.chart.addCandlestickSeries(this.seriesOptions);
        } else if (typeof this.chart.addSeries === "function" && window.LightweightCharts.CandlestickSeries) {
            this.series = this.chart.addSeries(window.LightweightCharts.CandlestickSeries, this.seriesOptions);
        } else {
            console.error("[ModuleTerminal] LightweightCharts API mismatch: kon geen series aanmaken.");
        }
    },

    ensurePredictionSeries() {
        if (this.predictionSeries || !this.chart) return;
        const LW = window.LightweightCharts;
        const dashStyle = LW && LW.LineStyle !== undefined ? LW.LineStyle.Dashed : 2;
        const lineOpts = {
            color: "#00f3ff",
            lineWidth: 2,
            lineStyle: dashStyle,
            lastValueVisible: true,
            priceLineVisible: false,
        };
        if (typeof this.chart.addLineSeries === "function") {
            this.predictionSeries = this.chart.addLineSeries(lineOpts);
        } else if (typeof this.chart.addSeries === "function" && LW && LW.LineSeries) {
            this.predictionSeries = this.chart.addSeries(LW.LineSeries, lineOpts);
        }
    },

    _isoToChartTime(iso) {
        const s = String(iso || "");
        if (!s) return null;
        const t = Math.floor(new Date(s.endsWith("Z") || s.includes("+") ? s : s + "Z").getTime() / 1000) + 7200;
        return Number.isFinite(t) && t > 0 ? t : null;
    },

    /**
     * Payload van GET /api/v1/predictions?symbol=... (of pad-variant): zet ``predicted`` op de overlay-serie.
     */
    applyPredictionsApiResponse(res) {
        if (!res || typeof res !== "object") return;
        window.__lastApiPredictionResponse = res;
        this.initChart("priceChart");
        if (!this.chart || !this.series) return;
        const pred = res.predicted;
        if (!Array.isArray(pred) || !pred.length) {
            this.ensurePredictionSeries();
            if (this.predictionSeries) this.predictionSeries.setData([]);
            return;
        }
        const pts = [];
        for (const row of pred) {
            const tm = this._isoToChartTime(row.timestamp);
            const v = Number(row.predicted_price);
            if (tm && Number.isFinite(v)) pts.push({ time: tm, value: v });
        }
        if (!pts.length) return;
        this.ensurePredictionSeries();
        if (!this.predictionSeries) return;
        try {
            this.predictionSeries.setData(pts);
        } catch (e) {
            console.warn("[ModuleTerminal] API prediction overlay:", e);
        }
    },

    applyPredictionFromStats(stats) {
        if (!stats || typeof stats !== "object") return;
        window.__lastPredictionStats = stats;
        this.initChart("priceChart");
        if (!this.chart || !this.series) return;

        const mSel = (window.AppCore && window.AppCore.state && window.AppCore.state.selectedMarket) || window.selectedMarket || "";
        if (stats.market && mSel && String(stats.market).toUpperCase() !== String(mSel).toUpperCase()) {
            this.ensurePredictionSeries();
            if (this.predictionSeries) this.predictionSeries.setData([]);
            return;
        }

        const arr = stats.predicted_price;
        if (!Array.isArray(arr) || !arr.length) {
            this.ensurePredictionSeries();
            if (this.predictionSeries) this.predictionSeries.setData([]);
            return;
        }

        const t0 = this._lastBarTime;
        const step = Math.max(60, Number(this._lastBarStepSec) || 300);
        if (!t0 || !Number.isFinite(t0)) return;

        const leadBars = Math.max(1, Math.min(8, Number(stats.prediction_bar_lead) || 2));
        const pts = [];
        for (let i = 0; i < arr.length; i++) {
            const v = Number(arr[i]);
            if (!Number.isFinite(v)) continue;
            pts.push({ time: t0 + step * (leadBars + i + 1), value: v });
        }
        if (!pts.length) return;

        this.ensurePredictionSeries();
        if (!this.predictionSeries) return;
        try {
            this.predictionSeries.setData(pts);
        } catch (e) {
            console.warn("[ModuleTerminal] prediction overlay:", e);
        }
    },

    async refresh() {
        this.initChart("priceChart");
        const market = (window.AppCore && window.AppCore.state && window.AppCore.state.selectedMarket) ? window.AppCore.state.selectedMarket : (window.selectedMarket || "BTC-EUR");
        let interval = window.chartInterval || "5m";
        if (!window.chartInterval) {
            try {
                const saved = String(window.localStorage.getItem("trading_timeframe") || "").toLowerCase();
                interval = saved === "ai" ? "5m" : (saved || "5m");
            } catch (_e) {
                interval = "5m";
            }
        }
        try {
            const res = await fetch(`/api/v1/history?pair=${encodeURIComponent(market)}&lookback_days=30&interval=${interval}`);
            if (!res.ok) return;
            const data = await res.json();
            requestAnimationFrame(() => {
                this.updateChartData(data);
            });
        } catch (err) {
            console.error("[ModuleTerminal] Fetch failed:", err);
        }
    },

    updateChartData(data) {
        if (!this.series || !this.chart) return;

        let currentRange = null;
        let isAtEnd = true;
        try { currentRange = this.chart.timeScale().getVisibleLogicalRange(); } catch(e) {}

        const lineData = (data.labels || []).map((l, i) => ({
            // Forceer UTC parsing + 7200s (2 uur) offset voor lokale Amsterdamse tijd
            time: Math.floor(new Date(l.endsWith('Z') || l.includes('+') ? l : l + 'Z').getTime() / 1000) + 7200,
            value: Number(data.prices[i])
        })).filter(p => p.time && Number.isFinite(p.value));

        if (currentRange && lineData.length > 0) {
            isAtEnd = currentRange.to >= lineData.length - 3;
        }

        const candles = [];
        for (let i = 0; i < lineData.length; i++) {
            const prev = i > 0 ? lineData[i - 1].value : lineData[i].value;
            const cur = lineData[i].value;
            const wick = Math.abs(cur) * 0.001;
            candles.push({ time: lineData[i].time, open: prev, high: Math.max(prev, cur) + wick, low: Math.min(prev, cur) - wick, close: cur });
        }

        this.series.setData(candles);

        if (lineData.length >= 2) {
            const n = lineData.length;
            this._lastBarTime = lineData[n - 1].time;
            this._lastBarStepSec = Math.max(60, lineData[n - 1].time - lineData[n - 2].time);
        } else if (lineData.length === 1) {
            this._lastBarTime = lineData[0].time;
            this._lastBarStepSec = 300;
        }

        if (window.__lastPredictionStats) {
            this.applyPredictionFromStats(window.__lastPredictionStats);
        }
        if (window.__lastApiPredictionResponse) {
            this.applyPredictionsApiResponse(window.__lastApiPredictionResponse);
        }

        // Zoom Persistentie & Scroll Beveiliging
        if (!currentRange) {
            const totalBars = lineData.length;
            if (totalBars > 120) this.chart.timeScale().setVisibleLogicalRange({ from: totalBars - 120, to: totalBars + 2 });
            else this.chart.timeScale().fitContent();
        } else if (!isAtEnd) {
            this.chart.timeScale().setVisibleLogicalRange(currentRange);
        } else {
            const rangeWidth = currentRange.to - currentRange.from;
            this.chart.timeScale().setVisibleLogicalRange({ from: lineData.length - rangeWidth, to: lineData.length + 2 });
        }
    }
};