
window.ModuleTerminal = {
    chart: null,
    series: null,
    
    onActivate() {
        if (typeof window.runSequencedStartup === 'function' || typeof window.switchTab === 'function') {
            console.log("[Legacy ModuleTerminal] Bypassed because monolithic terminal.js is active.");
            return;
        }
        this.refresh();
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

    async refresh() {
        this.initChart("priceChart");
        const market = (window.AppCore && window.AppCore.state && window.AppCore.state.selectedMarket) ? window.AppCore.state.selectedMarket : (window.selectedMarket || "BTC-EUR");
        const interval = window.chartInterval || "5m";
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