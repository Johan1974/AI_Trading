/*
BESTANDSNAAM: /home/johan/AI_Trading/app/static/js/main.js
FUNCTIE: Frontend logica voor UI updates, WebSocket/API polling en grafiek interacties.
*/

const LEGACY_CHART_INTERVAL_STORAGE_KEY = "trading_timeframe";
const LEGACY_ALLOWED_INTERVALS = new Set(["ai", "1m", "5m", "15m", "60m"]);

function legacyNormalizeInterval(raw) {
    const v = String(raw || "").toLowerCase();
    return LEGACY_ALLOWED_INTERVALS.has(v) ? v : "ai";
}

function legacySelectionToInterval(sel) {
    const s = legacyNormalizeInterval(sel);
    return s === "ai" ? "5m" : s;
}

function legacyUpdateTimeframeUi(activeSelection) {
    const chosen = legacyNormalizeInterval(activeSelection);
    try {
        const btns = document.querySelectorAll("#chartTimeframeControls .chart-timeframe-btn[data-timeframe]");
        btns.forEach((btn) => {
            const tf = legacyNormalizeInterval(btn.getAttribute("data-timeframe"));
            const on = tf === chosen;
            btn.classList.toggle("is-active", on);
            btn.setAttribute("aria-pressed", on ? "true" : "false");
        });
    } catch (_e) {}
}

let legacySavedSelection = "ai";
try {
    legacySavedSelection = legacyNormalizeInterval(window.localStorage.getItem(LEGACY_CHART_INTERVAL_STORAGE_KEY));
} catch (_e) {}

window.chartInterval = legacySelectionToInterval(legacySavedSelection);

window.setChartInterval = function(interval) {
    const selection = legacyNormalizeInterval(interval);
    console.log("Chart interval veranderd naar:", selection);
    window.chartInterval = legacySelectionToInterval(selection);
    try {
        window.localStorage.setItem(LEGACY_CHART_INTERVAL_STORAGE_KEY, selection);
    } catch (_e) {}

    const foot = document.getElementById("chartTimeFootnote");
    if (foot) foot.textContent = `Timeframes: AI | 1m | 5m | 15m | 60m | active=${selection.toUpperCase()}`;
    legacyUpdateTimeframeUi(selection);

    if (typeof window.updateChart === 'function') {
        window.updateChart(window.selectedChartPair || window.selectedMarket || 'BTC-EUR');
    } else if (window.ModuleTerminal && typeof window.ModuleTerminal.refresh === 'function') {
        window.ModuleTerminal.refresh();
    }
};

// Schrijft alleen naar DOM als de waarde veranderd is (voorkomt onnodige reflows)
function setTextIfChanged(el, text) {
    if (el && el.textContent !== text) el.textContent = text;
}

// Stale-data teller: toont een indicator als meerdere polls achter elkaar mislukken
let _fetchFailCount = 0;
let _wasStale = false;
function _setStaleIndicator(failed) {
    if (failed) { _fetchFailCount++; } else { _fetchFailCount = 0; }
    const stale = _fetchFailCount >= 3;
    if (stale === _wasStale) return; // niets veranderd, sla DOM-werk over
    _wasStale = stale;
    const priceEl = document.getElementById('btc-price');
    if (priceEl) {
        priceEl.style.opacity = stale ? "0.45" : "";
        priceEl.title = stale ? "Data verouderd — verbinding verbroken?" : "";
    }
    const liveEl = document.querySelector('.cockpit-status-live__dot');
    if (liveEl) { liveEl.style.background = stale ? "#ff3e3e" : ""; }
}

// Cache voor ring-meter CSS-waarden: sla style.setProperty over als waarde niet veranderd is
const _ringMeterCache = {};
function updateRingMeter(id, value) {
    if (value === undefined) return;
    const v = String(Number(value).toFixed(1));
    if (_ringMeterCache[id] === v) return;
    _ringMeterCache[id] = v;
    const el = document.getElementById(id);
    if (el) el.style.setProperty('--pct', value);
}

/**
 * Update de volledige gebruikersinterface met nieuwe data van de bot.
 */
function updateUI(data) {
    if (!data || typeof data !== 'object') return;

    // Prijs: nl-NL formattering via window.formatEurNl (gedeeld met app_core.js)
    const priceEl = document.getElementById('btc-price');
    if (priceEl) {
        const n = Number(data.price);
        const priceStr = Number.isFinite(n) && n > 0 ? window.formatEurNl(n) : "Laden...";
        setTextIfChanged(priceEl, priceStr);
    }

    const changeEl = document.getElementById('market-24h-change');
    if (changeEl && data.price_change_pct_24h !== undefined) {
        setTextIfChanged(changeEl, `${Number(data.price_change_pct_24h).toFixed(2)}%`);
    }

    const volEl = document.getElementById('market-volatility');
    if (volEl && data.volatility_pct_4h !== undefined) {
        setTextIfChanged(volEl, `${Number(data.volatility_pct_4h).toFixed(2)}%`);
    }

    const reasonEl = document.getElementById('market-reason');
    if (reasonEl && data.decision_reasoning) setTextIfChanged(reasonEl, data.decision_reasoning);

    const allocEl = document.getElementById('allocatie');
    if (allocEl && data.allocation_summary) setTextIfChanged(allocEl, data.allocation_summary);

    const fgEl = document.getElementById('terminalFearGreed');
    if (fgEl && data.fear_greed_score !== undefined) {
        const val = Number(data.fear_greed_score);
        const displayVal = val <= 1.0 ? Math.round(val * 100) : Math.round(val);
        setTextIfChanged(fgEl, `F&G: ${displayVal}/100 (${data.fear_greed_class || 'Neutral'})`);
    }

    const sentEl = document.getElementById('sentiment-value');
    if (sentEl && data.sentiment_score !== undefined) {
        setTextIfChanged(sentEl, Number(data.sentiment_score).toFixed(3));
    }

    const confEl = document.getElementById('rl-confidence');
    if (confEl && data.rl_confidence !== undefined) {
        setTextIfChanged(confEl, Number(data.rl_confidence).toFixed(2));
    }

    const modEl = document.getElementById('model-version');
    if (modEl && data.model_version) setTextIfChanged(modEl, data.model_version);

    if (data.paper_portfolio) {
        // window.formatEurNl is de gedeelde nl-NL formatter (app_core.js:1)
        const fmt = (v) => window.formatEurNl(Number(v) || 0);
        ['total-balance', 'headerBalanceEuro'].forEach(id => {
            const el = document.getElementById(id);
            if (el) setTextIfChanged(el, fmt(data.paper_portfolio.equity));
        });
        const cashEl = document.getElementById('headerCash');
        if (cashEl && data.paper_portfolio.cash !== undefined) {
            setTextIfChanged(cashEl, fmt(data.paper_portfolio.cash));
        }
        ['trade-count', 'ledgerPerfClosed'].forEach(id => {
            const el = document.getElementById(id);
            if (el) setTextIfChanged(el, String(data.paper_portfolio.trades_count || 0));
        });
        ['pnl-total', 'ledgerPerfPnlEur'].forEach(id => {
            const el = document.getElementById(id);
            if (el) setTextIfChanged(el, fmt(data.paper_portfolio.realized_pnl_eur));
        });
    }

    // Hardware ring-meters
    const hwMap = {
        ringCpuVal:  `${Number(data.cpu_load  || 0).toFixed(0)}%`,
        ringRamVal:  `${Number(data.ram_usage || 0).toFixed(0)}%`,
        ringGpuVal:  `${Number(data.gpu_temp  || 0).toFixed(0)}°C`,
        ringDiskVal: `${Number(data.disk_usage || 0).toFixed(0)}%`,
    };
    for (const [id, val] of Object.entries(hwMap)) {
        const el = document.getElementById(id);
        if (el) setTextIfChanged(el, val);
    }
    updateRingMeter('ringMeterCpu',  data.cpu_load);
    updateRingMeter('ringMeterRam',  data.ram_usage);
    updateRingMeter('ringMeterGpu',  data.gpu_temp);
    updateRingMeter('ringMeterDisk', data.disk_usage);

    // Multi-Ticker Dropdown: alleen rebuild als inhoud verschilt
    if (data.active_markets && Array.isArray(data.active_markets)) {
        const select = document.getElementById('marketSelect');
        if (select) {
            const currentVal = select.value;
            let newHtml = '';
            let foundCurrent = false;
            data.active_markets.forEach(m => {
                const mName = m.market || m;
                newHtml += `<option value="${mName}">${mName}</option>`;
                if (mName === currentVal) foundCurrent = true;
            });
            if (select.innerHTML !== newHtml) {
                select.innerHTML = newHtml;
                if (foundCurrent) select.value = currentVal;
            }
        }
    }

    // AI Brain Weights
    if (data.ai_weights) {
        const corrVal  = Number(data.ai_weights.correlation || 0).toFixed(4);
        const newsVal  = Number(data.ai_weights.news || 0).toFixed(4);
        const priceVal = Number(data.ai_weights.price || 0).toFixed(4);
        ['weight-correlation', 'js-corr-value'].forEach(id => { const el = document.getElementById(id); if (el) setTextIfChanged(el, corrVal); });
        ['weight-news', 'js-news-weight'].forEach(id => { const el = document.getElementById(id); if (el) setTextIfChanged(el, newsVal); });
        ['weight-price', 'js-price-weight'].forEach(id => { const el = document.getElementById(id); if (el) setTextIfChanged(el, priceVal); });
    }

    // Social Buzz strip: top-3 snelst bewegende munten (deadband +-1%/h om ruis te filteren)
    const buzzEl = document.getElementById('socialBuzzStrip');
    if (buzzEl && data.social_buzz_summary && Array.isArray(data.social_buzz_summary.lines)) {
        const lines = data.social_buzz_summary.lines.slice(0, 3);
        if (lines.length) {
            const parts = lines.map(l => {
                const ticker = String(l.market || l.ticker || "").replace("-EUR", "");
                const vel    = Number(l.velocity_pct_1h || 0);
                const arrow  = vel > 1 ? "↑" : vel < -1 ? "↓" : "→";
                return `${ticker} ${arrow}${Math.abs(vel).toFixed(1)}%/h`;
            });
            setTextIfChanged(buzzEl, "Social Buzz: " + parts.join("  ·  "));
        }
    }

    document.dispatchEvent(new CustomEvent('ai-trading-stats', { detail: data }));
}

// Deze functie haalt de data op van de server
async function fetchStats() {
    try {
        const mk = (document.getElementById('marketSelect') && document.getElementById('marketSelect').value)
            || (window.AppCore && window.AppCore.state && window.AppCore.state.selectedMarket)
            || window.selectedMarket
            || 'BTC-EUR';
        const response = await fetch(`/api/v1/stats?symbol=${encodeURIComponent(String(mk).trim())}`);
        if (!response.ok) throw new Error(`Server fout: ${response.status}`);
        const data = await response.json();
        _setStaleIndicator(false);
        updateUI(data);
    } catch (error) {
        _setStaleIndicator(true);
        console.error("Oeps! Kon de data niet ophalen:", error);
    }
}

// START DE MOTOR
document.addEventListener("DOMContentLoaded", () => {
    if (typeof window.runSequencedStartup === 'function' || typeof window.switchTab === 'function') {
        console.log("[Legacy main.js] Bypassed because monolithic terminal.js is active.");
        return;
    }
    legacyUpdateTimeframeUi(legacySavedSelection);
    const foot = document.getElementById("chartTimeFootnote");
    if (foot) foot.textContent = `Timeframes: AI | 1m | 5m | 15m | 60m | active=${legacySavedSelection.toUpperCase()}`;
    fetchStats();
    setInterval(fetchStats, 2000);

    // Market-switch: valideer POST succes, directe feedback, geen vaste delay
    document.body.addEventListener('change', async (e) => {
        if (!e.target || e.target.id !== 'marketSelect') return;
        const newMarket = e.target.value;
        console.log(`[UI] Market switch getriggerd naar: ${newMarket}`);

        const priceEl = document.getElementById('btc-price');
        if (priceEl) priceEl.textContent = "Laden...";

        try {
            const res = await fetch(`/markets/select?market=${encodeURIComponent(newMarket)}`, { method: 'POST' });
            if (!res.ok) {
                console.warn(`[UI] Market-switch POST mislukt (${res.status}), UI blijft op vorige markt.`);
                if (priceEl) priceEl.textContent = "Fout bij wisselen";
                return;
            }
        } catch (err) {
            console.error("[UI] Fout bij wisselen van markt:", err);
            if (priceEl) priceEl.textContent = "Netwerkfout";
            return;
        }

        document.dispatchEvent(new Event('ai-trading-refresh-activity'));
        void fetchStats();
    });
});
