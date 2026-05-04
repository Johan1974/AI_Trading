/*
BESTANDSNAAM: /home/johan/AI_Trading/app/static/js/main.js
FUNCTIE: Frontend logica voor UI updates, WebSocket/API polling en grafiek interacties.
*/

window.chartInterval = '5m';

window.setChartInterval = function(interval) {
    console.log("Chart interval veranderd naar:", interval);
    window.chartInterval = interval;
    
    const foot = document.getElementById("chartTimeFootnote");
    if (foot) foot.textContent = `Timeframes: 1m | 5m | 15m | 60m | active=${interval}`;
    
    if (typeof window.updateChart === 'function') {
        window.updateChart(window.selectedChartPair || window.selectedMarket || 'BTC-EUR');
    } else if (window.ModuleTerminal && typeof window.ModuleTerminal.refresh === 'function') {
        window.ModuleTerminal.refresh();
    }
};

/**
 * Update de volledige gebruikersinterface met nieuwe data van de bot.
 */
function updateUI(data) {
    if (!data || typeof data !== 'object') return;

    // 1. Restaureer de verwijderde mappings voor Terminal, Brain en Ledger
    const priceEl = document.getElementById('btc-price');
    if (priceEl) {
        if (data.price > 0) priceEl.innerText = `€ ${Number(data.price).toFixed(2)}`;
        else if (!priceEl.innerText.includes("Laden")) priceEl.innerText = "Laden...";
    }

    const changeEl = document.getElementById('market-24h-change');
    if (changeEl && data.price_change_pct_24h !== undefined) changeEl.innerText = `${Number(data.price_change_pct_24h).toFixed(2)}%`;

    const volEl = document.getElementById('market-volatility');
    if (volEl && data.volatility_pct_4h !== undefined) volEl.innerText = `${Number(data.volatility_pct_4h).toFixed(2)}%`;

    const reasonEl = document.getElementById('market-reason');
    if (reasonEl && data.decision_reasoning) reasonEl.innerText = data.decision_reasoning;

    const allocEl = document.getElementById('allocatie');
    if (allocEl && data.allocation_summary) allocEl.innerText = data.allocation_summary;

    const fgEl = document.getElementById('terminalFearGreed');
    if (fgEl && data.fear_greed_score !== undefined) {
        const val = Number(data.fear_greed_score);
        const displayVal = val <= 1.0 ? Math.round(val * 100) : Math.round(val);
        fgEl.innerText = `F&G: ${displayVal}/100 (${data.fear_greed_class || 'Neutral'})`;
    }

    const sentEl = document.getElementById('sentiment-value');
    if (sentEl && data.sentiment_score !== undefined) sentEl.innerText = Number(data.sentiment_score).toFixed(3);

    const confEl = document.getElementById('rl-confidence');
    if (confEl && data.rl_confidence !== undefined) confEl.innerText = Number(data.rl_confidence).toFixed(2);

    const modEl = document.getElementById('model-version');
    if (modEl && data.model_version) modEl.innerText = data.model_version;

    if (data.paper_portfolio) {
        ['total-balance', 'headerBalanceEuro'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerText = `€ ${Number(data.paper_portfolio.equity || 0).toFixed(2)}`;
        });

        const cashEl = document.getElementById('headerCash');
        if (cashEl && data.paper_portfolio.cash !== undefined) {
            cashEl.innerText = `€ ${Number(data.paper_portfolio.cash).toFixed(2)}`;
        }

        ['trade-count', 'ledgerPerfClosed'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerText = data.paper_portfolio.trades_count || 0;
        });

        ['pnl-total', 'ledgerPerfPnlEur'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerText = `€ ${Number(data.paper_portfolio.realized_pnl_eur || 0).toFixed(2)}`;
        });
    }

    const hardwareMappings = {
        // Waardes voor de tekst-labels in de Hardware-tab
        'ringCpuVal': (data.cpu_load || "0") + "%",
        'ringRamVal': (data.ram_usage || "0") + "%",
        'ringGpuVal': (data.gpu_temp || "0") + "°C",
        'ringDiskVal': (data.disk_usage || "0") + "%",

        // Waardes voor de header (voor de zekerheid meenemen)
        'headerStatCpu': "🖥️ " + (data.cpu_load || "0") + "%",
        'headerStatRam': "🧠 " + (data.ram_usage || "0") + "%",
        'headerStatGpu': "🎮 " + (data.gpu_temp || "0") + "°C",
        'headerStatDisk': "💾 " + (data.disk_usage || "0") + "%"
    };

    // 1. Update de tekst-waardes
    for (const [id, value] of Object.entries(hardwareMappings)) {
        const el = document.getElementById(id);
        if (el) el.innerText = value;
    }

    // 2. Update de visuele "ringen" (de arcs)
    // We passen de CSS variabele --pct aan zodat de cirkel volloopt
    updateRingMeter('ringMeterCpu', data.cpu_load);
    updateRingMeter('ringMeterRam', data.ram_usage);
    updateRingMeter('ringMeterGpu', (data.gpu_temp / 100) * 100); // GPU temp als percentage van 100°C
    updateRingMeter('ringMeterDisk', data.disk_usage);

    // 3. Update Multi-Ticker Dropdown (Synchronisatie met Scanner)
    if (data.active_markets && Array.isArray(data.active_markets)) {
        const select = document.getElementById('marketSelect');
        if (select) {
            const currentVal = select.value;
            let newHtml = '';
            let foundCurrent = false;
            data.active_markets.forEach(m => {
                const mName = m.market || m; // Ondersteuning voor dict of string
                newHtml += `<option value="${mName}">${mName}</option>`;
                if (mName === currentVal) foundCurrent = true;
            });
            if (select.innerHTML !== newHtml) {
                select.innerHTML = newHtml;
                if (foundCurrent) select.value = currentVal; // Behoud de actieve selectie
            }
        }
    }

    
    // 4. Update AI Brain Weights (Voorkomt timeouts en 0.000 in de validator)
    if (data.ai_weights) {
        const corrVal = Number(data.ai_weights.correlation || 0).toFixed(4);
        const newsVal = Number(data.ai_weights.news || 0).toFixed(4);
        const priceVal = Number(data.ai_weights.price || 0).toFixed(4);

        ['weight-correlation', 'js-corr-value'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerText = corrVal;
        });
        ['weight-news', 'js-news-weight'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerText = newsVal;
        });
        ['weight-price', 'js-price-weight'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerText = priceVal;
        });
    }
}

// Hulpfunctie om de ringen visueel te laten bewegen
function updateRingMeter(id, value) {
    const el = document.getElementById(id);
    if (el && value !== undefined) {
        // Dit zet de CSS property --pct die in je stylesheet wordt gebruikt voor de animatie
        el.style.setProperty('--pct', value);
    }
}

// Deze functie haalt de data op van de server
async function fetchStats() {
    try {
        const response = await fetch('/api/v1/stats');
        if (!response.ok) throw new Error(`Server fout: ${response.status}`);

        const data = await response.json();
        updateUI(data); // Stuur de ECHTE data naar de UI

    } catch (error) {
        console.error("Oeps! Kon de data niet ophalen:", error);
    }
}

// START DE MOTOR
document.addEventListener("DOMContentLoaded", () => {
    if (typeof window.runSequencedStartup === 'function' || typeof window.switchTab === 'function') {
        console.log("[Legacy main.js] Bypassed because monolithic terminal.js is active.");
        return;
    }
    // Haal direct data op bij het laden
    fetchStats();
    
    // Ververs daarna elke 2 seconden (niet vaker nodig, scheelt stroom en CPU)
    setInterval(fetchStats, 2000);

    // Fix de 'Frozen Price' Bug: Luister globaal naar de dropdown en forceer een update
    document.body.addEventListener('change', async (e) => {
        if (e.target && e.target.id === 'marketSelect') {
            const newMarket = e.target.value;
            console.log(`[UI] Market switch getriggerd naar: ${newMarket}`);
            
            // DIRECTE UI UPDATE: Voorkom 'Frozen Price' en geef direct visuele feedback
            const priceEl = document.getElementById('btc-price');
            if (priceEl) priceEl.innerText = "Laden...";
            
            try {
                await fetch(`/markets/select?market=${newMarket}`, { method: 'POST' });
            } catch (err) {
                console.error("[UI] Fout bij wisselen van markt:", err);
            }
            
            // Wacht heel even zodat de backend Redis snapshot is gesynct, en forceer dan refresh
            setTimeout(fetchStats, 600);
        }
    });
});