window.ModuleBrain = {
    socket: null,
    reconnectTimer: null,
    pollTimer: null,
    latestPayload: null,
    REWARD_MA_WINDOW: 100,
    BRAIN_NEON_LOSS_POLICY: "#00f8ff",
    BRAIN_NEON_LOSS_VALUE: "#ff3131",
    BRAIN_NEON_REWARD: "#39ff14",
    BRAIN_NEON_REWARD_MA: "#fff176",

    onActivate() {
        if (typeof window.runSequencedStartup === 'function' || typeof window.switchTab === 'function') {
            console.log("[Legacy ModuleBrain] Bypassed because monolithic terminal.js is active.");
            return;
        }
        if (this.latestPayload) {
            this.onSocketMessage(this.latestPayload);
            this.latestPayload = null;
        }
        this.refresh();
        this.connectSocket();
        
        // Polling herstellen: Zorg dat reasoning en stats actueel blijven
        if (this.pollTimer) clearInterval(this.pollTimer);
        this.pollTimer = setInterval(() => {
            if (window.AppCore?.state?.activeTab === 'aibrain') {
                // Fallback: Als de WebSocket gesloten is of faalt, forceer een volledige API data-sync
                if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
                    console.warn("[ModuleBrain] WebSocket niet open. Fallback fetch naar /api/v1/brain/state-overview...");
                    this.refresh();
                } else {
                    // Zelfs als WS open is, halen we periodiek de tekstuele reasoning op (zit niet in WS)
                    this.refresh();
                }
            }
        }, 10000);

    if (this.scanTimer) clearInterval(this.scanTimer);
    this.scanTimer = setInterval(() => {
        const scanEl = document.getElementById("brainLastScanLine");
        if (scanEl && window.__lastEngineTickIso) {
            const sec = Math.max(0, Math.floor((Date.now() - new Date(window.__lastEngineTickIso).getTime()) / 1000));
            scanEl.textContent = `Laatste scan: ${sec} seconden geleden`;
        }
    }, 1000);
    },

    connectSocket() {
        if (this.socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(this.socket.readyState)) return;
        const protocol = window.location.protocol === "https:" ? "wss" : "ws";
        this.socket = new WebSocket(`${protocol}://${window.location.host}/ws/brain-stats`);
        
        this.socket.onmessage = (event) => {
            if (event.data.includes('"hb"')) {
                try { this.socket.send("hb_ack"); } catch(_) {}
                return;
            }
            if (document.hidden || window.AppCore.state.activeTab !== 'aibrain') {
                this.latestPayload = event.data;
                return;
            }
            this.onSocketMessage(event.data);
        };
        this.socket.onclose = () => {
            this.socket = null;
            if (window.AppCore.state.activeTab === 'aibrain') {
                this.reconnectTimer = setTimeout(() => this.connectSocket(), 5000);
            }
        };
    },

    stopSocket() {
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        if (this.socket) {
            const s = this.socket;
            if (s.readyState === WebSocket.OPEN) s.close();
            else if (s.readyState === WebSocket.CONNECTING) s.onopen = () => s.close();
            this.socket = null;
        }
    },

    async refresh() {
        const market = window.AppCore.state.selectedMarket;
        try {
            const reqs = await Promise.allSettled([
                fetch("/api/v1/brain/reasoning"),
                fetch(`/api/v1/brain/feature-importance?market=${encodeURIComponent(market)}`),
                fetch("/api/v1/brain/training-monitor"),
                fetch("/api/v1/brain/state-overview"),
                fetch("/api/v1/brain/news-lag"),
            ]);
            
            const responses = reqs.map(r => r.status === 'fulfilled' ? r.value : null);
            if (responses.some(r => !r || !r.ok)) {
                console.warn("[ModuleBrain] Een of meerdere API endpoints gaven een foutmelding. Fallbacks worden gebruikt.");
            }
            
            const safeJson = async (res, fallback) => (res && res.ok) ? await res.json() : fallback;
            
            const reasoningData = await safeJson(responses[0], { reasoning: "Data niet beschikbaar." });
            const fiData = await safeJson(responses[1], { market });
            const monitorData = await safeJson(responses[2], { stats: {} });
            const stateData = await safeJson(responses[3], { state: {}, weight_focus: {} });
            const lagData = await safeJson(responses[4], { items: [] });
            
            this.renderDashboard(reasoningData, fiData, monitorData, stateData, lagData);
        } catch (err) {
            console.error("[ModuleBrain] Fetch failed:", err);
        }
    },

    onSocketMessage(rawData) {
        let d = rawData;
        if (typeof d === "string") { try { d = JSON.parse(d); } catch (_) { return; } }
        if (!d || d.__ws === "hb" || (d.t !== "brain_stats" && d.t !== "brain_data")) return;
        
        const market = String(window.AppCore.state.selectedMarket || "BTC-EUR").toUpperCase();
        this.paintCharts(d.tm || {}, {
            feature_weights: d.fw || {},
            rl_observation: d.rl || {},
            feature_weights_policy: d.fwp || {},
            social_buzz: d.sb || {},
            market: market
        });
    },

    escapeHtmlText(text) { return String(text || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); },

    renderDashboard(reasoningData, fiData, monitorData, stateData, lagData) {
        // Verberg de laad-overlay zodra we de eerste data renderen
        const fallback = document.getElementById("brainTabFallback");
        if (fallback) fallback.classList.add("hidden");

        this.paintCharts(monitorData, fiData);

        const net = monitorData.network_logs || monitorData.nl || {};
        const latestKl = (net.approx_kl || net.ak || []).slice(-1)[0];
        const latestValueLoss = (net.value_loss || net.vl || []).slice(-1)[0];
        
        let text = reasoningData.reasoning || "Nog geen RL-besluit beschikbaar.";
        if (latestKl !== undefined) text += `\nNetwork health: approx_kl=${Number(latestKl).toFixed(6)}, value_loss=${Number(latestValueLoss).toFixed(6)}.`;
        
        const rb = document.getElementById("brainReasoningBox");
        if (rb) {
            let t = this.escapeHtmlText(text).replace(/(Besluit:\s*(?:HOLD|BUY|SELL)\.)/gi, '<span class="cockpit-besluit-callout">$1</span>');
            rb.innerHTML = t.replace(/\n/g, "<br>");
        }
        this.renderStats(monitorData, stateData);
        this.renderSecondaryCharts(monitorData, lagData);
    },

    paintCharts(monitor, fi) {
        if (!window.ChartUtils) return;
        const net = monitor.network_logs || monitor.nl || {};
        const policyLoss = Array.isArray(monitor.loss) ? monitor.loss : (Array.isArray(monitor.l) ? monitor.l : []);
        const valueLoss = Array.isArray(net.value_loss) ? net.value_loss : (Array.isArray(net.vl) ? net.vl : []);
        const rawRewards = Array.isArray(monitor.reward) ? monitor.reward : (Array.isArray(monitor.r) ? monitor.r : []);

        // Status bar
        const calibrationEl = document.getElementById("strategyCalibrationStatus");
        if (calibrationEl) calibrationEl.textContent = `Strategy actief voor ${fi.market || window.AppCore.state.selectedMarket}`;

        // Loss Chart
        let n = Math.max(policyLoss.length, valueLoss.length, 1);
        window.ChartUtils.upsertChart("brainTabTrainingLossChart", {
            type: "line",
            data: {
                labels: Array.from({ length: n }, (_, i) => String(i + 1)),
                datasets: [
                    { label: "Policy loss", data: policyLoss, borderColor: this.BRAIN_NEON_LOSS_POLICY, borderWidth: 3, pointRadius: 0 },
                    { label: "Value loss", data: valueLoss, borderColor: this.BRAIN_NEON_LOSS_VALUE, borderWidth: 3, pointRadius: 0 }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });

        // Reward Chart
        const normRewards = Array.isArray(monitor.reward_normalized) && monitor.reward_normalized.length ? monitor.reward_normalized : rawRewards.map(x => Math.max(-1, Math.min(1, x/100)));
        const ma = [];
        let sum = 0;
        for (let i = 0; i < normRewards.length; i++) {
            sum += normRewards[i];
            if (i >= this.REWARD_MA_WINDOW) sum -= normRewards[i - this.REWARD_MA_WINDOW];
            ma.push(sum / Math.min(i + 1, this.REWARD_MA_WINDOW));
        }
        
        window.ChartUtils.upsertChart("brainTabRewardChart", {
            type: "line",
            data: {
                labels: (normRewards.length ? normRewards : [0]).map((_, i) => i + 1),
                datasets: [
                    { label: "Reward", data: normRewards, borderColor: this.BRAIN_NEON_REWARD, borderWidth: 3, pointRadius: 0 },
                    { label: "Reward MA", data: ma, borderColor: this.BRAIN_NEON_REWARD_MA, borderDash: [6,4], borderWidth: 2, pointRadius: 0 }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: window.ChartUtils.mergeContrastScale({ min: -1, max: 1 }) } }
        });
        
        // Feature Weights Chart
        const fw = fi.feature_weights || {};
        const fwPairs = Object.keys(fw).map(k => ({k, v: Number(fw[k])||0})).sort((a,b) => b.v - a.v);
        const fwKeys = fwPairs.map(p => p.k);
        const fwVals = fwPairs.map(p => p.v);
        
        window.ChartUtils.upsertChart("brainTabFeatureChart", {
            type: "bar",
            data: {
                labels: fwKeys.length ? fwKeys : ["—"],
                datasets: [{ label: "Weights", data: fwKeys.length ? fwVals : [0], backgroundColor: "#00f8ff" }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
    },

    renderStats(monitorData, stateData) {
        if (!window.ChartUtils) return;
        const stats = monitorData.stats || monitorData.s || {};
        if (document.getElementById("brainTabStatLR")) document.getElementById("brainTabStatLR").textContent = Math.max(1e-5, Number(stats.learning_rate || 0)).toExponential(2);
        if (document.getElementById("brainTabStatSteps")) document.getElementById("brainTabStatSteps").textContent = Number(stats.global_step_count || 0).toLocaleString();
        if (document.getElementById("brainTabStatExplore")) document.getElementById("brainTabStatExplore").textContent = `${Math.max(5.0, Number(stats.exploration_rate_pct || 0)).toFixed(2)}%`;
        if (document.getElementById("brainTabStatDiscount")) document.getElementById("brainTabStatDiscount").textContent = Number(stats.discount_factor || 0.99).toFixed(3);
        if (document.getElementById("brainTabStatBatch")) document.getElementById("brainTabStatBatch").textContent = String(Number(stats.batch_size || 128).toFixed(0));
        
        const corr = monitorData.correlation || {};
        window.ChartUtils.upsertChart("brainCorrelationChart", {
            type: "bar",
            data: {
                labels: ["Sentiment/Price", "News Weight", "Price Weight"],
                datasets: [{ data: [corr.sentiment_price_correlation||0, corr.news_weight||0, corr.price_weight||0], backgroundColor: ["#ff4dff", "#39ff14", "#00f0ff"] }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
        
        document.querySelectorAll(".js-corr-value").forEach(el => el.textContent = (corr.sentiment_price_correlation||0).toFixed(4));
        document.querySelectorAll(".js-news-weight").forEach(el => el.textContent = (corr.news_weight||0).toFixed(4));
        document.querySelectorAll(".js-price-weight").forEach(el => el.textContent = (corr.price_weight||0).toFixed(4));
    },

    renderSecondaryCharts(monitorData, lagData) {
        if (!window.ChartUtils) return;
        const lineOpts = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } };
        const ep = monitorData.episode_length || monitorData.ep || [];
        const ent = monitorData.policy_entropy || monitorData.ent || [];
        
        window.ChartUtils.upsertChart("brainEpisodeChart", {
            type: "line", data: { labels: ep.map((_, i) => i + 1), datasets: [{ data: ep, borderColor: "#66e8ff", borderWidth: 4, pointRadius: 0 }] }, options: lineOpts
        });
        window.ChartUtils.upsertChart("brainEntropyChart", {
            type: "line", data: { labels: ent.map((_, i) => i + 1), datasets: [{ data: ent, borderColor: "#ff66ff", borderWidth: 4, pointRadius: 0 }] }, options: lineOpts
        });
    }
};