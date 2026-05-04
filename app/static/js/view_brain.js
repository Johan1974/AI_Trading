window.BrainView = {
    REWARD_MA_WINDOW: 100,
    BRAIN_NEON_LOSS_POLICY: "#00f8ff",
    BRAIN_NEON_LOSS_VALUE: "#ff3131",
    BRAIN_NEON_REWARD: "#39ff14",
    BRAIN_NEON_REWARD_MA: "#fff176",

    onActivate() {
        if (typeof window.runSequencedStartup === 'function' || typeof window.switchTab === 'function') {
            console.log("[Legacy BrainView] Bypassed because monolithic terminal.js is active.");
            return;
        }
        const buffer = window.AppState.buffers.latestBrainPayload;
        if (buffer) {
            this.onSocketMessage(buffer);
            window.AppState.buffers.latestBrainPayload = null;
        }
        this.fetchBrainLab();
    },

    async fetchBrainLab() {
        try {
            const marketParam = encodeURIComponent(window.selectedChartPair || "BTC-EUR");
            const [reasoningRes, fiRes, monitorRes, stateRes, lagRes] = await Promise.all([
                fetch("/api/v1/brain/reasoning"),
                fetch(`/api/v1/brain/feature-importance?market=${marketParam}`),
                fetch("/api/v1/brain/training-monitor"),
                fetch("/api/v1/brain/state-overview"),
                fetch("/api/v1/brain/news-lag"),
            ]);
            if (!reasoningRes.ok) return;
            
            this.renderDashboard(
                await reasoningRes.json(), await fiRes.json(),
                await monitorRes.json(), await stateRes.json(), await lagRes.json()
            );
        } catch (error) {
            console.error("[BrainView] Fetch failed", error);
        }
    },

    onSocketMessage(rawData) {
        let d = rawData;
        if (typeof d === "string") { try { d = JSON.parse(d); } catch (_) { return; } }
        if (!d || d.__ws === "hb" || (d.t !== "brain_stats" && d.t !== "brain_data")) return;
        
        const market = String(window.selectedChartPair || "BTC-EUR").toUpperCase();
        this.paintCharts(d.tm, {
            feature_weights: d.fw || {},
            rl_observation: d.rl || {},
            feature_weights_policy: d.fwp || {},
            social_buzz: d.sb || {},
            market: market
        });
    },

    escapeHtmlText(text) { return String(text || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); },

    renderDashboard(reasoningData, fiData, monitorData, stateData, lagData) {
        this.paintCharts(monitorData, fiData);

        const net = monitorData.network_logs || {};
        const latestKl = (net.approx_kl || []).slice(-1)[0];
        const latestValueLoss = (net.value_loss || []).slice(-1)[0];
        
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
        const net = monitor.network_logs || {};
        const policyLoss = Array.isArray(monitor.loss) ? monitor.loss : [];
        const valueLoss = Array.isArray(net.value_loss) ? net.value_loss : [];
        const rawRewards = Array.isArray(monitor.reward) ? monitor.reward : [];

        // Status bar
        const calibrationEl = document.getElementById("strategyCalibrationStatus");
        if (calibrationEl) calibrationEl.textContent = `Strategy actief voor ${fi.market}`;

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
    },

    renderStats(monitorData, stateData) {
        const stats = monitorData.stats || {};
        if (document.getElementById("brainTabStatLR")) document.getElementById("brainTabStatLR").textContent = Math.max(1e-5, Number(stats.learning_rate || 0)).toExponential(2);
        if (document.getElementById("brainTabStatSteps")) document.getElementById("brainTabStatSteps").textContent = Number(stats.global_step_count || 0).toLocaleString();
        if (document.getElementById("brainTabStatExplore")) document.getElementById("brainTabStatExplore").textContent = `${Math.max(5.0, Number(stats.exploration_rate_pct || 0)).toFixed(2)}%`;
        
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
        const lineOpts = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } };
        const ep = monitorData.episode_length || [];
        const ent = monitorData.policy_entropy || [];
        
        window.ChartUtils.upsertChart("brainEpisodeChart", {
            type: "line", data: { labels: ep.map((_, i) => i + 1), datasets: [{ data: ep, borderColor: "#66e8ff", borderWidth: 4, pointRadius: 0 }] }, options: lineOpts
        });
        window.ChartUtils.upsertChart("brainEntropyChart", {
            type: "line", data: { labels: ent.map((_, i) => i + 1), datasets: [{ data: ent, borderColor: "#ff66ff", borderWidth: 4, pointRadius: 0 }] }, options: lineOpts
        });
        
        const bench = monitorData.benchmark || {};
        const benchCfg = {
            type: "bar",
            data: { labels: ["RL Agent", "Buy & Hold", "Alpha"], datasets: [{ data: [bench.rl_pnl_pct||0, bench.buy_hold_pnl_pct||0, bench.alpha_pct||0], backgroundColor: ["#66ff66", "#00f8ff", "#ffff66"] }] },
            options: lineOpts
        };
        window.ChartUtils.upsertChart("brainBenchmarkChart", benchCfg);
        window.ChartUtils.upsertChart("terminalBenchmarkChart", benchCfg);
    }
};