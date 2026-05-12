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
                scanEl.textContent =
                    sec === 0 ? "Laatste scan: zojuist" : `Laatste scan: ${sec} seconden geleden`;
            }
        }, 1000);
    },

    _syncBrainScanLabel(iso) {
        if (!iso) return;
        window.__lastEngineTickIso = iso;
        const scanEl = document.getElementById("brainLastScanLine");
        if (!scanEl) return;
        const t = new Date(iso).getTime();
        if (!Number.isFinite(t)) {
            scanEl.textContent = "Laatste scan: —";
            return;
        }
        const sec = Math.max(0, Math.floor((Date.now() - t) / 1000));
        scanEl.textContent = sec === 0 ? "Laatste scan: zojuist" : `Laatste scan: ${sec} seconden geleden`;
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
                fetch("/api/v1/brain/market-models"),
            ]);

            const responses = reqs.map(r => r.status === 'fulfilled' ? r.value : null);
            const safeJson = async (res, fallback) => (res && res.ok) ? await res.json() : fallback;

            const reasoningData = await safeJson(responses[0], { reasoning: "Wachten op eerste besluit (RL inferentie)..." });
            const fiData = await safeJson(responses[1], { market });
            const monitorData = await safeJson(responses[2], { stats: {} });
            const stateData = await safeJson(responses[3], { state: {}, weight_focus: {} });
            const lagData = await safeJson(responses[4], { items: [] });
            const modelsData = await safeJson(responses[5], { models: [] });

            this.renderDashboard(reasoningData, fiData, monitorData, stateData, lagData);
            this.renderModelTable(modelsData);
        } catch (err) {
            console.error("[ModuleBrain] Fetch failed:", err);
        }
    },

    renderModelTable(data) {
        const tbody = document.getElementById("brainModelTableBody");
        if (!tbody) return;
        const models = (data && data.models) || [];
        if (!models.length) { tbody.innerHTML = '<tr><td colspan="7" style="color:#475569;padding:8px;">Geen data</td></tr>'; return; }
        tbody.innerHTML = models.map(m => {
            const badge = m.has_model
                ? `<span style="color:#22d3ee;">✓</span>`
                : `<span style="color:#475569;">—</span>`;
            const ageNum = m.model_age_min != null ? Number(m.model_age_min) : NaN;
            const age = Number.isFinite(ageNum) ? String(ageNum) : "—";
            const ageColor = Number.isFinite(ageNum) && ageNum > 60 ? "#f87171" : "#64748b";
            const buyPct = m.buy_pct ?? 0;
            const buyColor = buyPct > 20 ? "#22d3ee" : buyPct > 5 ? "#94a3b8" : "#475569";
            return `<tr style="border-bottom:1px solid #0f172a;">
                <td style="padding:4px 8px;color:#e2e8f0;">${m.market}</td>
                <td style="text-align:center;padding:4px 8px;">${badge}</td>
                <td style="text-align:right;padding:4px 8px;color:${ageColor};font-weight:${Number.isFinite(ageNum)&&ageNum>60?"600":"400"};">${age}</td>
                <td style="text-align:right;padding:4px 8px;color:#64748b;">${m.replay_rows ?? 0}</td>
                <td style="text-align:right;padding:4px 8px;color:${buyColor};">${buyPct}%</td>
                <td style="text-align:right;padding:4px 8px;color:#64748b;">${m.hold_pct ?? 0}%</td>
                <td style="text-align:right;padding:4px 8px;color:#94a3b8;">${m.sell_pct ?? 0}%</td>
            </tr>`;
        }).join("");
    },

    onSocketMessage(rawData) {
        let d = rawData;
        if (typeof d === "string") { try { d = JSON.parse(d); } catch (_) { return; } }
        if (!d || d.__ws === "hb" || (d.t !== "brain_stats" && d.t !== "brain_data")) return;

        const market = String(window.AppCore.state.selectedMarket || "BTC-EUR").toUpperCase();
        const tm = d.tm || {};
        this.paintCharts(tm, {
            feature_weights: d.fw || {},
            rl_observation: d.rl || {},
            feature_weights_policy: d.fwp || {},
            social_buzz: d.sb || {},
            market: market
        });
        this.renderSecondaryCharts(tm, {});
        this.renderSignalMixBars(d.rl || {}, tm);
        if (d.generated_at) this._syncBrainScanLabel(d.generated_at);
        if (d.thinking_sections) this._renderThinkingSections(d.thinking_sections);
        if (Array.isArray(d.sh)) this.renderShadowTrades(d.sh);
        if (Array.isArray(d.tp) && d.tp.length) this._renderThinkingSteps(d.tp);
        if (Array.isArray(d.buy_block_factors)) this._renderBuyBlockFactors(d.buy_block_factors);
    },

    renderSignalMixBars(rlObs, monitorData) {
        const corr = (monitorData && monitorData.correlation) || {};
        const sentScore = Number(rlObs.sentiment_score ?? corr.sentiment_price_correlation ?? NaN);
        const newsW = Math.min(1, Math.max(0, Number(corr.news_weight || 0)));
        const priceW = Math.min(1, Math.max(0, Number(corr.price_weight || 0)));

        if (Number.isFinite(sentScore)) {
            // Map -1..+1 to 0..100% from center
            const sentPct = Math.min(100, Math.max(0, (sentScore + 1) / 2 * 100));
            const sentColor = sentScore >= 0.05 ? "#39ff14" : sentScore <= -0.05 ? "#ff3131" : "#f8d04f";
            document.querySelectorAll(".js-sentiment-mix-bar").forEach(el => {
                if (el._lastPct === sentPct && el._lastColor === sentColor) return;
                el.style.width = sentPct.toFixed(1) + "%";
                el.style.background = sentColor;
                el._lastPct = sentPct; el._lastColor = sentColor;
            });
            document.querySelectorAll(".js-sentiment-mix-val").forEach(el => {
                const txt = sentScore.toFixed(3);
                if (el.textContent !== txt) el.textContent = txt;
            });
        }

        const newsPct = (newsW * 100).toFixed(1) + "%";
        document.querySelectorAll(".js-news-mix-bar").forEach(el => {
            if (el._lastW !== newsPct) { el.style.width = newsPct; el._lastW = newsPct; }
        });
        document.querySelectorAll(".js-news-weight").forEach(el => {
            const txt = newsW.toFixed(4);
            if (el.textContent !== txt) el.textContent = txt;
        });

        const pricePct = (priceW * 100).toFixed(1) + "%";
        document.querySelectorAll(".js-price-mix-bar").forEach(el => {
            if (el._lastW !== pricePct) { el.style.width = pricePct; el._lastW = pricePct; }
        });
        document.querySelectorAll(".js-price-weight").forEach(el => {
            const txt = priceW.toFixed(4);
            if (el.textContent !== txt) el.textContent = txt;
        });

        const corrVal = Number(corr.sentiment_price_correlation || 0);
        document.querySelectorAll(".js-corr-value").forEach(el => {
            const txt = corrVal.toFixed(4);
            if (el.textContent !== txt) el.textContent = txt;
        });
    },

    escapeHtmlText(text) { return String(text || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); },

    reasoningSourceMeta(reasoningData, bodyLower) {
        const st = String((reasoningData && reasoningData.status) || "").toLowerCase();
        if (st === "model_loading") {
            return { badge: "Model", variant: "model", note: "Weights laden of worker koppelt…" };
        }
        if (st === "warming_up") {
            return { badge: "Scanner", variant: "scanner", note: "RL-snapshot nog niet gevuld" };
        }
        const t = bodyLower || "";
        if (t.includes("rl-snapshot nog leeg") || t.includes("scanner (") || /^\s*scanner:/i.test(t)) {
            return { badge: "Scanner", variant: "scanner", note: "Context tot RL inferentie klaar is" };
        }
        if (t.includes("laatste voorspelling")) {
            return { badge: "Voorspelling", variant: "prediction", note: "Signaal uit last_prediction" };
        }
        if (/besluit:\s*(hold|buy|sell)/i.test(t)) {
            return { badge: "RL policy", variant: "rl", note: "Policy-tekst van de agent" };
        }
        if (t.includes("engine draait")) {
            return { badge: "Status", variant: "status", note: "Worker actief" };
        }
        if (
            t.includes("geen rl-policytekst nog") ||
            t.includes("model aan het inladen") ||
            t.includes("wachten op eerste besluit")
        ) {
            return { badge: "Wacht", variant: "wait", note: "Nog geen volledige policy-tekst" };
        }
        return { badge: "Context", variant: "mixed", note: "Payload of mix" };
    },

    formatReasoningBodyHtml(raw) {
        let t = this.escapeHtmlText(String(raw || ""));
        t = t.replace(/(Besluit:\s*(?:HOLD|BUY|SELL)\.)/gi, '<span class="cockpit-besluit-callout">$1</span>');
        return t.replace(/\n/g, "<br>");
    },

    renderDashboard(reasoningData, fiData, monitorData, stateData, lagData) {
        // Verberg de laad-overlay zodra we de eerste data renderen
        const fallback = document.getElementById("brainTabFallback");
        if (fallback) fallback.classList.add("hidden");

        this.paintCharts(monitorData, fiData);

        if (reasoningData && reasoningData.generated_at) {
            this._syncBrainScanLabel(reasoningData.generated_at);
        }

        if (reasoningData && reasoningData.thinking_sections) {
            this._renderThinkingSections(reasoningData.thinking_sections);
        }
        if (reasoningData && Array.isArray(reasoningData.buy_block_factors)) {
            this._renderBuyBlockFactors(reasoningData.buy_block_factors);
        }

        const net = monitorData.network_logs || monitorData.nl || {};
        const latestKl = (net.approx_kl || net.ak || []).slice(-1)[0];
        const latestValueLoss = (net.value_loss || net.vl || []).slice(-1)[0];
        
        let rawR = String(reasoningData.reasoning || "").trim();
        const isWaitPlaceholder =
            !rawR ||
            rawR.toLowerCase().includes("wachten op eerste besluit");
        let text = isWaitPlaceholder
            ? "Geen RL-policytekst nog — engine of worker kan nog initialiseren; zie ook Terminal (scanner/voorspelling)."
            : rawR;
        if (latestKl !== undefined) text += `\nNetwork health: approx_kl=${Number(latestKl).toFixed(6)}, value_loss=${Number(latestValueLoss).toFixed(6)}.`;
        
        const rb = document.getElementById("brainReasoningBox");
        if (rb && text !== rb._lastReasoningText) {
            rb._lastReasoningText = text;
            const meta = this.reasoningSourceMeta(reasoningData, text.toLowerCase());
            const bodyHtml = this.formatReasoningBodyHtml(text);
            rb.innerHTML =
                `<div class="brain-reasoning-stack">` +
                `<header class="brain-reasoning-kicker" aria-label="Bron van de tekst">` +
                `<span class="brain-reasoning-badge brain-reasoning-badge--${meta.variant}">${this.escapeHtmlText(meta.badge)}</span>` +
                `<span class="brain-reasoning-kicker-note">${this.escapeHtmlText(meta.note)}</span>` +
                `</header>` +
                `<div class="brain-reasoning-body">${bodyHtml}</div>` +
                `</div>`;
        }
        this.renderStats(monitorData, stateData);
        this.renderSecondaryCharts(monitorData, lagData);
        this.renderSocialBuzz(stateData);
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

        // Loss Chart — auto-scale to variance
        let n = Math.max(policyLoss.length, valueLoss.length, 1);
        const allLoss = [...policyLoss, ...valueLoss].filter(v => Number.isFinite(v));
        const lossMin = allLoss.length ? Math.min(...allLoss) : 0;
        const lossMax = allLoss.length ? Math.max(...allLoss) : 1;
        const lossPad = Math.max((lossMax - lossMin) * 0.15, 1e-6);
        window.ChartUtils.upsertChart("brainTabTrainingLossChart", {
            type: "line",
            data: {
                labels: Array.from({ length: n }, (_, i) => String(i + 1)),
                datasets: [
                    { label: "Policy loss", data: policyLoss, borderColor: this.BRAIN_NEON_LOSS_POLICY, borderWidth: 3, pointRadius: 0 },
                    { label: "Value loss", data: valueLoss, borderColor: this.BRAIN_NEON_LOSS_VALUE, borderWidth: 3, pointRadius: 0 }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: lossMin - lossPad, max: lossMax + lossPad } } }
        });

        // Reward Error Chart (√value_loss = TD-residual)
        const reData = Array.isArray(net.reward_error) ? net.reward_error : (Array.isArray(valueLoss) ? valueLoss.map(v => Math.sqrt(Math.max(0, v))) : []);
        const reMa = [];
        const RE_WIN = 20;
        let reSum = 0;
        for (let i = 0; i < reData.length; i++) {
            reSum += reData[i];
            if (i >= RE_WIN) reSum -= reData[i - RE_WIN];
            reMa.push(reSum / Math.min(i + 1, RE_WIN));
        }
        const reFinite = reData.filter(v => Number.isFinite(v));
        const reMin = reFinite.length ? Math.min(...reFinite) : 0;
        const reMax = reFinite.length ? Math.max(...reFinite) : 1;
        const rePad = Math.max((reMax - reMin) * 0.15, 1e-6);
        window.ChartUtils.upsertChart("brainTabRewardErrorChart", {
            type: "line",
            data: {
                labels: reData.map((_, i) => i + 1),
                datasets: [
                    { label: "Reward Error", data: reData, borderColor: "#ff6b6b", borderWidth: 1, pointRadius: 0, fill: false },
                    { label: "MA-20", data: reMa, borderColor: "#f8fafc", borderDash: [6, 3], borderWidth: 2, pointRadius: 0 }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: Math.max(0, reMin - rePad), max: reMax + rePad } } }
        });

        // Feature Weights Chart (klikbaar)
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

        // Sla huidige data op canvas op voor de klik-handler
        const fwCanvas = document.getElementById("brainTabFeatureChart");
        if (fwCanvas) {
            fwCanvas._fwKeys = fwKeys;
            fwCanvas._fwVals = fwVals;
            fwCanvas._rlObs = fi.rl_observation || {};
            if (!fwCanvas._clickBound) {
                fwCanvas._clickBound = true;
                fwCanvas.addEventListener("click", (e) => {
                    const chart = window.ChartUtils.registry["brainTabFeatureChart"];
                    if (!chart) return;
                    const pts = chart.getElementsAtEventForMode(e, "nearest", { intersect: true }, false);
                    if (!pts.length) return;
                    const idx = pts[0].index;
                    const keys = fwCanvas._fwKeys || [];
                    const vals = fwCanvas._fwVals || [];
                    const obs = fwCanvas._rlObs || {};
                    if (idx < keys.length) window.ModuleBrain._explainFeature(keys[idx], vals[idx], obs[keys[idx]]);
                });
            }
        }
    },

    renderStats(monitorData, stateData) {
        if (!window.ChartUtils) return;
        const stats = monitorData.stats || monitorData.s || {};

        // Learning rate with active-learning color
        const policyLossArr = Array.isArray(monitorData.loss) ? monitorData.loss : (Array.isArray(monitorData.l) ? monitorData.l : []);
        const vlossNet = monitorData.network_logs || monitorData.nl || {};
        const valueLossArr = Array.isArray(vlossNet.value_loss) ? vlossNet.value_loss : (Array.isArray(vlossNet.vl) ? vlossNet.vl : []);
        const recentLoss = [...policyLossArr.slice(-20), ...valueLossArr.slice(-20)].filter(v => Number.isFinite(v));
        const lrEl = document.getElementById("brainTabStatLR");
        if (lrEl) {
            lrEl.textContent = Math.max(1e-5, Number(stats.learning_rate || 0)).toExponential(2);
            if (recentLoss.length >= 4) {
                const mean = recentLoss.reduce((s, v) => s + v, 0) / recentLoss.length;
                const variance = recentLoss.reduce((s, v) => s + (v - mean) ** 2, 0) / recentLoss.length;
                lrEl.style.color = variance > 0.01 ? "#4ade80" : "#475569";
            } else {
                lrEl.style.color = "";
            }
        }

        if (document.getElementById("brainTabStatSteps")) document.getElementById("brainTabStatSteps").textContent = Number(stats.global_step_count || 0).toLocaleString();
        if (document.getElementById("brainTabStatExplore")) document.getElementById("brainTabStatExplore").textContent = `${Math.max(5.0, Number(stats.exploration_rate_pct || 0)).toFixed(2)}%`;
        if (document.getElementById("brainTabStatDiscount")) document.getElementById("brainTabStatDiscount").textContent = Number(stats.discount_factor || 0.99).toFixed(3);
        if (document.getElementById("brainTabStatBatch")) document.getElementById("brainTabStatBatch").textContent = String(Number(stats.batch_size || 128).toFixed(0));

        const entSeries = Array.isArray(monitorData.policy_entropy) ? monitorData.policy_entropy
            : Array.isArray(monitorData.ent) ? monitorData.ent : [];
        const entLast = entSeries.length ? Number(entSeries[entSeries.length - 1]) : NaN;
        const entEl = document.getElementById("brainTabStatEntropy");
        if (entEl) entEl.textContent = Number.isFinite(entLast) ? entLast.toFixed(4) : "—";

        // Entropy sparkline
        const sparkCanvas = document.getElementById("brainEntropySparkline");
        if (sparkCanvas && entSeries.length >= 2) {
            const ctx = sparkCanvas.getContext("2d");
            const w = sparkCanvas.width, h = sparkCanvas.height;
            const pts = entSeries.slice(-30).map(Number).filter(Number.isFinite);
            if (pts.length >= 2) {
                const mn = Math.min(...pts), mx = Math.max(...pts);
                const rng = mx - mn || 1e-9;
                ctx.clearRect(0, 0, w, h);
                ctx.beginPath();
                pts.forEach((v, i) => {
                    const x = (i / (pts.length - 1)) * w;
                    const y = h - ((v - mn) / rng) * (h - 2) - 1;
                    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                });
                ctx.strokeStyle = pts[pts.length - 1] > pts[0] ? "#4ade80" : "#f87171";
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }
        
        const corr = monitorData.correlation || {};
        window.ChartUtils.upsertChart("brainCorrelationChart", {
            type: "bar",
            data: {
                labels: ["Sentiment/Price", "News Weight", "Price Weight"],
                datasets: [{ data: [corr.sentiment_price_correlation||0, corr.news_weight||0, corr.price_weight||0], backgroundColor: ["#ff4dff", "#39ff14", "#00f0ff"] }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
        
        this.renderSignalMixBars({}, monitorData);
    },

    renderSocialBuzz(stateData) {
        const el = document.getElementById("socialBuzzStrip");
        if (!el) return;
        const buzz = stateData && stateData.social_buzz;
        if (!buzz || !Array.isArray(buzz.lines) || !buzz.lines.length) return;

        // Build a stable cache key to skip identical renders
        const key = buzz.lines.slice(0, 5).map(l => `${l.market}|${l.velocity_pct_1h}|${l.regime}`).join(";");
        if (el._lastBuzzKey === key) return;
        el._lastBuzzKey = key;

        // Build DOM nodes (no innerHTML with server-supplied strings)
        const frag = document.createDocumentFragment();
        frag.append("Social Buzz: ");
        buzz.lines.slice(0, 5).forEach((l, i) => {
            if (i > 0) frag.append("  ·  ");
            const ticker = String(l.market || l.ticker || "").replace("-EUR", "");
            const vel    = Number(l.velocity_pct_1h || 0);
            const regime = String(l.regime || "").toLowerCase();
            const arrow  = vel > 1 ? "↑" : vel < -1 ? "↓" : "→";
            const text   = `${ticker} ${arrow}${Math.abs(vel).toFixed(1)}%`;
            if (regime === "high") {
                const strong = document.createElement("strong");
                strong.textContent = text;
                frag.append(strong);
            } else {
                frag.append(text);
            }
        });
        el.replaceChildren(frag);
    },

    renderShadowTrades(trades) {
        const tbody = document.getElementById("brainShadowTradesBody");
        if (!tbody) return;
        if (!trades || !trades.length) return;
        const REASON_LABELS = {
            volatility_filter_spread_too_high: "Spread te hoog",
            emergency_exit_negative_sentiment_shock: "Negatief sentiment noodstop",
            no_trade_signal: "Geen signaal",
            core_risk_check_safety: "Core risk: budget/exposure",
        };
        const rows = [...trades].reverse().map(t => {
            const sigColor = t.signal === "BUY" ? "#4ade80" : "#f87171";
            const reason = REASON_LABELS[t.reason] || this.escapeHtmlText(t.reason || "—");
            const extra = t.extra ? ` <span style="color:#64748b">(${this.escapeHtmlText(String(t.extra).slice(0, 60))})</span>` : "";
            const ts = String(t.ts || "").replace("T", " ").slice(0, 19);
            return `<tr style="border-bottom:1px solid #0f172a;">
                <td style="padding:4px 8px;color:#64748b;white-space:nowrap;">${this.escapeHtmlText(ts)}</td>
                <td style="padding:4px 8px;color:#e2e8f0;">${this.escapeHtmlText(String(t.market || "—").replace("-EUR",""))}</td>
                <td style="text-align:center;padding:4px 8px;color:${sigColor};font-weight:600;">${this.escapeHtmlText(t.signal || "—")}</td>
                <td style="padding:4px 8px;color:#cbd5e1;">${reason}${extra}</td>
                <td style="text-align:right;padding:4px 8px;color:#64748b;">${Number(t.spread_bps || 0).toFixed(1)}</td>
                <td style="text-align:right;padding:4px 8px;color:#64748b;">€${Number(t.price || 0).toLocaleString("nl-NL",{maximumFractionDigits:2})}</td>
            </tr>`;
        });
        tbody.innerHTML = rows.join("");
    },

    _renderThinkingSteps(steps) {
        const rb = document.getElementById("brainReasoningBox");
        if (!rb || !steps || !steps.length) return;
        const existing = rb.querySelector(".brain-reasoning-body");
        if (!existing) return;
        const tpId = "brainThinkingSteps";
        let tpEl = rb.querySelector(`#${tpId}`);
        if (!tpEl) {
            tpEl = document.createElement("ol");
            tpEl.id = tpId;
            tpEl.style.cssText = "margin:8px 0 0 0;padding:0 0 0 1.2em;font-size:0.78rem;color:#94a3b8;line-height:1.5;";
            existing.after(tpEl);
        }
        const frag = document.createDocumentFragment();
        steps.forEach(s => {
            const li = document.createElement("li");
            li.textContent = s;
            frag.append(li);
        });
        tpEl.replaceChildren(frag);
    },

    renderSecondaryCharts(monitorData, lagData) {
        if (!window.ChartUtils) return;
        const lineOpts = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } };
        const ep = monitorData.episode_length || monitorData.ep || [];
        const ent = monitorData.policy_entropy || monitorData.ent || [];

        // TD-Error chart (value_loss vs global training step)
        const nl = monitorData.network_logs || monitorData.nl || {};
        const tdVl = Array.isArray(nl.value_loss) ? nl.value_loss : (Array.isArray(nl.vl) ? nl.vl : []);
        const tdSteps = Array.isArray(nl.td_steps) ? nl.td_steps : tdVl.map((_, i) => i + 1);
        if (tdVl.length >= 2) {
            const tdFinite = tdVl.filter(v => Number.isFinite(v));
            const tdMin = tdFinite.length ? Math.min(...tdFinite) : 0;
            const tdMax = tdFinite.length ? Math.max(...tdFinite) : 1;
            const tdPad = Math.max((tdMax - tdMin) * 0.15, 1e-6);
            const stepLabels = tdSteps.map(s => s >= 1000 ? `${(s/1000).toFixed(0)}k` : String(s));
            window.ChartUtils.upsertChart("brainTdErrorChart", {
                type: "line",
                data: {
                    labels: stepLabels,
                    datasets: [{ label: "TD-Error", data: tdVl, borderColor: "#f59e0b", borderWidth: 2, pointRadius: 0, fill: false }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: { y: { min: tdMin - tdPad, max: tdMax + tdPad }, x: { ticks: { maxTicksLimit: 8, color: "#64748b" } } },
                    plugins: { legend: { display: false } }
                }
            });
        }
        
        window.ChartUtils.upsertChart("brainEpisodeChart", {
            type: "line", data: { labels: ep.map((_, i) => i + 1), datasets: [{ data: ep, borderColor: "#66e8ff", borderWidth: 4, pointRadius: 0 }] }, options: lineOpts
        });
        window.ChartUtils.upsertChart("brainEntropyChart", {
            type: "line", data: { labels: ent.map((_, i) => i + 1), datasets: [{ data: ent, borderColor: "#ff66ff", borderWidth: 4, pointRadius: 0 }] }, options: lineOpts
        });
    },

    _renderThinkingSections(sections) {
        const obsEl = document.getElementById("brainThinkingObservations");
        const confEl = document.getElementById("brainThinkingConflict");
        const verdEl = document.getElementById("brainThinkingVerdict");
        if (obsEl && sections.observations) obsEl.textContent = sections.observations;
        if (confEl && sections.conflict) confEl.textContent = sections.conflict;
        if (verdEl && sections.verdict) {
            verdEl.textContent = sections.verdict;
            this._lastAction = sections.verdict.split(" ")[0];
        }
    },

    _renderBuyBlockFactors(factors) {
        const el = document.getElementById("brainBuyBlockFactors");
        if (!el) return;
        if (!factors || !factors.length) { el.textContent = "Geen actieve blokkades."; return; }
        const frag = document.createDocumentFragment();
        factors.forEach(f => {
            const pill = document.createElement("span");
            pill.className = `brain-context-pill brain-context-pill--${f.status || "ok"}`;
            pill.title = f.detail || "";
            const lbl = document.createElement("span");
            lbl.className = "brain-context-pill__label";
            lbl.textContent = f.label;
            const val = document.createElement("span");
            val.className = "brain-context-pill__value";
            val.textContent = f.value;
            pill.append(lbl, val);
            frag.append(pill);
        });
        el.replaceChildren(frag);
    },

    _explainFeature(featureName, weight, obsVal) {
        const box = document.getElementById("featureImpactBox");
        if (!box) return;
        const obsStr = obsVal !== undefined && obsVal !== null
            ? `Huidige waarde: ${Number(obsVal).toFixed(4)}`
            : "Waarde onbekend in huidige observatie";
        const pct = (weight * 100).toFixed(1);
        const impact = weight > 0.15 ? "sterk" : weight > 0.08 ? "matig" : "zwak";
        const action = this._lastAction || "het huidige besluit";
        const frag = document.createDocumentFragment();
        const inner = document.createElement("div");
        inner.className = "brain-feature-impact-inner";
        const nameEl = document.createElement("strong");
        nameEl.className = "brain-feature-impact-name";
        nameEl.textContent = featureName;
        const weightEl = document.createElement("span");
        weightEl.className = "brain-feature-impact-weight";
        weightEl.textContent = `gewicht: ${pct}% — ${impact} signaal`;
        const explEl = document.createElement("p");
        explEl.className = "brain-feature-impact-explain";
        explEl.textContent = `${obsStr}. Dit feature draagt ${impact} bij aan ${action}.`;
        inner.append(nameEl, weightEl, explEl);
        frag.append(inner);
        box.replaceChildren(frag);
        box.style.display = "";
    }
};