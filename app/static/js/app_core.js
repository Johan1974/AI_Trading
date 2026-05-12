/** Nederlands: duizendtallen met punt, decimalen met komma (bv. € 1.234,56). */
window.formatEurNl = function formatEurNl(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "—";
    const neg = n < 0;
    const s = Math.abs(n).toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    return (neg ? "-€ " : "€ ") + s;
};

window.AppCore = {
    state: {
        activeTab: 'terminal',
        selectedMarket: 'BTC-EUR',
        balance: 0,
        cash: 0
    },
    sockets: {},
    /** Rolling minuut-evaluaties: voorspelde richting vs. gerealiseerde prijsbeweging. */
    _hitTracker: {
        minuteKey: null,
        openPx: null,
        predNext: null,
        lastPx: null,
        samples: [],
    },

    Logger: {
        isSending: false,
        init() {
            const originalError = console.error;
            const originalWarn = console.warn;

            const sendLog = (level, args) => {
                if (this.isSending) return;
                this.isSending = true; // No-Loop Guarantee
                try {
                    const message = args.map(a => typeof a === 'object' ? (a instanceof Error ? a.message : JSON.stringify(a)) : String(a)).join(' ');
                    const stacktrace = args.find(a => a instanceof Error)?.stack || "";

                    fetch('/api/v1/log/browser', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ level, message, stacktrace, url: window.location.href })
                    }).catch(() => {}).finally(() => {
                        this.isSending = false;
                    });
                } catch (e) {
                    // Negeer stringify fouten
                    this.isSending = false;
                }
            };

            console.error = (...args) => {
                originalError.apply(console, args);
                sendLog('ERROR', args);
            };

            console.warn = (...args) => {
                originalWarn.apply(console, args);
                sendLog('WARN', args);
            };

            window.addEventListener('error', (event) => sendLog('ERROR', [event.error || event.message]));
            window.addEventListener('unhandledrejection', (event) => sendLog('ERROR', [event.reason || 'Unhandled Promise Rejection']));
        }
    },

    init() {
        if (typeof window.runSequencedStartup === 'function' || typeof window.switchTab === 'function') {
            console.log("[Legacy AppCore] Bypassed because monolithic terminal.js is active.");
            wireCockpitTerminalMidSplitResizeObserver();
            void this.wirePositionSizingWidget();
            void this.wireTerminalResetPaperButton();
            const pollStatsForOverlay = () => {
                const sym = encodeURIComponent(
                    String(
                        (window.AppCore && window.AppCore.state && window.AppCore.state.selectedMarket) ||
                            window.selectedMarket ||
                            (document.getElementById("marketSelect") && document.getElementById("marketSelect").value) ||
                            "BTC-EUR"
                    )
                );
                fetch(`/api/v1/stats?symbol=${sym}`)
                    .then((r) => (r.ok ? r.json() : null))
                    .then((d) => {
                        if (d) document.dispatchEvent(new CustomEvent("ai-trading-stats", { detail: d }));
                    })
                    .catch(() => {});
            };
            pollStatsForOverlay();
            setInterval(pollStatsForOverlay, 5000);
            return;
        }
        this.Logger.init();
        this.bindNavigation();
        this.wirePositionSizingWidget();
        void this.wireTerminalResetPaperButton();
        this.connectSockets();
        this.switchTab('terminal'); // Default start
        wireCockpitTerminalMidSplitResizeObserver();
    },

    async wireTerminalResetPaperButton() {
        const wrap = document.querySelector(".bv-reset-paper-wrap");
        if (!wrap || wrap.dataset.wired === "1") return;
        wrap.dataset.wired = "1";
        const btn = document.getElementById("terminalResetPaperBtn");
        const recBtn = document.getElementById("terminalReconcilePaperBtn");

        const refreshAfterReset = () => {
            document.dispatchEvent(new CustomEvent("ai-trading-refresh-activity"));
            const sym = encodeURIComponent(
                String(
                    (window.AppCore && window.AppCore.state && window.AppCore.state.selectedMarket) ||
                        window.selectedMarket ||
                        (document.getElementById("marketSelect") && document.getElementById("marketSelect").value) ||
                        "BTC-EUR"
                )
            );
            fetch(`/api/v1/stats?symbol=${sym}`)
                .then((r) => (r.ok ? r.json() : null))
                .then((d) => {
                    if (d) document.dispatchEvent(new CustomEvent("ai-trading-stats", { detail: d }));
                })
                .catch(() => {});
        };

        if (recBtn) {
            recBtn.addEventListener("click", async () => {
                if (
                    !confirm(
                        "Alle open paper-posities wissen (spook-notional) en Saldo gelijkzetten aan huidige cash? De trade-ledger in de database blijft behouden."
                    )
                ) {
                    return;
                }
                recBtn.disabled = true;
                try {
                    const res = await fetch("/api/v1/paper/reconcile-balance", { method: "POST" });
                    const j = await res.json().catch(() => ({}));
                    if (!res.ok) {
                        alert(j.detail || res.statusText || "Corrigeren mislukt");
                        return;
                    }
                    if (j.queued) {
                        alert("Saldo-correctie gepland op de worker; over enkele seconden verversen de cijfers.");
                    } else if (j.equity_after != null) {
                        alert(`Saldo bijgewerkt: € ${Number(j.equity_after).toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} (cash).`);
                    }
                    refreshAfterReset();
                } catch (e) {
                    alert(e && e.message ? e.message : "Netwerkfout");
                } finally {
                    recBtn.disabled = false;
                }
            });
        }

        if (btn) {
            btn.addEventListener("click", async () => {
                if (
                    !confirm(
                        "Volledige paper-reset: startbedrag en trade-geschiedenis archiveren/wissen? Winstberekening begint weer bij nul."
                    )
                ) {
                    return;
                }
                btn.disabled = true;
                try {
                    const res = await fetch("/api/v1/reset-paper", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({}),
                    });
                    const j = await res.json().catch(() => ({}));
                    if (!res.ok) {
                        const d = j.detail;
                        let errTxt = j.detail || res.statusText || "Reset mislukt";
                        if (Array.isArray(d)) errTxt = d.map((x) => (x && x.msg) || JSON.stringify(x)).join("; ");
                        alert(errTxt);
                        return;
                    }
                    if (j.queued) {
                        alert("Reset gepland op de worker; over enkele seconden verversen de cijfers.");
                    }
                    refreshAfterReset();
                } catch (e) {
                    alert(e && e.message ? e.message : "Netwerkfout");
                } finally {
                    btn.disabled = false;
                }
            });
        }
    },

    async wirePositionSizingWidget() {
        const root = document.getElementById("positionSizingWidget");
        if (!root || root.dataset.wired === "1") return;
        root.dataset.wired = "1";

        const tabBtns = root.querySelectorAll("[data-pos-tab]");
        const panels = root.querySelectorAll("[data-pos-panel]");
        const inpFixed = document.getElementById("positionSizingInputFixed");
        const inpPct = document.getElementById("positionSizingInputPct");
        const btn = document.getElementById("positionSizingUpdateBtn");
        const fb = document.getElementById("positionSizingFeedback");
        const prFixed = document.getElementById("positionSizingOrderPreviewFixed");
        const prPctTab = document.getElementById("positionSizingOrderPreviewPctTab");
        const elCash = document.getElementById("positionSizingCashAvailable");
        const elMax = document.getElementById("positionSizingMaxBuys");

        let sizingActivityCache = null;

        const fmtEur = (v) => {
            const n = Number(v);
            return Number.isFinite(n)
                ? `€${n.toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                : "—";
        };

        const parseBadgeExposurePct = () => {
            const badgeEl = document.getElementById("hpDashAllocMainBadge");
            if (!badgeEl) return NaN;
            const t = String(badgeEl.textContent || "").replace("%", "").replace(/\s/g, "").replace(",", ".").trim();
            const n = parseFloat(t);
            return Number.isFinite(n) ? n : NaN;
        };

        const tweakAllocSummaryCopy = (raw) => {
            const ex = parseBadgeExposurePct();
            if (!Number.isFinite(ex) || ex > 0.01) return String(raw || "");
            let t = String(raw || "");
            t = t.replace(/\d+\/\d+\s+slots\s+bezet\s*\([^)]*\)\s*·\s*/gi, "");
            t = t.replace(/^Allocatie:\s*/i, "Cap & orders — ");
            return t.trim() || String(raw || "");
        };

        const applyExposureSummaryTweak = () => {
            const sumEl = document.getElementById("hpDashAllocSummary");
            if (sumEl && sumEl.textContent) sumEl.textContent = tweakAllocSummaryCopy(sumEl.textContent);
            const leg = document.getElementById("allocatie");
            if (leg && leg.textContent && !leg.querySelector("ul")) {
                leg.textContent = tweakAllocSummaryCopy(leg.textContent);
            }
        };

        const refreshSizingMeta = () => {
            const data = sizingActivityCache;
            const pp = data && data.paper_portfolio && typeof data.paper_portfolio === "object" ? data.paper_portfolio : {};
            const cash = Number(pp.cash != null ? pp.cash : pp.equity);
            if (elCash) elCash.textContent = Number.isFinite(cash) && cash > 0 ? fmtEur(cash) : "—";

            const active = root.querySelector(".hp-position-sizing-tab.is-active");
            const tab = (active && active.getAttribute("data-pos-tab")) || "fixed";
            if (tab === "pct") {
                if (prFixed) prFixed.textContent = "";
                const eq = Number(pp.equity);
                const p = parseFloat((inpPct && inpPct.value) || "0");
                let orderEur = 0;
                if (Number.isFinite(eq) && Number.isFinite(p) && eq > 0) orderEur = (eq * p) / 100;
                if (prPctTab && Number.isFinite(p)) {
                    prPctTab.textContent = `Bot zal ${p.toFixed(2)}% van je equity inzetten per trade.`;
                } else if (prPctTab) prPctTab.textContent = "";
                if (elMax && Number.isFinite(cash) && cash > 0 && orderEur > 0) {
                    elMax.textContent = `~${Math.floor(cash / orderEur)} trades`;
                } else if (elMax) elMax.textContent = "—";
                return;
            }
            if (prPctTab) prPctTab.textContent = "";
            const orderEur = parseFloat((inpFixed && inpFixed.value) || "0");
            if (prFixed && Number.isFinite(cash) && cash > 0 && Number.isFinite(orderEur) && orderEur > 0) {
                const pct = (orderEur / cash) * 100;
                prFixed.textContent = `Bot zal ${pct.toFixed(2)}% van je cash inzetten per trade.`;
            } else if (prFixed) prFixed.textContent = "";
            if (elMax && Number.isFinite(cash) && cash > 0 && Number.isFinite(orderEur) && orderEur > 0) {
                elMax.textContent = `~${Math.floor(cash / orderEur)} trades`;
            } else if (elMax) elMax.textContent = "—";
        };

        const setTab = (t) => {
            const tab = t === "pct" ? "pct" : "fixed";
            tabBtns.forEach((b) => {
                const on = (b.getAttribute("data-pos-tab") || "") === tab;
                b.classList.toggle("is-active", on);
                b.setAttribute("aria-selected", on ? "true" : "false");
            });
            panels.forEach((p) => {
                const id = p.getAttribute("data-pos-panel") || "";
                p.style.display = id === tab ? "block" : "none";
            });
            refreshSizingMeta();
        };

        tabBtns.forEach((b) => {
            b.addEventListener("click", () => setTab(b.getAttribute("data-pos-tab") || "fixed"));
        });
        if (inpFixed) inpFixed.addEventListener("input", () => refreshSizingMeta());
        if (inpPct) inpPct.addEventListener("input", () => refreshSizingMeta());

        const syncFromActivity = async () => {
            try {
                const url =
                    typeof window.buildActivityFetchUrl === "function" ? window.buildActivityFetchUrl() : "/activity";
                const init = window.activityFetchInit || { cache: "no-store", credentials: "same-origin" };
                const res = await fetch(url, init);
                if (!res.ok) return;
                const data = await res.json();
                sizingActivityCache = data;
                const rp = data.risk_profile;
                if (rp && typeof rp === "object") {
                    const mode = String(rp.sizing_mode || "fixed_eur").toLowerCase();
                    if (mode === "equity_pct") {
                        setTab("pct");
                        if (inpPct && rp.max_risk_pct != null) inpPct.value = String(rp.max_risk_pct);
                    } else {
                        setTab("fixed");
                        if (inpFixed && rp.base_trade_eur != null) inpFixed.value = String(rp.base_trade_eur);
                    }
                } else {
                    refreshSizingMeta();
                }
            } catch (_) {
                /* ignore */
            }
        };

        if (btn) {
            btn.addEventListener("click", async () => {
                const active = root.querySelector(".hp-position-sizing-tab.is-active");
                const tab = (active && active.getAttribute("data-pos-tab")) || "fixed";
                const raw =
                    tab === "pct"
                        ? (inpPct && inpPct.value) || ""
                        : (inpFixed && inpFixed.value) || "";
                const value = parseFloat(String(raw).replace(",", "."));
                if (!Number.isFinite(value)) {
                    if (fb) fb.textContent = "Voer een geldig getal in.";
                    return;
                }
                if (fb) fb.textContent = "Bezig…";
                try {
                    const res = await fetch("/api/v1/settings", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ position_sizing_tab: tab, value }),
                    });
                    const j = await res.json().catch(() => ({}));
                    if (!res.ok) {
                        let errTxt = j.error || res.statusText || "Fout";
                        const d = j.detail;
                        if (Array.isArray(d)) errTxt = d.map((x) => (x && x.msg) || JSON.stringify(x)).join("; ");
                        else if (d != null) errTxt = String(d);
                        if (fb) fb.textContent = errTxt;
                        return;
                    }
                    if (fb) fb.textContent = "Opgeslagen.";
                    document.dispatchEvent(new CustomEvent("ai-trading-refresh-activity"));
                } catch (e) {
                    if (fb) fb.textContent = e && e.message ? e.message : "Netwerkfout";
                }
            });
        }

        if (!window.__hpExposureAllocHooks) {
            window.__hpExposureAllocHooks = true;
            const runTweak = () => requestAnimationFrame(() => applyExposureSummaryTweak());
            ["hpDashAllocSummary", "hpDashAllocMainBadge"].forEach((id) => {
                const el = document.getElementById(id);
                if (!el || el.dataset.hpAllocObs) return;
                el.dataset.hpAllocObs = "1";
                new MutationObserver(runTweak).observe(el, { characterData: true, subtree: true, childList: true });
            });
            document.addEventListener("ai-trading-stats", (ev) => {
                const d = ev && ev.detail;
                if (!d || typeof d !== "object") return;
                sizingActivityCache = { ...(sizingActivityCache || {}), ...d };
                if (d.paper_portfolio && typeof d.paper_portfolio === "object") {
                    sizingActivityCache.paper_portfolio = d.paper_portfolio;
                }
                refreshSizingMeta();
                runTweak();
            });
            document.addEventListener("ai-trading-refresh-activity", () => {
                void syncFromActivity().then(() => runTweak());
            });
        }

        await syncFromActivity();
        refreshSizingMeta();
        requestAnimationFrame(() => applyExposureSummaryTweak());
    },
    
    bindNavigation() {
        // Tab routing
        ['terminal', 'aibrain', 'ledger', 'logs', 'sentiment'].forEach((tab) => {
            const btn = document.getElementById(`btn-${tab}`);
            if (btn) {
                btn.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.switchTab(tab);
                });
            }
        });

        // Market selection
        const marketSelect = document.getElementById("marketSelect");
        if (marketSelect) {
            marketSelect.addEventListener("change", (e) => {
                this.state.selectedMarket = e.target.value;
                const headerPair = document.getElementById("headerPairDisplay");
                if (headerPair) headerPair.textContent = this.state.selectedMarket;
                if (typeof window.syncElite8AssetToolbarSelection === "function") {
                    window.syncElite8AssetToolbarSelection();
                }

                // Notificeer actieve views over de nieuwe markt
                if (window.ModuleTerminal) window.ModuleTerminal.refresh();
                if (window.ModuleBrain) window.ModuleBrain.refresh();
                if (typeof window.__reloadTerminalPredictionChart === "function") {
                    window.__reloadTerminalPredictionChart();
                }
            });
        }

        const ledgerTabs = document.querySelector(".ledger-footer-tabs");
        if (ledgerTabs) {
            ledgerTabs.addEventListener("click", (e) => {
                const btn = e.target && e.target.closest ? e.target.closest("[data-ledger-tab]") : null;
                if (!btn || !window.ModuleLedger || typeof window.ModuleLedger.setLedgerView !== "function") return;
                const tab = btn.getAttribute("data-ledger-tab");
                if (!tab) return;
                window.ModuleLedger.setLedgerView(tab);
                ledgerTabs.querySelectorAll("[data-ledger-tab]").forEach((b) => {
                    const on = b === btn;
                    b.classList.toggle("is-active", on);
                    b.setAttribute("aria-selected", on ? "true" : "false");
                });
            });
        }

        document.getElementById("btnSystemLogsFromLedger")?.addEventListener("click", (e) => {
            e.preventDefault();
            const top = document.getElementById("btn-logs");
            if (top) top.click();
            else this.switchTab("logs");
        });
    },
    
    switchTab(tabName) {
        const prevTab = this.state.activeTab;
        this.state.activeTab = tabName;

        if (prevTab === 'hardware' && tabName !== 'hardware' && window.ModuleHardware && typeof window.ModuleHardware.stop === 'function') {
            window.ModuleHardware.stop();
        }
        if (prevTab === 'logs' && tabName !== 'logs' && window.TerminalLiveTail && typeof window.TerminalLiveTail.onLogsTabHidden === 'function') {
            window.TerminalLiveTail.onLogsTabHidden();
        }
        
        // Reset UI — .hidden heeft display:none !important met hoge specificiteit, wint van terminal's display:flex !important
        ['terminal', 'aibrain', 'ledger', 'hardware', 'logs', 'sentiment'].forEach(name => {
            const el = document.getElementById(`tab-${name}`);
            const btn = document.getElementById(`btn-${name}`);
            if (el) el.classList.add("hidden");
            if (btn) btn.classList.remove("active");
        });

        // Activeer nieuwe tab
        const activeEl = document.getElementById(`tab-${tabName}`);
        const activeBtn = document.getElementById(`btn-${tabName}`);
        if (activeEl) {
            activeEl.classList.remove("hidden");
            activeEl.style.display = "flex";
        }
        if (activeBtn) activeBtn.classList.add("active");
        
        // Tab-Isolatie: Cleanup grafieken en stop sockets
        if (window.ChartUtils) window.ChartUtils.clearAllCharts();
        if (tabName !== 'aibrain' && window.ModuleBrain && typeof window.ModuleBrain.stopSocket === 'function') window.ModuleBrain.stopSocket();
        if (prevTab === 'terminal' && window.ModuleTerminal && typeof window.ModuleTerminal.onDeactivate === 'function') window.ModuleTerminal.onDeactivate();
        const brainHintPortal = document.getElementById("brainHintPortal");
        if (brainHintPortal) brainHintPortal.style.display = "none";
        const newsModal = document.getElementById("newsModal");
        if (newsModal) newsModal.classList.add("hidden");

        // Dispatch activation signal naar de geïsoleerde module
        if (tabName === 'terminal' && window.ModuleTerminal) window.ModuleTerminal.onActivate();
        if (tabName === 'ledger' && window.ModuleLedger) window.ModuleLedger.onActivate();
        if (tabName === 'aibrain' && window.ModuleBrain) window.ModuleBrain.onActivate();
        if (tabName === 'hardware' && window.ModuleHardware && typeof window.ModuleHardware.onActivate === 'function') {
            window.ModuleHardware.onActivate();
        }
        if (tabName === 'logs' && window.TerminalLiveTail && typeof window.TerminalLiveTail.onLogsTabShown === 'function') {
            window.TerminalLiveTail.onLogsTabShown();
        }
        if (tabName === 'sentiment') window._loadSentimentTab();
    },
    
    connectSockets() {
        // Live trading/redis UI: `module_feeds.js` (/ws/trading-updates). Geen dubbele socket hier.
    },

    applyPortfolioFromStats(stats) {
        if (!stats || typeof stats !== "object") return;
        const p = stats.paper_portfolio && typeof stats.paper_portfolio === "object" ? stats.paper_portfolio : null;
        if (!p) return;
        const equity = Number(p.equity);
        const cash = Number(p.cash);
        const fmtEur = (n) =>
            Number.isFinite(n) ? `€ ${n.toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "—";

        if (Number.isFinite(equity)) this.state.balance = equity;
        if (Number.isFinite(cash)) this.state.cash = cash;

        // Header + sidebar uit exact dezelfde /api/v1/stats payload voeden.
        ["total-balance", "headerBalanceEuro"].forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.textContent = Number.isFinite(equity) ? fmtEur(equity) : "—";
        });
        const headerCash = document.getElementById("headerCash");
        if (headerCash) headerCash.textContent = Number.isFinite(cash) ? fmtEur(cash) : "—";

        document.querySelectorAll(".js-portfolio-total").forEach((el) => {
            el.textContent = Number.isFinite(equity) ? fmtEur(equity) : "—";
        });
        document.querySelectorAll(".js-portfolio-cash").forEach((el) => {
            el.textContent = Number.isFinite(cash) ? fmtEur(cash) : "—";
        });
        const hta = document.getElementById("hpDashTotalAsset");
        if (hta) hta.textContent = Number.isFinite(equity) ? fmtEur(equity) : "—";
        const hca = document.getElementById("hpDashCashAvailable");
        if (hca) hca.textContent = Number.isFinite(cash) ? fmtEur(cash) : "—";
    },

    applySystemStatusFromStats(stats) {
        if (!stats || typeof stats !== "object") return;
        const setDot = (id, state) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.classList.remove("is-ok", "is-error", "is-loading");
            el.classList.add(state === "ok" ? "is-ok" : state === "error" ? "is-error" : "is-loading");
        };

        const workerRaw = String(stats.worker_status || "loading").toLowerCase();
        const gpu = String(stats.gpu_status || "loading").toLowerCase();
        const db = stats.db_connected === true ? "ok" : stats.db_connected === false ? "error" : "loading";
        const lastInf = String(stats.last_inference_time || "").trim();
        const probs = stats.ai_action_probs && typeof stats.ai_action_probs === "object" ? stats.ai_action_probs : {};
        const hasMultiPolicyProbs = (() => {
            const m = stats.rl_multi_decisions;
            if (!m || typeof m !== "object") return false;
            for (const k of Object.keys(m)) {
                const d = m[k];
                if (!d || typeof d !== "object") continue;
                if ([d.prob_buy, d.prob_hold, d.prob_sell].some((v) => Number.isFinite(Number(v)) && Math.abs(Number(v)) > 1e-6)) return true;
            }
            return false;
        })();
        const hasLivePolicy =
            [probs.buy_pct, probs.hold_pct, probs.sell_pct].some((v) => Number.isFinite(Number(v)) && Number(v) > 0.01) ||
            (() => {
                const d = stats.rl_last_decision;
                if (!d || typeof d !== "object") return false;
                return [d.prob_buy, d.prob_hold, d.prob_sell].some((v) => Number.isFinite(Number(v)) && Number(v) > 0.0001);
            })() ||
            hasMultiPolicyProbs;
        const predFresh = stats.prediction_fresh === true;
        const hasAnyFlowSignal = predFresh || db === "ok" || lastInf.length > 0;
        const botAppearsRunning = (() => {
            const bs = String(stats.bot_status || "").toLowerCase();
            if (!bs) return false;
            return bs !== "paused" && bs !== "panic_stop" && bs !== "stopped" && bs !== "stop";
        })();
        const hasWorkerHints =
            (Array.isArray(stats.worker_calc_hints) &&
                stats.worker_calc_hints.some((x) => String(x || "").trim().length > 4)) ||
            (stats.worker_calc_hints_by_market &&
                typeof stats.worker_calc_hints_by_market === "object" &&
                Object.keys(stats.worker_calc_hints_by_market).length > 0);
        const workerStopped = workerRaw === "offline" || workerRaw === "stopped" || workerRaw === "dead";
        const worker = hasLivePolicy ? "online" : workerRaw;
        const aiState = hasLivePolicy ? "ok" : workerStopped ? "error" : "loading";

        setDot("termDotAiEngine", aiState);
        setDot("termDotGpuStatus", gpu === "ok" ? "ok" : gpu === "error" ? "error" : "loading");
        setDot("termDotLiveData", db);

        const txtAi = document.getElementById("termTxtAiEngine");
        const txtGpu = document.getElementById("termTxtGpuStatus");
        const txtData = document.getElementById("termTxtLiveData");
        if (txtAi) {
            if (hasLivePolicy) txtAi.textContent = "GPU OK";
            else if (aiState === "error") txtAi.textContent = "AI Engine Error";
            else if (botAppearsRunning && (hasWorkerHints || hasAnyFlowSignal)) txtAi.textContent = "Thinking...";
            else txtAi.textContent = "AI Engine ...";
        }
        if (txtGpu) txtGpu.textContent = gpu === "ok" ? "GPU OK" : gpu === "error" ? "GPU Error" : "GPU ...";
        if (txtData) txtData.textContent = db === "ok" ? "Live Data OK" : db === "error" ? "Live Data Error" : "Live Data ...";

        const hud = document.getElementById("hpMiniHud");
        const box = document.getElementById("cockpitTerminalHwMini");
        const warn = document.getElementById("hpMiniWarnIcon");
        const cpu = Number(stats.cpu_load);
        const ram = Number(stats.ram_usage);
        const temp = Number(stats.gpu_temp);
        if (hud) {
            const cpuTxt = Number.isFinite(cpu) ? `${cpu.toFixed(0)}%` : "—";
            const ramTxt = Number.isFinite(ram) ? `${ram.toFixed(0)}%` : "—";
            const tmpTxt = Number.isFinite(temp) && temp > 0 ? `${temp.toFixed(0)}°C` : "—";
            hud.textContent = `CPU ${cpuTxt} | RAM ${ramTxt} | TMP ${tmpTxt}`;
        }
        const gpuErr = gpu === "error";
        if (box) box.classList.toggle("hp-hw-mini-monitor--alert", gpuErr);
        if (warn) warn.textContent = gpuErr ? "⚠" : "●";

        const sent = Number(stats.sentiment_score);
        const sentEl = document.getElementById("market-sentiment-score");
        if (sentEl) sentEl.textContent = Number.isFinite(sent) ? sent.toFixed(3) : "--";

        // Compact debug hint for operators.
        if (lastInf && txtAi && worker === "online") {
            txtAi.title = `Last inference: ${lastInf}`;
        }
    },

    onStatsForPrediction(data) {
        const el = document.getElementById("hpLiveHitRate");
        if (!data || typeof data !== "object") return;
        const px = Number(data.price);
        if (!Number.isFinite(px) || px <= 0) return;

        const predNext = Number(data.predicted_next_close);
        const t = this._hitTracker;
        const mk = Math.floor(Date.now() / 60000);

        if (t.minuteKey === null) {
            t.minuteKey = mk;
            t.openPx = px;
            t.predNext = Number.isFinite(predNext) ? predNext : px;
            t.lastPx = px;
            this._renderLiveHitRate(el, t);
            return;
        }

        if (mk !== t.minuteKey) {
            const open = t.openPx;
            const close = t.lastPx;
            const pred = t.predNext;
            if (Number.isFinite(open) && Number.isFinite(close) && Number.isFinite(pred) && open > 0) {
                const predDir = Math.sign(pred - open);
                const actDir = Math.sign(close - open);
                if (predDir !== 0 && actDir !== 0) {
                    t.samples.push(predDir === actDir);
                    if (t.samples.length > 120) t.samples.shift();
                }
            }
            t.minuteKey = mk;
            t.openPx = px;
            t.predNext = Number.isFinite(predNext) ? predNext : px;
            t.lastPx = px;
        } else {
            t.lastPx = px;
            if (!Number.isFinite(t.predNext) || t.predNext <= 0) {
                t.predNext = Number.isFinite(predNext) ? predNext : px;
            }
        }
        this._renderLiveHitRate(el, t);
    },

    _renderLiveHitRate(el, t) {
        if (!el) return;
        if (!t.samples.length) {
            el.textContent = "Hit: —";
            return;
        }
        const hits = t.samples.filter(Boolean).length;
        const pct = (hits / t.samples.length) * 100;
        el.textContent = `Hit: ${pct.toFixed(0)}% (${t.samples.length}m)`;
    },
};

(function hookAiTradingStats() {
    if (window.__aiTradingStatsHooked) return;
    window.__aiTradingStatsHooked = true;
    document.addEventListener("ai-trading-stats", (ev) => {
        const d = ev && ev.detail;
        if (d && typeof d === "object") window.__lastDashboardStats = d;
        if (d && typeof d === "object" && d.market) {
            const mk = String(d.market).toUpperCase().replace("/", "-");
            if (window.AppCore && window.AppCore.state) {
                window.AppCore.state.selectedMarket = mk;
            }
            const sel = document.getElementById("marketSelect");
            if (sel && sel.value !== mk) sel.value = mk;
            if (typeof window.syncElite8AssetToolbarSelection === "function") {
                window.syncElite8AssetToolbarSelection();
            }
        }
        if (window.AppCore && typeof window.AppCore.applyPortfolioFromStats === "function") {
            window.AppCore.applyPortfolioFromStats(d);
        }
        if (window.AppCore && typeof window.AppCore.applySystemStatusFromStats === "function") {
            window.AppCore.applySystemStatusFromStats(d);
        }
        if (typeof window.updateTerminalHealthFromStats === "function") {
            window.updateTerminalHealthFromStats(d);
        }
        if (window.AppCore && typeof window.AppCore.onStatsForPrediction === "function") {
            window.AppCore.onStatsForPrediction(d);
        }
        if (window.ModuleTerminal && typeof window.ModuleTerminal.applyPredictionFromStats === "function") {
            window.ModuleTerminal.applyPredictionFromStats(d);
        }
        if (typeof window.paintAiPolicyProbsFromPayload === "function" && d && typeof d === "object") {
            window.paintAiPolicyProbsFromPayload(d);
        }
        if (typeof window.syncTerminalPredictionChartFromPolicyStats === "function" && d && typeof d === "object") {
            window.syncTerminalPredictionChartFromPolicyStats(d);
        }
        if (typeof window.applyHpAiProbThresholdMarkers === "function" && d && typeof d === "object") {
            window.applyHpAiProbThresholdMarkers(d);
        }
        if (typeof window.setChartTimeHint === "function" && d && typeof d === "object") {
            const sel = document.getElementById("marketSelect");
            const domMk = sel && sel.value ? String(sel.value).trim() : "";
            const mkU = String(
                d.market || domMk || (window.AppCore && window.AppCore.state && window.AppCore.state.selectedMarket) || "BTC-EUR"
            )
                .toUpperCase()
                .replace("/", "-");
            window.setChartTimeHint({
                market: mkU,
                updated_at: d.last_update || d.last_engine_tick_utc || null,
                predicted_at: (d.last_prediction && d.last_prediction.generated_at) || null,
            });
        }
    });
})();

function hpNormalizeProbToPercent(x) {
    const n = Number(x);
    if (!Number.isFinite(n) || n < 0) return null;
    return n <= 1.0 + 1e-9 ? n * 100.0 : Math.min(100, n);
}

function hpHasAnyPositivePolicyProb(stats) {
    if (!stats || typeof stats !== "object") return false;
    const ap = stats.ai_action_probs;
    if (ap && typeof ap === "object") {
        for (const k of ["buy_pct", "hold_pct", "sell_pct"]) {
            const p = hpNormalizeProbToPercent(ap[k]);
            if (p != null && p > 0.0001) return true;
        }
    }
    const d = stats.rl_last_decision;
    if (d && typeof d === "object") {
        for (const k of ["prob_buy", "prob_hold", "prob_sell", "buy", "hold", "sell"]) {
            const p = hpNormalizeProbToPercent(d[k]);
            if (p != null && p > 0.0001) return true;
        }
    }
    return false;
}

function hpDominantPolicyConfidence(st) {
    if (!st || typeof st !== "object") return null;
    let b = null;
    let h = null;
    let s = null;
    const ap = st.ai_action_probs;
    if (ap && typeof ap === "object") {
        b = hpNormalizeProbToPercent(ap.buy_pct);
        h = hpNormalizeProbToPercent(ap.hold_pct);
        s = hpNormalizeProbToPercent(ap.sell_pct);
    }
    if (b == null && h == null && s == null) {
        const d = st.rl_last_decision;
        if (d && typeof d === "object") {
            b = hpNormalizeProbToPercent(d.prob_buy);
            h = hpNormalizeProbToPercent(d.prob_hold);
            s = hpNormalizeProbToPercent(d.prob_sell);
        }
    }
    const bb = b == null ? 0 : b;
    const hh = h == null ? 0 : h;
    const ss = s == null ? 0 : s;
    if (!bb && !hh && !ss) return null;
    const max = Math.max(bb, hh, ss);
    let label = "HOLD";
    if (ss === max && s != null) label = "SELL";
    else if (bb === max && b != null) label = "BUY";
    else     if (hh === max && h != null) label = "HOLD";
    return { max, label };
}

/** Zelfde schaal als RL_ACTION_MIN_CONFIDENCE / reasoning (0..1), niet de hoogste policy-balk. */
function hpRlTradeConfidence01(st, mkU) {
    if (!st || typeof st !== "object") return NaN;
    const m = String(mkU || "")
        .trim()
        .toUpperCase()
        .replace("/", "-");
    let c = Number(st.rl_confidence);
    if (Number.isFinite(c) && c >= 0 && c <= 1.000001) return c;
    if (Number.isFinite(c) && c > 1) return c / 100;
    const multi = st.rl_multi_decisions;
    if (multi && typeof multi === "object" && m && m !== "—") {
        let row = multi[m];
        if (!row) {
            const keys = Object.keys(multi);
            for (let i = 0; i < keys.length; i++) {
                const ku = String(keys[i] || "")
                    .trim()
                    .toUpperCase()
                    .replace("/", "-");
                if (ku === m) {
                    row = multi[keys[i]];
                    break;
                }
            }
        }
        if (row && typeof row === "object" && row.confidence != null) {
            c = Number(row.confidence);
            if (Number.isFinite(c)) return c > 1.000001 ? c / 100 : c;
        }
    }
    const gl = st.rl_last_decision;
    if (gl && typeof gl === "object") {
        const t = String(gl.ticker || gl.market || "")
            .trim()
            .toUpperCase()
            .replace("/", "-");
        if (t && t === m && gl.confidence != null) {
            c = Number(gl.confidence);
            if (Number.isFinite(c)) return c > 1.000001 ? c / 100 : c;
        }
    }
    return NaN;
}

function hpActionThreshold01FromStats(st) {
    const t = Number(st && st.rl_decision_threshold_pct);
    if (!Number.isFinite(t) || t < 0) return 0.55;
    return t <= 1.0 + 1e-6 ? t : t / 100;
}

function hpCockpitSelectedMarketKey() {
    let domMk = "";
    try {
        const el = typeof document !== "undefined" ? document.getElementById("marketSelect") : null;
        if (el && el.value) domMk = String(el.value);
    } catch (_e) {
        /* ignore */
    }
    return String(
        (window.AppCore && window.AppCore.state && window.AppCore.state.selectedMarket) ||
            window.selectedMarket ||
            domMk ||
            "BTC-EUR"
    )
        .toUpperCase()
        .replace("/", "-");
}

/** Ghost/chart-sync: volg cockpit-selectie; laat stats-`market` (worker-focus) de UI niet blokkeren. */
function hpChartSyncMarketAligned(stats) {
    const sel = hpCockpitSelectedMarketKey();
    const last = window.__lastApiPredictionResponse;
    const predSym =
        last && last.symbol != null ? String(last.symbol).toUpperCase().replace("/", "-") : "";
    if (predSym && predSym === sel) return true;
    const sm = stats && stats.market != null ? String(stats.market).toUpperCase().replace("/", "-") : "";
    if (!sm) return true;
    return sm === sel;
}

window.__buildMergedPredictionPayloadForChart = function buildMergedPredictionPayloadForChart(stats) {
    const last = window.__lastApiPredictionResponse;
    const selKey = hpCockpitSelectedMarketKey();
    window.__hpLastMergedByMarket = window.__hpLastMergedByMarket || {};
    const out = { historical: [], predicted: [], prediction_data: [], accuracy_score: undefined };
    if (last && last.accuracy_score != null && Number.isFinite(Number(last.accuracy_score))) {
        out.accuracy_score = Number(last.accuracy_score);
    }
    if (last && Array.isArray(last.historical) && last.historical.length) {
        out.historical = last.historical
            .map((item) => {
                if (!item || typeof item !== "object") return null;
                const ts = item.timestamp;
                const y = Number(item.price);
                if (ts == null || !Number.isFinite(y)) return null;
                return { timestamp: ts, price: y };
            })
            .filter(Boolean);
    }
    const px = Number(stats && stats.price);
    if (!out.historical.length && Number.isFinite(px) && px > 0) {
        const now = Date.now();
        out.historical = [
            { timestamp: new Date(now - 20 * 60_000).toISOString(), price: px },
            { timestamp: new Date(now).toISOString(), price: px },
        ];
    }
    if (last && Array.isArray(last.predicted) && last.predicted.length) {
        out.predicted = last.predicted.slice();
        out.prediction_data = out.predicted.slice();
    } else if (stats && Array.isArray(stats.predicted_price) && stats.predicted_price.length >= 2 && out.historical.length) {
        const lh = out.historical[out.historical.length - 1];
        const tx = new Date(lh.timestamp).getTime();
        const stepMs = 5 * 60_000;
        const leadBars = Math.max(1, Math.min(12, Number(stats.prediction_bar_lead) || 2));
        let t = tx + stepMs * leadBars;
        for (let i = 0; i < stats.predicted_price.length; i++) {
            const y = Number(stats.predicted_price[i]);
            if (!Number.isFinite(y)) continue;
            out.predicted.push({ timestamp: new Date(t).toISOString(), predicted_price: y });
            t += stepMs;
        }
        out.prediction_data = out.predicted.slice();
    }
    if (!out.predicted.length && out.historical.length && stats) {
        const lh = out.historical[out.historical.length - 1];
        const lcRaw = Number(stats.predicted_latest_close);
        const lc = Number.isFinite(lcRaw) && lcRaw > 0 ? lcRaw : Number(lh.price);
        let pn = Number(stats.predicted_next_close);
        if (!Number.isFinite(pn) || pn <= 0) pn = lc;
        if (!Number.isFinite(lc) || lc <= 0) return out;
        if (Math.abs(pn - lc) < lc * 1e-9) pn = lc * 1.0004;
        const tx = new Date(lh.timestamp).getTime();
        for (let i = 1; i <= 6; i++) {
            const y = lc + (pn - lc) * (i / 6);
            out.predicted.push({ timestamp: new Date(tx + i * 5 * 60_000).toISOString(), predicted_price: y });
        }
        out.prediction_data = out.predicted.slice();
    }
    if (!out.predicted.length && window.__hpLastMergedByMarket[selKey] && Array.isArray(window.__hpLastMergedByMarket[selKey])) {
        out.predicted = window.__hpLastMergedByMarket[selKey].map((row) => ({ ...row }));
        out.prediction_data = out.predicted.slice();
    }
    if (out.predicted.length) {
        window.__hpLastMergedByMarket[selKey] = out.predicted.map((row) => ({ ...row }));
    }
    return out;
};

window.syncTerminalPredictionChartFromPolicyStats = function syncTerminalPredictionChartFromPolicyStats(stats) {
    if (!hpHasAnyPositivePolicyProb(stats)) return;
    if (!hpChartSyncMarketAligned(stats)) return;
    const sel = hpCockpitSelectedMarketKey();
    const last = window.__lastApiPredictionResponse;
    if (last && Array.isArray(last.predicted) && last.predicted.length > 0) {
        const sym = String(last.symbol || "").toUpperCase().replace("/", "-");
        if (sym === sel) return;
    }
    const merged = window.__buildMergedPredictionPayloadForChart(stats);
    if (!merged.historical || !merged.historical.length) return;
    updateTerminalChartJsPredictionDataset(merged);
};

window.applyHpAiProbThresholdMarkers = function applyHpAiProbThresholdMarkers(data) {
    const st = data && typeof data === "object" ? data : window.__lastDashboardStats || {};
    let th = Number(st.signal_threshold_pct);
    if (!Number.isFinite(th)) th = Number(st.rl_decision_threshold_pct);
    if (!Number.isFinite(th)) th = 55;
    th = Math.max(0, Math.min(99.5, th));
    document.querySelectorAll("#bvAiFeedCard .bv-ai-prob-bar-wrap").forEach((el) => {
        el.style.setProperty("--hp-ai-threshold-pct", `${th}%`);
    });
};

/** Terminal: AI Brain-icoon boven voorspellingsgrafiek — groene gloed bij sterke RL-confidence (>0.50). */
window.syncHpAiBrainConfidenceGlow = function syncHpAiBrainConfidenceGlow(data) {
    const wrap = document.getElementById("hpAiBrainPulseWrap");
    const readout = document.getElementById("hpAiBrainConfReadout");
    if (!wrap) return;
    const mk = hpCockpitSelectedMarketKey();
    const merged = {
        ...(window.__lastDashboardStats && typeof window.__lastDashboardStats === "object" ? window.__lastDashboardStats : {}),
        ...(data && typeof data === "object" ? data : {}),
    };
    const c = hpRlTradeConfidence01(merged, mk);
    if (!Number.isFinite(c)) {
        wrap.classList.remove("hp-ai-brain-pulse--strong");
        if (readout) readout.textContent = "";
        return;
    }
    const c01 = Math.min(1, Math.max(0, c));
    wrap.classList.toggle("hp-ai-brain-pulse--strong", c01 > 0.5);
    if (readout) readout.textContent = `conf ${c01.toFixed(2)}`;
};

/** Actieve berekeningen: periodieke heartbeat (naast worker_calc_hints uit activity). */
(function hookRlCalcHeartbeat() {
    if (window.__rlCalcHeartbeatHooked) return;
    window.__rlCalcHeartbeatHooked = true;
    const tick = () => {
        const hbSlot = document.getElementById("hpRlHeartbeatSlot");
        const root = document.getElementById("bvAiThoughtFeed");
        const host = hbSlot || root;
        if (!host) return;
        const st = window.__lastDashboardStats || {};
        const sel = document.getElementById("marketSelect")?.value;
        const mk = String(sel || window.AppCore?.state?.selectedMarket || "—")
            .toUpperCase()
            .replace("/", "-");
        const dom = hpDominantPolicyConfidence(st);
        const c01 = hpRlTradeConfidence01(st, mk);
        const th01 = hpActionThreshold01FromStats(st);
        let msg;
        if (Number.isFinite(c01) && c01 >= 0) {
            msg = `RL-Agent (${mk}): live… actie-confidence ${c01.toFixed(2)} (drempel ${th01.toFixed(2)})`;
            if (dom != null) msg += ` · policy ${dom.label} ${dom.max.toFixed(0)}%`;
        } else if (dom != null) {
            msg = `RL-Agent (${mk}): live… policy ${dom.label} ${dom.max.toFixed(0)}%`;
        } else {
            msg = `RL-Agent (${mk}): candles & policy…`;
        }
        const ts = new Intl.DateTimeFormat("nl-NL", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: false,
        }).format(new Date());
        let row = document.getElementById("hpRlHeartbeatRow");
        if (!row) {
            row = document.createElement("div");
            row.id = "hpRlHeartbeatRow";
            row.className = "bv-ai-trade-row bv-ai-trade-row--neu bv-ai-trade-row--rl-heartbeat";
            row.setAttribute("role", "status");
            host.appendChild(row);
        } else {
            if (hbSlot && row.parentElement !== hbSlot) hbSlot.appendChild(row);
            row.classList.add("bv-ai-trade-row--rl-heartbeat");
        }
        const tsEl = document.createElement("span");
        tsEl.className = "bv-ai-trade-ts";
        tsEl.textContent = ts;
        const txtEl = document.createElement("span");
        txtEl.className = "bv-ai-trade-txt";
        txtEl.textContent = msg;
        const sideEl = document.createElement("span");
        sideEl.className = "bv-ai-trade-side";
        sideEl.textContent = "·";
        row.replaceChildren(tsEl, txtEl, sideEl);
    };
    window.__rlHeartbeatRefresh = tick;
    window.setTimeout(tick, 1200);
    window.setInterval(tick, 10000);
})();

const TERMINAL_PREDICTION_CHART_ID = "tradingPredictionChart";

(function injectPredictionWaitingStylesAppCore() {
    if (typeof document === "undefined" || document.getElementById("predictionChartWaitingSpinStyle")) return;
    const s = document.createElement("style");
    s.id = "predictionChartWaitingSpinStyle";
    s.textContent =
        "@keyframes predChartSpin{to{transform:rotate(360deg)}} .prediction-chart-ai-waiting{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;background:rgba(0,0,0,0.62);color:#67e8f9;font-family:'JetBrains Mono',ui-monospace,monospace;font-size:14px;font-weight:600;z-index:4;pointer-events:none;letter-spacing:0.02em}.prediction-chart-ai-waiting[hidden]{display:none!important}.prediction-chart-ai-waiting .prediction-chart-ai-waiting__spin{width:26px;height:26px;border:3px solid rgba(103,232,249,0.22);border-top-color:#67e8f9;border-radius:50%;animation:predChartSpin 0.75s linear infinite}";
    document.head.appendChild(s);
})();

if (typeof window.ensurePredictionChartWaitingOverlay !== "function") {
    window.ensurePredictionChartWaitingOverlay = function appCoreEnsurePredictionWaitingOverlay() {
        const canvas = document.getElementById(TERMINAL_PREDICTION_CHART_ID);
        if (!canvas || typeof document === "undefined") return null;
        let el = document.getElementById("predictionChartAiWaiting");
        if (!el) {
            const wrap = canvas.parentElement;
            if (!wrap) return null;
            const pos = window.getComputedStyle(wrap).position;
            if (pos === "static" || !pos) wrap.style.position = "relative";
            el = document.createElement("div");
            el.id = "predictionChartAiWaiting";
            el.className = "prediction-chart-ai-waiting";
            el.setAttribute("role", "status");
            el.setAttribute("aria-live", "polite");
            el.hidden = true;
            const spin = document.createElement("span");
            spin.className = "prediction-chart-ai-waiting__spin";
            spin.setAttribute("aria-hidden", "true");
            const txt = document.createElement("span");
            txt.textContent = "Waiting for AI…";
            el.appendChild(spin);
            el.appendChild(txt);
            wrap.appendChild(el);
        }
        return el;
    };
}
if (typeof window.setPredictionChartWaitingVisible !== "function") {
    window.setPredictionChartWaitingVisible = function appCoreSetPredictionWaitingVisible(show) {
        const fn = window.ensurePredictionChartWaitingOverlay;
        const el = typeof fn === "function" ? fn() : null;
        if (!el) return;
        el.hidden = !show;
    };
}

/** Zelfde logica als terminal.js::inferChartJsYDecimalsFromRange (cockpit laadt soms geen terminal.js). */
function __appCoreInferChartJsYDecimals(yMin, yMax) {
    const lo = Math.min(yMin, yMax);
    const hi = Math.max(yMin, yMax);
    const span = Math.abs(hi - lo);
    if (!(span > 0) && hi > 0) return Math.max(4, Math.min(8, Math.ceil(-Math.log10(hi)) + 2));
    if (!(span > 0)) return 4;
    return Math.max(4, Math.min(8, Math.ceil(-Math.log10(span / 6)) + 1));
}

if (typeof window.setChartTimeHint !== "function") {
    window.setChartTimeHint = function appCoreSetChartTimeHint(meta) {
        const el = document.getElementById("chartTimeHint");
        if (!el) return;
        const m = meta && typeof meta === "object" ? meta : {};
        const market = String(m.market || "BTC-EUR")
            .toUpperCase()
            .replace("/", "-");
        const updRaw = m.updated_at || m.last_engine_tick_utc || m.generated_at || m.ts || null;
        const predRaw = m.predicted_at || m.prediction_timestamp || null;
        const parse = (v) => {
            if (!v) return null;
            const d = new Date(v);
            return Number.isFinite(d.getTime()) ? d : null;
        };
        const fmtClock = (d) =>
            d
                ? d.toLocaleTimeString("nl-NL", {
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                      hour12: false,
                  })
                : "—";
        const fmtAgo = (d) => {
            if (!d) return "—";
            const sec = Math.max(0, Math.round((Date.now() - d.getTime()) / 1000));
            if (sec < 60) return `${sec}s geleden`;
            const min = Math.round(sec / 60);
            return `${min}m geleden`;
        };
        const updDt = parse(updRaw);
        const predDt = parse(predRaw);
        const updTxt = fmtClock(updDt);
        const updAgo = fmtAgo(updDt);
        const predTxt = fmtClock(predDt);
        if (predDt) {
            el.textContent = `${market} · update ${updTxt} (${updAgo}) · voorspelling ${predTxt}`;
        } else {
            el.textContent = `${market} · update ${updTxt} (${updAgo})`;
        }
    };
}

/** Subtiele gloed alleen op dataset 1 (AI / ghost line). */
const terminalPredictionGhostGlowPlugin = {
    id: "terminalPredictionGhostGlow",
    beforeDatasetDraw(chart, args) {
        if (args.index !== 1) return;
        const { ctx } = chart;
        ctx.save();
        ctx.shadowColor = "rgba(103, 232, 249, 0.72)";
        ctx.shadowBlur = 22;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
    },
    afterDatasetDraw(chart, args) {
        if (args.index !== 1) return;
        chart.ctx.restore();
    },
};

function getTradingPredictionChartJsInstance() {
    const U = window.ChartUtils;
    if (!U || typeof Chart === "undefined") return null;
    if (U.registry && U.registry[TERMINAL_PREDICTION_CHART_ID]) return U.registry[TERMINAL_PREDICTION_CHART_ID];
    if (U.charts && U.charts[TERMINAL_PREDICTION_CHART_ID]) return U.charts[TERMINAL_PREDICTION_CHART_ID];
    if (window.tradingChart && window.tradingChart.canvas && window.tradingChart.canvas.id === TERMINAL_PREDICTION_CHART_ID) {
        return window.tradingChart;
    }
    return null;
}

let __terminalChartTickRaf = null;
let __terminalChartTickPrice = null;

/** Alleen laatste candle (y) bijwerken; dataset 1 ongemoeid; chart.update('none'). */
function tickTerminalPredictionChartLivePrice(currentPrice) {
    const y = Number(currentPrice);
    if (!Number.isFinite(y) || y <= 0) return;
    const el = document.getElementById(TERMINAL_PREDICTION_CHART_ID);
    if (!el || el.offsetParent === null) return;
    __terminalChartTickPrice = y;
    if (__terminalChartTickRaf != null) return;
    __terminalChartTickRaf = requestAnimationFrame(() => {
        __terminalChartTickRaf = null;
        const px = __terminalChartTickPrice;
        __terminalChartTickPrice = null;
        const chart = getTradingPredictionChartJsInstance();
        if (!chart || !chart.data || !chart.data.datasets || !chart.data.datasets[0]) return;
        const ds0 = chart.data.datasets[0].data;
        if (!Array.isArray(ds0) || ds0.length === 0) return;
        const last = ds0[ds0.length - 1];
        if (!last || typeof last !== "object") return;
        const py = Number(px);
        if (!Number.isFinite(py) || py <= 0) return;
        last.y = py;
        const ds1 = chart.data.datasets[1] && chart.data.datasets[1].data;
        if (Array.isArray(ds1) && ds1.length && ds1[0] && typeof ds1[0] === "object") {
            ds1[0].y = py;
        }
        chart.update("none");
    });
}
window.tickTerminalPredictionChartLivePrice = tickTerminalPredictionChartLivePrice;

function updateTerminalChartJsPredictionDataset(payload) {
    if (typeof Chart === "undefined" || !payload || !window.ChartUtils || typeof window.ChartUtils.upsertChart !== "function") {
        return;
    }
    const tab = document.getElementById("tab-terminal");
    if (tab && tab.classList.contains("hidden")) return;

    const canvasId = TERMINAL_PREDICTION_CHART_ID;
    const el = document.getElementById(canvasId);
    if (!el) return;

    if (typeof window.ChartUtils.initDefaults === "function") window.ChartUtils.initDefaults();

    const hist = Array.isArray(payload.historical) ? payload.historical : [];
    const rawPred =
        Array.isArray(payload.predicted) && payload.predicted.length
            ? payload.predicted
            : Array.isArray(payload.prediction_data) && payload.prediction_data.length
              ? payload.prediction_data
              : [];
    const pr = rawPred;
    const symU = String(payload.symbol || "BTC-EUR").toUpperCase().replace("/", "-");
    const H = window.__terminalPredictionChartHelpers || {};

    let historicalPoints = hist.length
        ? hist
              .map((item) => {
                  const x = new Date(item.timestamp).getTime();
                  const y = Number(item.price);
                  return { x, y };
              })
              .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
        : [];
    if (!historicalPoints.length && typeof H.buildChartJsRealPriceFromMainLw === "function") {
        const lw = H.buildChartJsRealPriceFromMainLw(symU);
        if (lw && lw.length) historicalPoints = lw;
    }
    if (!historicalPoints.length) return;

    const lh = historicalPoints[historicalPoints.length - 1];
    const predPriceKey = (item) =>
        item.predicted_price != null ? item.predicted_price : item.price != null ? item.price : item.close;
    let predictedPoints = pr
        .map((item) => {
            const x = new Date(item.timestamp).getTime();
            const y = Number(predPriceKey(item));
            return { x, y };
        })
        .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (lh && predictedPoints.length) {
        predictedPoints.unshift({ x: lh.x, y: lh.y });
    }
    /* Geen applyRlSellConfidence hier: die kneep de Y-as alleen op Chart.js en niet op de LW-overlay,
       waardoor hoofd- vs tijdslijn-grafiek tegenstrijdig leken. Buy/Sell blijft op de cockpit-balkjes. */

    try {
        if (typeof window.setPredictionChartWaitingVisible === "function") {
            window.setPredictionChartWaitingVisible(predictedPoints.length === 0);
        }
    } catch (_e) {
        /* noop */
    }

    try {
        if (typeof window.setChartTimeHint === "function") {
            const lastPred = pr.length ? pr[pr.length - 1] : null;
            const predTs = lastPred && lastPred.timestamp ? String(lastPred.timestamp) : null;
            const histLast = hist.length ? hist[hist.length - 1] : null;
            const updTs = histLast && histLast.timestamp ? String(histLast.timestamp) : (window.__lastEngineTickIso || null);
            window.setChartTimeHint({
                market: hpCockpitSelectedMarketKey(),
                updated_at: updTs,
                predicted_at: predTs,
            });
        }
    } catch (_e) {
        /* noop */
    }

    const mergedPts = [...historicalPoints, ...predictedPoints];
    const allXs = mergedPts.map((p) => p.x).filter(Number.isFinite);
    const tMin = Math.min(...allXs);
    const tMax = Math.max(...allXs);
    const span = Math.max(60_000, tMax - tMin);
    const padLeft = Math.max(60_000, span * 0.1);
    const padRight = Math.max(120_000, span * 0.28);

    const ys = mergedPts.map((p) => p.y).filter(Number.isFinite);
    let yScale = {};
    const inferYDec =
        typeof H.inferChartJsYDecimalsFromRange === "function"
            ? H.inferChartJsYDecimalsFromRange
            : typeof window.inferChartJsYDecimalsFromRange === "function"
              ? window.inferChartJsYDecimalsFromRange
              : __appCoreInferChartJsYDecimals;
    const computeYB =
        typeof H.computePredictionChartYBounds === "function"
            ? H.computePredictionChartYBounds
            : typeof window.computePredictionChartYBounds === "function"
              ? window.computePredictionChartYBounds
              : null;
    if (ys.length && typeof inferYDec === "function") {
        const yb =
            computeYB != null
                ? computeYB(historicalPoints, predictedPoints)
                : (() => {
                      const yMin = Math.min(...ys);
                      const yMax = Math.max(...ys);
                      const spanY = Math.max(yMax - yMin, yMax * 1e-9);
                      const yPad = Math.max(spanY * 0.1, yMax * 0.001);
                      return { min: yMin - yPad, max: yMax + yPad };
                  })();
        if (yb && Number.isFinite(yb.min) && Number.isFinite(yb.max)) {
            const prec = inferYDec(yb.min, yb.max);
            yScale = {
                beginAtZero: false,
                min: yb.min,
                max: yb.max,
                ticks: {
                    maxTicksLimit: 8,
                    callback(v) {
                        const n = Number(v);
                        if (!Number.isFinite(n)) return "";
                        return n.toLocaleString("nl-NL", { minimumFractionDigits: prec, maximumFractionDigits: prec });
                    },
                },
            };
        }
    }

    window.ChartUtils.upsertChart(canvasId, {
        type: "line",
        plugins: [terminalPredictionGhostGlowPlugin],
        data: {
            datasets: [
                {
                    label: "Real Price",
                    data: historicalPoints,
                    parsing: false,
                    borderColor: "#00ff88",
                    backgroundColor: "rgba(0, 255, 136, 0.08)",
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.18,
                    fill: false,
                },
                {
                    label: "AI Prediction",
                    data: predictedPoints,
                    parsing: false,
                    borderColor: "#67e8f9",
                    backgroundColor: "transparent",
                    borderWidth: 3,
                    borderDash: [7, 5],
                    borderCapStyle: "round",
                    pointRadius: 0,
                    tension: 0.18,
                    fill: false,
                    spanGaps: false,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: "nearest", intersect: false },
            layout: {
                padding: { top: 28, bottom: 36, left: 4, right: 8 },
            },
            plugins: {
                legend: {
                    display: true,
                    position: "top",
                    align: "end",
                    labels: { boxWidth: 12, padding: 10 },
                },
                title: {
                    display: false,
                    text: "",
                    color: "#888888",
                    position: "bottom",
                    font: { size: 13, weight: "600" },
                },
            },
            scales: {
                x: {
                    type: "linear",
                    suggestedMin: tMin - padLeft * 0.08,
                    suggestedMax: tMax + padRight,
                    offset: false,
                    title: {
                        display: true,
                        text: "Tijd",
                        color: "#888888",
                        padding: { top: 10, bottom: 0 },
                    },
                    ticks: {
                        maxTicksLimit: 12,
                        maxRotation: 0,
                        padding: 6,
                        callback(v) {
                            const fn =
                                (typeof H.formatPredictionChartTickTime === "function" && H.formatPredictionChartTickTime) ||
                                (typeof window.formatPredictionChartTickTime === "function" && window.formatPredictionChartTickTime);
                            if (fn) return fn(v);
                            const n = Number(v);
                            if (!Number.isFinite(n)) return "";
                            const ms = n >= 1e11 ? n : n * 1000;
                            try {
                                return new Date(ms).toLocaleTimeString("nl-NL", {
                                    hour: "2-digit",
                                    minute: "2-digit",
                                    second: "2-digit",
                                    hour12: false,
                                });
                            } catch (_e) {
                                return String(v);
                            }
                        },
                    },
                },
                y: yScale,
            },
        },
    });

    requestAnimationFrame(() => {
        const ch = getTradingPredictionChartJsInstance();
        if (!ch) return;
        try {
            if (typeof ch.resize === "function") ch.resize();
        } catch (_e) {
            /* noop */
        }
        try {
            if (typeof ch.update === "function") ch.update("none");
        } catch (_e) {
            /* noop */
        }
    });

    if (typeof window.scheduleSyncPredictionChartJsXFromLightweight === "function") {
        window.scheduleSyncPredictionChartJsXFromLightweight();
    }

    const symRaw = symU;
    if (symRaw && Array.isArray(pr) && pr.length) {
        window.__hpLastMergedByMarket = window.__hpLastMergedByMarket || {};
        try {
            window.__hpLastMergedByMarket[symRaw] = JSON.parse(JSON.stringify(pr));
        } catch (_e) {
            window.__hpLastMergedByMarket[symRaw] = pr.slice();
        }
    }

    const accuracyElement = document.getElementById("ai-accuracy-score");
    if (accuracyElement && payload.accuracy_score != null && Number.isFinite(Number(payload.accuracy_score))) {
        const ac = Number(payload.accuracy_score);
        accuracyElement.textContent = `${ac.toFixed(1)}%`;
        accuracyElement.style.color = ac > 80 ? "#00ff88" : "#ff3e3e";
    }

    const predMeta = document.getElementById("prediction");
    if (predMeta && historicalPoints.length) {
        const tx = String(predMeta.textContent || "");
        if (/^laden/i.test(tx) || tx.includes("Laden…") || tx.includes("Laden...")) predMeta.textContent = "";
    }
    if (typeof window.setChartTimeHint === "function" && historicalPoints.length) {
        const histRows = Array.isArray(payload.historical) ? payload.historical : [];
        const predRows = pr.length ? pr : [];
        const hiRow = histRows.length ? histRows[histRows.length - 1] : null;
        const prRow = predRows.length ? predRows[predRows.length - 1] : null;
        window.setChartTimeHint({
            market: symU,
            updated_at: hiRow && hiRow.timestamp != null ? String(hiRow.timestamp) : new Date().toISOString(),
            predicted_at: prRow && prRow.timestamp != null ? String(prRow.timestamp) : null,
        });
    }
}

window.updateTerminalChartJsPredictionDataset = updateTerminalChartJsPredictionDataset;

if (typeof window.__repaintCockpitTerminalFromCache !== "function") {
    window.__repaintCockpitTerminalFromCache = function () {};
}

(function predictionsApiPoll() {
    if (window.__predictionsPollHooked) return;
    window.__predictionsPollHooked = true;
    const tick = () => {
        const sel = document.getElementById("marketSelect");
        const domMk = sel && sel.value ? String(sel.value).trim() : "";
        const sym =
            domMk ||
            (window.AppCore && window.AppCore.state && window.AppCore.state.selectedMarket) ||
            window.selectedMarket ||
            "BTC-EUR";
        const symU = String(sym).trim().toUpperCase().replace("/", "-");
        if (window.AppCore && window.AppCore.state) {
            window.AppCore.state.selectedMarket = symU;
        }
        const predUrl = `/api/v1/predictions?symbol=${encodeURIComponent(symU)}`;
        fetch(predUrl)
            .then((r) => (r.ok ? r.json() : null))
            .then((payload) => {
                if (!payload || payload.error) return;
                try {
                    if (window.localStorage && window.localStorage.getItem("debugPredictions") === "1") {
                        // Network-tab: vergelijk URL + ai_action_probs per symbool (localStorage.debugPredictions=1)
                        console.info("[predictions]", predUrl, payload.symbol, payload.ai_action_probs);
                    }
                } catch (_e) {
                    /* ignore */
                }
                window.__lastApiPredictionResponse = payload;
                window.__predictionCacheBySymbol = window.__predictionCacheBySymbol || {};
                window.__predictionCacheBySymbol[symU] = payload;
                if (typeof window.__repaintCockpitTerminalFromCache === "function") {
                    window.__repaintCockpitTerminalFromCache();
                }
                if (window.ModuleTerminal && typeof window.ModuleTerminal.refresh === "function") {
                    void window.ModuleTerminal.refresh();
                }
                updateTerminalChartJsPredictionDataset(payload);
                const pollSym = String(payload.symbol || symU)
                    .trim()
                    .toUpperCase()
                    .replace("/", "-");
                const probPayload = {
                    market: pollSym,
                    ticker: pollSym,
                    ai_action_probs: payload.ai_action_probs,
                    rl_last_decision: payload.rl_last_decision,
                    rl_confidence: payload.rl_confidence,
                    signal_threshold_pct: payload.signal_threshold_pct,
                    rl_decision_threshold_pct: payload.rl_decision_threshold_pct,
                };
                if (typeof window.paintAiPolicyProbsFromPayload === "function") {
                    window.paintAiPolicyProbsFromPayload(probPayload);
                }
                if (typeof window.applyHpAiProbThresholdMarkers === "function") {
                    window.applyHpAiProbThresholdMarkers(Object.assign({}, window.__lastDashboardStats || {}, probPayload));
                }
                if (typeof window.syncTerminalPredictionChartFromPolicyStats === "function") {
                    window.syncTerminalPredictionChartFromPolicyStats(window.__lastDashboardStats || {});
                }
                const el = document.getElementById("hpLiveHitRate");
                if (el && payload.accuracy_score != null && Number.isFinite(Number(payload.accuracy_score))) {
                    el.textContent = `Hit: ${Number(payload.accuracy_score).toFixed(1)}%`;
                }
            })
            .catch(() => {});
    };
    window.__reloadTerminalPredictionChart = tick;
    tick();
    setInterval(tick, 10000);
})();

/**
 * Chart.js + window.resize na wijziging terminal mid-layout.
 * Hoogtes en viewport-lock komen uit CSS (flex/grid); dit triggert alleen chart.resize na DOM-maten.
 */
function resizeChartsAfterTerminalMidLayout() {
    const chart = getTradingPredictionChartJsInstance();
    if (chart && typeof chart.resize === "function") {
        try {
            chart.resize();
        } catch (_e) {
            /* noop */
        }
    }
    try {
        window.dispatchEvent(new Event("resize"));
    } catch (_e) {
        /* noop */
    }
}
window.resizeChartsAfterTerminalMidLayout = resizeChartsAfterTerminalMidLayout;

function wireCockpitTerminalMidSplitResizeObserver() {
    if (typeof ResizeObserver === "undefined") return;
    if (window.__cockpitMidSplitResizeWired) return;
    const attach = () => {
        const root = document.getElementById("tab-terminal");
        if (!root || root.dataset.layoutRo === "1") return false;
        const targets = [
            root.querySelector("#dashboard.tcc-command-grid") ||
                root.querySelector(".dashboard-container.tcc-command-grid"),
            root.querySelector(".dashboard-col-main.dashboard-col-main--stack"),
            root.querySelector(".dashboard-col-right"),
            root.querySelector(".dashboard-mid-split"),
            root.querySelector(".dashboard-mid-row--primary.tcc-mid-row--chart"),
            root.querySelector(".dashboard-mid-row--prediction.tcc-mid-row--prediction"),
            root.querySelector("#priceChart"),
            root.querySelector(".hp-trading-chartjs-wrap"),
        ].filter(Boolean);
        if (!targets.length) return false;
        root.dataset.layoutRo = "1";
        let raf = null;
        const ro = new ResizeObserver(() => {
            if (raf != null) return;
            raf = requestAnimationFrame(() => {
                raf = null;
                resizeChartsAfterTerminalMidLayout();
            });
        });
        targets.forEach((t) => ro.observe(t));
        window.__cockpitMidSplitResizeRO = ro;
        window.__cockpitMidSplitResizeWired = "1";
        resizeChartsAfterTerminalMidLayout();
        return true;
    };
    if (!attach()) {
        const t0 = Date.now();
        const id = setInterval(() => {
            if (attach() || Date.now() - t0 > 12000) clearInterval(id);
        }, 120);
    }
}

// Strategy Settings inklapbaar
document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.getElementById("strategySettingsToggle");
    const body   = document.getElementById("strategySettingsBody");
    if (toggle && body) {
        toggle.addEventListener("click", () => {
            const expanded = toggle.getAttribute("aria-expanded") === "true";
            toggle.setAttribute("aria-expanded", String(!expanded));
            body.classList.toggle("is-collapsed", expanded);
            const chevron = toggle.querySelector(".bv-toggle-chevron");
            if (chevron) chevron.textContent = expanded ? "▸" : "▾";
        });
    }
});

// Debug-badge toggle: Ctrl+Shift+D toont/verbergt de payload-filter debug badge
document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.shiftKey && e.key === "D") {
        e.preventDefault();
        document.body.classList.toggle("bv-debug-mode");
    }
});

// ── Sentiment Analysis tab ──────────────────────────────────────────────────
window._loadSentimentTab = async function() {
    const tbody = document.getElementById("sentimentTableBody");
    const sidebar = document.getElementById("sentimentSidebar");
    const noData = document.getElementById("sentimentNoData");
    const lastUpdate = document.getElementById("sentimentLastUpdate");
    if (!tbody) return;

    tbody.innerHTML = '<tr><td colspan="4" style="padding:1.5rem;opacity:.35;text-align:center;">Laden…</td></tr>';
    if (sidebar) sidebar.innerHTML = "";

    const pill = (label) => {
        const lc = String(label || "neutral").toLowerCase();
        if (lc === "positive") return `<span style="background:#16a34a;color:#bbf7d0;border-radius:999px;padding:.15rem .6rem;font-size:.68rem;font-weight:700;letter-spacing:.03em;white-space:nowrap;display:inline-block;">POSITIVE</span>`;
        if (lc === "negative") return `<span style="background:#b91c1c;color:#fecaca;border-radius:999px;padding:.15rem .6rem;font-size:.68rem;font-weight:700;letter-spacing:.03em;white-space:nowrap;display:inline-block;">NEGATIVE</span>`;
        return `<span style="background:rgba(148,163,184,.15);color:#94a3b8;border-radius:999px;padding:.15rem .6rem;font-size:.68rem;font-weight:700;letter-spacing:.03em;white-space:nowrap;display:inline-block;">NEUTRAL</span>`;
    };

    try {
        const resp = await fetch("/api/v1/news/sentiment?limit=30");
        if (!resp.ok) throw new Error(resp.status);
        const data = await resp.json();
        const items = data.items || [];

        if (lastUpdate) {
            const now = new Date();
            lastUpdate.textContent = now.toLocaleTimeString("nl-NL", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
        }

        if (!items.length) {
            tbody.innerHTML = '<tr><td colspan="4" style="padding:1.5rem;opacity:.35;text-align:center;">Geen data — wacht op news-worker cycle (~5 min)</td></tr>';
            if (noData) noData.style.display = "none";
            return;
        }
        if (noData) noData.style.display = "none";

        const _fmtAms = (utcStr) => {
            if (!utcStr) return String(utcStr ?? "—");
            // Normaliseer naar iets dat Date() begrijpt:
            // "2026-05-12T11:00:49.477479+00:00" → vervang "+00:00"/"-00:00" door "Z"
            // "2026-05-12 11:00:49" (spatie, geen tz) → vervang spatie door T en voeg Z toe
            let normalized = String(utcStr).trim()
                .replace(/\+00:00$/, "Z")
                .replace(/-00:00$/, "Z")
                .replace(" ", "T");
            if (!/[Z+\-]\d*$/.test(normalized) && !normalized.endsWith("Z")) normalized += "Z";
            const d = new Date(normalized);
            if (isNaN(d.getTime())) return String(utcStr);   // fallback: toon rauwe waarde
            const hhmm = d.toLocaleTimeString("nl-NL", { timeZone: "Europe/Amsterdam", hour: "2-digit", minute: "2-digit" });
            const diffMs = Date.now() - d.getTime();
            const diffMin = Math.round(diffMs / 60000);
            const rel = diffMin < 1 ? "nu" : diffMin < 60 ? `${diffMin}m ago` : `${Math.floor(diffMin / 60)}u ago`;
            return `${hhmm} <span style="opacity:.4;font-size:.65rem;">(${rel})</span>`;
        };

        tbody.innerHTML = items.map((item, i) => {
            const ts = _fmtAms(item.ts_utc);
            const rowBg = i % 2 === 0 ? "" : "background:rgba(255,255,255,.02);";
            return `<tr style="border-bottom:1px solid rgba(255,255,255,.04);${rowBg}">
                <td style="padding:.3rem .6rem;opacity:.5;white-space:nowrap;font-size:.7rem;font-variant-numeric:tabular-nums;">${ts}</td>
                <td style="padding:.3rem .6rem;font-weight:700;white-space:nowrap;font-size:.75rem;">${item.market}</td>
                <td style="padding:.3rem .6rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:0;width:99%;font-size:.77rem;" title="${(item.headline || "").replace(/"/g, "&quot;")}">${item.headline || "—"}</td>
                <td style="padding:.3rem .6rem;text-align:center;">${pill(item.label)}</td>
            </tr>`;
        }).join("");

        if (!sidebar) return;
        const byMarket = {};
        for (const item of items) {
            if (!byMarket[item.market]) byMarket[item.market] = [];
            byMarket[item.market].push(Number(item.score));
        }
        const ttlStatus = data.ttl_status || {};
        sidebar.innerHTML = Object.keys(byMarket).sort().map(mkt => {
            const scores = byMarket[mkt];
            const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
            const ttl = ttlStatus[mkt] || {};
            const ttlSec = ttl.ttl_sec || 0;
            const valid = ttl.valid;
            const ttlBorder = valid ? (ttlSec > 300 ? "#22c55e" : "#eab308") : "#ef4444";
            const ttlText = valid ? `TTL ${ttlSec}s` : "verlopen";
            const pct = Math.max(2, Math.min(98, ((avg + 1) / 2) * 100));
            const gaugeCol = avg > 0.1 ? "#22c55e" : avg < -0.1 ? "#ef4444" : "#64748b";
            const scoreLabel = avg > 0.1 ? "Bullish" : avg < -0.1 ? "Bearish" : "Neutral";
            return `<div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:8px;padding:.85rem 1rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem;">
                    <span style="font-weight:700;font-size:.88rem;letter-spacing:.02em;">${mkt}</span>
                    <span style="border-left:3px solid ${ttlBorder};padding:.1rem .4rem;font-size:.65rem;opacity:.75;background:rgba(255,255,255,.05);border-radius:0 3px 3px 0;">${ttlText}</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.5rem;">
                    <span style="font-size:.7rem;opacity:.45;">${scoreLabel}</span>
                    <span style="font-size:.78rem;font-weight:600;color:${gaugeCol};font-variant-numeric:tabular-nums;">${avg >= 0 ? "+" : ""}${avg.toFixed(3)}</span>
                </div>
                <div style="background:rgba(255,255,255,.06);border-radius:4px;height:8px;overflow:hidden;">
                    <div style="width:${pct}%;height:100%;background:${gaugeCol};border-radius:4px;transition:width .5s ease;"></div>
                </div>
            </div>`;
        }).join("");
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="4" style="padding:1rem;color:#f87171;text-align:center;">Fout: ${err}</td></tr>`;
    }
};

document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("sentimentRefreshBtn");
    if (btn) btn.addEventListener("click", window._loadSentimentTab);
});

// Boot de applicatie
document.addEventListener("DOMContentLoaded", () => window.AppCore.init());