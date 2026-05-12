window.ModuleLedger = {
    _ledgerTab: "active",
    _lastTrades: [],
    /** Laatste `risk_profile` uit `/activity` (TP/SL-% voor ledger-kolommen). */
    _lastRiskProfile: {},

    _fmtEur(v) {
        if (typeof window.formatEurNl === "function") return window.formatEurNl(v);
        const n = Number(v);
        if (!Number.isFinite(n)) return "—";
        const neg = n < 0;
        const s = Math.abs(n).toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        return (neg ? "-€ " : "€ ") + s;
    },

    /** Werkelijke inleg (EUR): qty × entry, waar van toepassing. */
    _ledgerStakeEur(t) {
        const rt = String(t.row_type || "").toUpperCase();
        if (rt === "EVENT" && String(t.action || "").toUpperCase() !== "BUY") return null;
        const q = Number(t.qty || 0);
        const ep = Number(t.entry_price || 0);
        if (!Number.isFinite(q) || !Number.isFinite(ep) || q <= 0 || ep <= 0) return null;
        return q * ep;
    },

    /** Gerealiseerde winst (EUR) voor gesloten trades: (exit×qty) − (entry×qty) − fees, of engine-pnl. */
    _ledgerWinstEurClosed(t) {
        if (!this._isClosed(t)) return null;
        const rt = String(t.row_type || "").toUpperCase();
        const typ = String(t.type || "").toUpperCase();
        const q = Number(t.qty || 0);
        const ep = Number(t.entry_price || 0);
        const xp = Number(t.exit_price || 0);
        const fees = Number(t.fees_eur != null && t.fees_eur !== "" ? t.fees_eur : t.fee_eur) || 0;
        if (rt === "ROUND_TRIP" || typ === "ROUND_TRIP") {
            if (q > 0 && ep > 0 && xp > 0) return xp * q - ep * q - fees;
            const p = Number(t.pnl_eur);
            return Number.isFinite(p) ? p : null;
        }
        if (rt === "EVENT" && String(t.action || "").toUpperCase() === "SELL") {
            const p = Number(t.pnl_eur);
            return Number.isFinite(p) ? p : null;
        }
        if (this._isClosed(t) && xp > 0 && ep > 0 && q > 0) return xp * q - ep * q - fees;
        const p = Number(t.pnl_eur);
        return Number.isFinite(p) ? p : null;
    },

    onActivate() {
        if (typeof window.runSequencedStartup === "function" || typeof window.switchTab === "function") {
            console.log("[Legacy ModuleLedger] Bypassed because monolithic terminal.js is active.");
            return;
        }
        this.refresh();
        this.refreshAnalytics();
    },

    setLedgerView(tab) {
        const t = tab === "history" ? "history" : "active";
        this._ledgerTab = t;
        const table = document.getElementById("cockpitLedgerTable");
        if (table) table.classList.toggle("ledger-table--closed-mode", t === "history");
        this.renderTable({ trades: this._lastTrades });
    },

    async refresh() {
        let trades = null;
        let riskProfile = {};
        try {
            const act = await fetch(
                typeof window.buildActivityFetchUrl === "function" ? window.buildActivityFetchUrl() : "/activity",
                window.activityFetchInit || { cache: "no-store", credentials: "same-origin" }
            );
            if (act.ok) {
                const ad = await act.json();
                if (ad.risk_profile && typeof ad.risk_profile === "object") riskProfile = ad.risk_profile;
                if (Array.isArray(ad.trades)) trades = ad.trades;
            }
        } catch (_e) {}
        if (trades == null) {
            try {
                const res = await fetch(`/api/v1/trades?limit=500&view=roundtrip&market=all`);
                if (!res.ok) return;
                const data = await res.json();
                trades = Array.isArray(data.trades) ? data.trades : [];
            } catch (err) {
                console.error("[ModuleLedger] Fetch failed:", err);
                return;
            }
            if (!Object.keys(riskProfile).length) {
                try {
                    const act2 = await fetch(
                        typeof window.buildActivityFetchUrl === "function" ? window.buildActivityFetchUrl() : "/activity",
                        window.activityFetchInit || { cache: "no-store", credentials: "same-origin" }
                    );
                    if (act2.ok) {
                        const ad2 = await act2.json();
                        if (ad2.risk_profile && typeof ad2.risk_profile === "object") riskProfile = ad2.risk_profile;
                    }
                } catch (_e2) {}
            }
        }
        this._lastRiskProfile = riskProfile;
        this.renderTable({ trades });
    },

    /** Vast TP/SL-niveau t.o.v. entry (zelfde basis-% als risk engine; trailing SL kan eerder sluiten). */
    _ledgerTpSlCells(entryPx, closed) {
        if (closed || !Number.isFinite(entryPx) || entryPx <= 0) {
            return { tpHtml: "—", slHtml: "—" };
        }
        const rp = this._lastRiskProfile || {};
        const tpPct = Number(rp.take_profit_pct);
        const slPct = Number(rp.stop_loss_pct);
        const tpP = Number.isFinite(tpPct) && tpPct >= 0 ? tpPct : 5.0;
        const slP = Number.isFinite(slPct) && slPct >= 0 ? slPct : 2.5;
        const tpPx = entryPx * (1 + tpP / 100);
        const slPx = entryPx * (1 - slP / 100);
        return {
            tpHtml: tpPx.toFixed(4),
            slHtml: slPx.toFixed(4),
        };
    },

    async refreshAnalytics() {
        try {
            const res = await fetch("/api/v1/performance/analytics");
            if (!res.ok) return;
            const data = await res.json();

            if (window.ChartUtils) {
                const equity = data.equity_curve || [];
                const equityLabels = equity.map((r) => (r.ts || "").slice(11, 19));
                const equityValues = equity.map((r) => Number(r.equity || 0));
                window.ChartUtils.upsertChart("equityCurveChart", {
                    type: "line",
                    data: { labels: equityLabels, datasets: [{ label: "Equity", data: equityValues, borderColor: "#39ff14", borderWidth: 3, pointRadius: 0 }] },
                    options: { responsive: true, maintainAspectRatio: false },
                });

                const wl = data.analytics?.win_loss_ratio || {};
                window.ChartUtils.upsertChart("winLossChart", {
                    type: "doughnut",
                    data: { labels: ["Wins", "Losses"], datasets: [{ data: [Number(wl.wins || 0), Number(wl.losses || 0)], backgroundColor: ["#39ff14", "#ff3131"] }] },
                    options: { responsive: true, maintainAspectRatio: false },
                });

                const svo = data.analytics?.sentiment_vs_outcome || [];
                window.ChartUtils.upsertChart("sentimentOutcomeChart", {
                    type: "bar",
                    data: { labels: svo.map((r) => r.bucket), datasets: [{ label: "Avg PnL EUR", data: svo.map((r) => Number(r.avg_pnl_eur || 0)), backgroundColor: ["#39ff14", "#f8d04f", "#ff3131"] }] },
                    options: { responsive: true, maintainAspectRatio: false },
                });
            }

            const ps = data.analytics?.performance_summary || {};
            const wl = data.analytics?.win_loss_ratio || {};
            const closed = Number(ps.closed_trades ?? (Number(wl.wins || 0) + Number(wl.losses || 0)));
            const wr = Number(ps.win_rate_pct ?? wl.win_rate_pct ?? 0);

            const wrEl = document.getElementById("ledgerPerfWinRate");
            const mwEl = document.getElementById("ledgerPerfMaxWin");
            const mlEl = document.getElementById("ledgerPerfMaxLoss");
            const hdEl = document.getElementById("ledgerPerfHold");

            if (wrEl) wrEl.textContent = `${wr.toFixed(1)}% (${Number(wl.wins || 0)}W / ${closed}T)`;
            if (mwEl) mwEl.textContent = `+€${Number(ps.max_win_eur || 0).toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            if (mlEl) mlEl.textContent = `€${Math.abs(Number(ps.max_loss_eur || 0)).toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

            const holdHours = Number(ps.avg_hold_hours || 0);
            if (hdEl) {
                hdEl.textContent =
                    holdHours > 0 ? (holdHours < 48 ? `${holdHours.toFixed(1)} u` : `${Math.floor(holdHours / 24)}d ${(holdHours % 24).toFixed(1)}u`) : "0.0 u";
            }
        } catch (err) {
            console.error("[ModuleLedger] Analytics fetch failed:", err);
        }
    },

    _tradeDedupeKey(t) {
        const st = String(t.status || "").toUpperCase();
        const m = String(t.market || t.pair || "").toUpperCase();
        const o = String(t.open_time_utc || t.entry_ts_utc || "");
        const c = String(t.close_time_utc || t.exit_ts_utc || "");
        const e = Number(t.entry_price || 0);
        const x = t.exit_price != null && t.exit_price !== "" ? Number(t.exit_price) : -1;
        return `${st}|${m}|${o}|${c}|${e}|${x}`;
    },

    /** Sorteerkey: meest recente activiteit eerst (exit bij gesloten, anders entry). */
    _rowSortTs(t) {
        const s = String(
            t.exit_ts_utc || t.close_time_utc || t.entry_ts_utc || t.open_time_utc || t.sort_ts || t.ts || ""
        ).trim();
        if (!s) return 0;
        const norm = s.includes("T") ? s : s.replace(" ", "T");
        const ms = Date.parse(norm);
        return Number.isFinite(ms) ? ms : 0;
    },

    _isClosed(t) {
        const rt = String(t.row_type || "").toUpperCase();
        if (rt === "EVENT") {
            // Event-rows zijn logregels (entry/exit moment), geen blijvende open positie.
            // Open-posities komen uit ACTIVE_LOT-rows.
            return true;
        }
        if (rt === "ACTIVE_LOT") return false;
        const st = String(t.status || "").toUpperCase();
        if (st === "CLOSED") return true;
        if (t.exit_price !== null && t.exit_price !== undefined && String(t.exit_price) !== "") return true;
        return false;
    },

    _statusLabel(t) {
        const rt = String(t.row_type || "").toUpperCase();
        if (rt === "EVENT") {
            const act = String(t.action || "").toUpperCase();
            if (act === "BUY" || act === "SELL") return act;
        }
        if (this._isClosed(t)) return "Closed";
        const openMs = Date.parse(String(t.open_time_utc || t.entry_ts_utc || ""));
        if (Number.isFinite(openMs) && Date.now() - openMs < 90_000) return "Entry";
        return "Open";
    },

    _formatDur(ms) {
        if (!Number.isFinite(ms) || ms < 0) return "—";
        const s = Math.floor(ms / 1000);
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        if (h >= 72) return `${Math.floor(h / 24)}d`;
        if (h > 0) return `${h}u ${m}m`;
        if (m > 0) return `${m}m`;
        return "<1m";
    },

    _durationCell(t) {
        const openMs = Date.parse(String(t.open_time_utc || t.entry_ts_utc || ""));
        if (!Number.isFinite(openMs)) return "—";
        if (this._isClosed(t)) {
            const closeMs = Date.parse(String(t.close_time_utc || t.exit_ts_utc || ""));
            if (!Number.isFinite(closeMs)) return "—";
            return this._formatDur(closeMs - openMs);
        }
        return this._formatDur(Date.now() - openMs);
    },

    _pnlPctCell(t) {
        if (this._isClosed(t)) return Number(t.pnl_pct || 0);
        const live = Number(t.live_pnl_pct);
        if (Number.isFinite(live)) return live;
        return Number(t.pnl_pct || 0);
    },

    _statusTdHtml(t, esc, closed) {
        const label = this._statusLabel(t);
        let dot = "ledger-status-dot--open";
        if (closed || label === "Closed") dot = "ledger-status-dot--closed";
        else if (label === "Entry") dot = "ledger-status-dot--entry";
        return (
            `<td class="cockpit-ledger-mono">` +
            `<span class="ledger-status-cell">` +
            `<span class="ledger-status-dot ${dot}" aria-hidden="true"></span>` +
            `<span>${esc(label)}</span>` +
            `</span></td>`
        );
    },

    _appendTradeRow(tbody, t, esc) {
        const tr = document.createElement("tr");
        const closed = this._isClosed(t);
        if (closed) tr.classList.add("cockpit-ledger-row--closed");
        tr.classList.add("cockpit-ledger-row--flat");

        const ts = String(t.open_time_utc || t.entry_ts_utc || t.exit_ts_utc || t.close_time_utc || t.ts || "")
            .replace("T", " ")
            .slice(0, 19);
        const entryPx = Number(t.entry_price || 0);
        const { tpHtml, slHtml } = this._ledgerTpSlCells(entryPx, closed);
        const hasExit = closed;
        const exitPx = hasExit ? Number(t.exit_price || 0) : null;
        const dur = this._durationCell(t);
        const pct = this._pnlPctCell(t);
        const pctCls = pct >= 0 ? "positive" : "negative";
        const pairKey = String(t.market || t.pair || "-")
            .trim()
            .toUpperCase()
            .replace("/", "-");
        const pairFull = esc(pairKey);
        const dupSet = this._ledgerDuplicateOpenPairs;
        if (!closed && dupSet instanceof Set && dupSet.has(pairKey)) {
            tr.classList.add("cockpit-ledger-row--duplicate-warn");
            tr.title = "Dubbele OPEN-regels voor dit paar — controleer worker/paper merge.";
        }
        const ctx = esc(String(t.ledger_context || "—").slice(0, 120));
        const stake = this._ledgerStakeEur(t);
        const winEur = closed ? this._ledgerWinstEurClosed(t) : (Number.isFinite(Number(t.live_pnl_eur)) ? Number(t.live_pnl_eur) : null);
        const inlegCell = stake != null && Number.isFinite(stake) ? esc(this._fmtEur(stake)) : "—";
        const winstCls = !closed && winEur != null ? (winEur >= 0 ? " positive" : " negative") : "";
        const winstCell = winEur != null && Number.isFinite(winEur) ? esc(this._fmtEur(winEur)) : "—";
        const statusCol = this._statusTdHtml(t, esc, closed);

        tr.innerHTML =
            `<td class="cockpit-ledger-mono cockpit-ledger-pair" title="${pairFull}">${pairFull}</td>` +
            `<td class="cockpit-ledger-mono" title="${esc(ts)}">${esc(ts || "—")}</td>` +
            `<td class="cockpit-ledger-mono">${Number.isFinite(entryPx) ? entryPx.toFixed(4) : "—"}</td>` +
            `<td class="cockpit-ledger-mono">${tpHtml}</td>` +
            `<td class="cockpit-ledger-mono">${slHtml}</td>` +
            `<td class="cockpit-ledger-mono">${hasExit && exitPx != null && Number.isFinite(exitPx) ? exitPx.toFixed(4) : "—"}</td>` +
            `<td class="cockpit-ledger-mono">${inlegCell}</td>` +
            `<td class="cockpit-ledger-mono${winstCls}">${winstCell}</td>` +
            statusCol +
            `<td class="cockpit-ledger-mono">${esc(dur)}</td>` +
            `<td class="${pctCls} cockpit-ledger-mono">${pct.toFixed(2)}%</td>` +
            `<td class="cockpit-ledger-context">${ctx}</td>`;
        tbody.appendChild(tr);
    },

    _renderFlatClosed(tbody, trades, esc) {
        const closed = trades.filter((t) => this._isClosed(t));
        if (!closed.length) {
            tbody.innerHTML = `<tr><td colspan="12" class="cockpit-ledger-empty">Geen gesloten trades.</td></tr>`;
            return;
        }
        closed.sort((a, b) => this._rowSortTs(b) - this._rowSortTs(a));
        for (const t of closed.slice(0, 200)) {
            this._appendTradeRow(tbody, t, esc);
        }
    },

    /** Meerdere niet-gesloten rijen voor hetzelfde paar (UI-waarschuwing tot worker merge/close). */
    _dupOpenPairSet(trades) {
        const m = new Map();
        for (const t of trades) {
            if (this._isClosed(t)) continue;
            const p = String(t.market || t.pair || "")
                .trim()
                .toUpperCase()
                .replace("/", "-");
            if (!p) continue;
            m.set(p, (m.get(p) || 0) + 1);
        }
        const out = new Set();
        for (const [p, c] of m) {
            if (c > 1) out.add(p);
        }
        return out;
    },

    _renderFlatOpen(tbody, trades, esc) {
        const open = trades.filter((t) => !this._isClosed(t));
        if (!open.length) {
            tbody.innerHTML = `<tr><td colspan="12" class="cockpit-ledger-empty">Geen actieve trades.</td></tr>`;
            return;
        }
        open.sort((a, b) => this._rowSortTs(b) - this._rowSortTs(a));
        for (const t of open.slice(0, 200)) {
            this._appendTradeRow(tbody, t, esc);
        }
    },

    renderTable(data) {
        const tbody = document.getElementById("cockpitLedgerBody");
        if (!tbody) return;
        const table = document.getElementById("cockpitLedgerTable");
        if (table) table.classList.toggle("ledger-table--closed-mode", this._ledgerTab === "history");

        if (data && data.risk_profile && typeof data.risk_profile === "object" && Object.keys(data.risk_profile).length) {
            this._lastRiskProfile = data.risk_profile;
        }

        const raw = Array.isArray(data.trades) ? data.trades : [];
        if (!raw.length && this._lastTrades.length) return;

        // If new data has no open ACTIVE_LOT rows but cache does, the source is event-only → keep current display
        if (this._ledgerTab !== "history") {
            const newHasOpen = raw.some((t) => !this._isClosed(t));
            const cacheHasOpen = this._lastTrades.some((t) => !this._isClosed(t));
            if (!newHasOpen && cacheHasOpen) return;
        }

        const seen = new Map();
        for (const t of raw) {
            const k = this._tradeDedupeKey(t);
            if (!seen.has(k)) seen.set(k, t);
        }
        const trades = Array.from(seen.values());
        trades.sort((a, b) => this._rowSortTs(b) - this._rowSortTs(a));
        this._lastTrades = trades;
        this._ledgerDuplicateOpenPairs = this._dupOpenPairSet(trades);

        const esc = (s) =>
            String(s || "")
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;");

        // Build into a staging element; swap once to avoid mid-render flash
        const staging = document.createElement("tbody");
        if (!trades.length) {
            staging.innerHTML = `<tr><td colspan="12" class="cockpit-ledger-empty">Nog geen trades in de geschiedenis.</td></tr>`;
        } else if (this._ledgerTab === "history") {
            this._renderFlatClosed(staging, trades, esc);
        } else {
            this._renderFlatOpen(staging, trades, esc);
        }

        const newHtml = staging.innerHTML;
        if (newHtml !== tbody.innerHTML) tbody.innerHTML = newHtml;

        // Update no-trades classes on real DOM elements
        const hasOpen = this._ledgerTab !== "history" && trades.some((t) => !this._isClosed(t));
        table?.classList.toggle("has-no-trades", !hasOpen);
        const grid = document.querySelector("#tab-terminal .tcc-mid-split");
        grid?.classList.toggle("ledger-has-no-trades", !hasOpen);
    },
};
