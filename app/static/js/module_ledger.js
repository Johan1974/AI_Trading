window.ModuleLedger = {
    onActivate() {
        if (typeof window.runSequencedStartup === 'function' || typeof window.switchTab === 'function') {
            console.log("[Legacy ModuleLedger] Bypassed because monolithic terminal.js is active.");
            return;
        }
        this.refresh();
        this.refreshAnalytics();
    },

    async refresh() {
        try {
            const res = await fetch(`/api/v1/trades?limit=500&view=roundtrip&market=all`);
            if (!res.ok) return;
            const data = await res.json();
            this.renderTable(data);
        } catch (err) {
            console.error("[ModuleLedger] Fetch failed:", err);
        }
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
                    type: "line", data: { labels: equityLabels, datasets: [{ label: "Equity", data: equityValues, borderColor: "#39ff14", borderWidth: 3, pointRadius: 0 }] }, options: { responsive: true, maintainAspectRatio: false }
                });

                const wl = data.analytics?.win_loss_ratio || {};
                window.ChartUtils.upsertChart("winLossChart", {
                    type: "doughnut", data: { labels: ["Wins", "Losses"], datasets: [{ data: [Number(wl.wins || 0), Number(wl.losses || 0)], backgroundColor: ["#39ff14", "#ff3131"] }] }, options: { responsive: true, maintainAspectRatio: false }
                });

                const svo = data.analytics?.sentiment_vs_outcome || [];
                window.ChartUtils.upsertChart("sentimentOutcomeChart", {
                    type: "bar", data: { labels: svo.map(r => r.bucket), datasets: [{ label: "Avg PnL EUR", data: svo.map(r => Number(r.avg_pnl_eur || 0)), backgroundColor: ["#39ff14", "#f8d04f", "#ff3131"] }] }, options: { responsive: true, maintainAspectRatio: false }
                });
            }

            const ps = data.analytics?.performance_summary || {};
            const wl = data.analytics?.win_loss_ratio || {};
            const closed = Number(ps.closed_trades ?? (Number(wl.wins||0) + Number(wl.losses||0)));
            const wr = Number(ps.win_rate_pct ?? wl.win_rate_pct ?? 0);
            
            const wrEl = document.getElementById("ledgerPerfWinRate");
            const mwEl = document.getElementById("ledgerPerfMaxWin");
            const mlEl = document.getElementById("ledgerPerfMaxLoss");
            const hdEl = document.getElementById("ledgerPerfHold");
            
            if (wrEl) wrEl.textContent = `${wr.toFixed(1)}% (${Number(wl.wins||0)}W / ${closed}T)`;
            if (mwEl) mwEl.textContent = `+€${Number(ps.max_win_eur || 0).toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            if (mlEl) mlEl.textContent = `€${Math.abs(Number(ps.max_loss_eur || 0)).toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            
            const holdHours = Number(ps.avg_hold_hours || 0);
            if (hdEl) {
                hdEl.textContent = holdHours > 0 ? (holdHours < 48 ? `${holdHours.toFixed(1)} u` : `${Math.floor(holdHours/24)}d ${(holdHours%24).toFixed(1)}u`) : "0.0 u";
            }
        } catch (err) {
            console.error("[ModuleLedger] Analytics fetch failed:", err);
        }
    },

    renderTable(data) {
        const tbody = document.getElementById("cockpitLedgerBody");
        if (!tbody) return;

        // BUGFIX: Leegmaken van container om dubbele rijen te voorkomen
        tbody.innerHTML = "";

        const trades = Array.isArray(data.trades) ? data.trades : [];
        if (trades.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="cockpit-ledger-empty">Nog geen trades in de geschiedenis.</td></tr>`;
            return;
        }

        // BUGFIX: Loop met forEach om DOM opbouw strak te houden
        trades.forEach(t => {
            const tr = document.createElement("tr");
            const ts = String(t.open_time_utc || t.entry_ts_utc || t.ts || "").replace("T", " ").slice(0, 19);
            const entryPx = Number(t.entry_price || 0);
            const hasExit = t.exit_price !== null && t.exit_price !== undefined;
            const pnl = Number(t.pnl_eur || 0);
            
            tr.innerHTML = `
                <td>${ts || "-"}</td>
                <td class="cockpit-ledger-asset"><span class="ledger-asset-chip">${String(t.coin || t.market || "?").split("-")[0].toUpperCase()}</span></td>
                <td>${entryPx.toFixed(4)}</td>
                <td>${hasExit ? Number(t.exit_price || 0).toFixed(4) : "ACTIVE"}</td>
                <td class="${pnl >= 0 ? "positive" : "negative"}">${hasExit ? pnl.toFixed(2) : "ACTIVE"}</td>
                <td class="${Number(t.pnl_pct || 0) >= 0 ? "positive" : "negative"}">${hasExit ? Number(t.pnl_pct || 0).toFixed(2) + "%" : "ACTIVE"}</td>
                <td class="cockpit-ledger-context">${String(t.ledger_context || "—").slice(0, 150)}</td>
            `;
            tbody.appendChild(tr);
        });
    }
};