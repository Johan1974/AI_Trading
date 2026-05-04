window.AppCore = {
    state: {
        activeTab: 'terminal',
        selectedMarket: 'BTC-EUR',
        balance: 0,
        cash: 0
    },
    sockets: {},
    
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
            return;
        }
        this.Logger.init();
        this.bindNavigation();
        this.connectSockets();
        this.switchTab('terminal'); // Default start
    },
    
    bindNavigation() {
        // Tab routing
        ['terminal', 'aibrain', 'ledger', 'hardware'].forEach(tab => {
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
                
                // Notificeer actieve views over de nieuwe markt
                if (window.ModuleTerminal) window.ModuleTerminal.refresh();
                if (window.ModuleBrain) window.ModuleBrain.refresh();
            });
        }

        // Ledger Footer Toggle
        const ledgerToggleBtn = document.getElementById("ledgerFooterToggle");
        if (ledgerToggleBtn) {
            ledgerToggleBtn.addEventListener("click", () => {
                const footer = document.getElementById("liveLedgerFooter");
                const icon = document.getElementById("ledgerFooterIcon");
                if (!footer) return;
                footer.classList.toggle("is-collapsed");
                if (icon) icon.textContent = footer.classList.contains("is-collapsed") ? "▲ Uitklappen" : "▼ Inklappen";
            });
        }
    },
    
    switchTab(tabName) {
        this.state.activeTab = tabName;
        
        // Reset UI
        ['terminal', 'aibrain', 'ledger', 'hardware'].forEach(name => {
            const el = document.getElementById(`tab-${name}`);
            const btn = document.getElementById(`btn-${name}`);
            if (el) { el.classList.add("hidden"); el.style.display = "none"; }
            if (btn) btn.classList.remove("active");
        });

        // Activeer nieuwe tab
        const activeEl = document.getElementById(`tab-${tabName}`);
        const activeBtn = document.getElementById(`btn-${tabName}`);
        if (activeEl) { activeEl.classList.remove("hidden"); activeEl.style.display = "flex"; }
        if (activeBtn) activeBtn.classList.add("active");
        
        // Tab-Isolatie: Cleanup grafieken en stop sockets
        if (window.ChartUtils) window.ChartUtils.clearAllCharts();
        if (tabName !== 'aibrain' && window.ModuleBrain && typeof window.ModuleBrain.stopSocket === 'function') window.ModuleBrain.stopSocket();

        // Dispatch activation signal naar de geïsoleerde module
        if (tabName === 'terminal' && window.ModuleTerminal) window.ModuleTerminal.onActivate();
        if (tabName === 'ledger' && window.ModuleLedger) window.ModuleLedger.onActivate();
        if (tabName === 'aibrain' && window.ModuleBrain) window.ModuleBrain.onActivate();
    },
    
    connectSockets() {
        // Hier beheer je centraal de WebSocket connecties
        const protocol = window.location.protocol === "https:" ? "wss" : "ws";
        this.sockets.trading = new WebSocket(`${protocol}://${window.location.host}/ws/trading-updates`);
        
        this.sockets.trading.onmessage = (e) => {
            // Verwerk balance / live data hier en sla op in this.state
        };
    }
};

// Boot de applicatie
document.addEventListener("DOMContentLoaded", () => window.AppCore.init());