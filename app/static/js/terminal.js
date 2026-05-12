/*
  Bestand: app/static/js/terminal.js
  Relatief pad: ./app/static/js/terminal.js
  Functie: Clientlogica voor immersive trading terminal met chart markers en live inzichten.
*/

window.ChartUtils = window.ChartUtils || {
    charts: {},
    upsertChart(canvasId, config) {
        const el = document.getElementById(canvasId);
        if (!el) return null;
        if (el.offsetParent === null) return null;
        
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }
        if (typeof Chart !== "undefined") {
            const ctx = el.getContext("2d");
            this.charts[canvasId] = new Chart(ctx, config);
            return this.charts[canvasId];
        }
        return null;
    },
    clearAllCharts() {
        for (const id in this.charts) {
            if (this.charts[id]) this.charts[id].destroy();
        }
        this.charts = {};
    }
};

let selectedMarket = "BTC-EUR";
let currentMarket = "BTC-EUR"; // Globale pointer voor externe compatibiliteit
let selectedChartPair = "BTC-EUR";
/** Laatste actieve markten uit /markets/active (voor Elite-switch zonder extra fetch). */
let lastActiveMarketsRows = [];
/** Standaard Bitvavo candle-interval voor hoofdchart (hartslag / detail). */
const CHART_CANDLE_INTERVAL = "5m";
/** Ledger roundtrip API: grote payloads veroorzaken ERR_EMPTY_RESPONSE; 500 is ruim voor UI. */
const LEDGER_ROUNDTRIP_FETCH_LIMIT = 500;
/** Sidebar trade-widget: alleen recente events nodig. */
const SIDEBAR_TRADES_FETCH_LIMIT = 200;

/** Euro's in NL-notatie (komma decimalen); valt terug als `app_core.js` nog niet geladen is. */
function formatEurNlTerminal(v) {
  if (typeof window.formatEurNl === "function") return window.formatEurNl(v);
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  const neg = n < 0;
  const s = Math.abs(n).toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return (neg ? "-€ " : "€ ") + s;
}

/**
 * Trade-ledger: meest recente rij eerst (exit- of entry-timestamp).
 * Platte chronologische tabel (geen groep per munt) staat in ``module_ledger.js`` (`renderTable` / `_renderFlatOpen` / `_renderFlatClosed`).
 */
function sortLedgerTradesChronoDesc(rows) {
  const key = (t) => {
    const s = String(
      t.exit_ts_utc || t.close_time_utc || t.entry_ts_utc || t.open_time_utc || t.sort_ts || t.ts || ""
    ).trim();
    if (!s) return 0;
    const ms = Date.parse(s.includes("T") ? s : s.replace(" ", "T"));
    return Number.isFinite(ms) ? ms : 0;
  };
  rows.sort((a, b) => key(b) - key(a));
}

/** Kleine delay-helper voor sequenced loader: voorkomt network-storm bij startup. */
function sleepMs(ms) {
  return new Promise((resolve) => setTimeout(resolve, Math.max(0, Number(ms) || 0)));
}

/**
 * Paper-saldo (`paper_portfolio` in `/activity`); altijd `no-store` + query-bust (geen oude HTTP-cache).
 * Zie `terminal_live_tail.js` voor `window.buildActivityFetchUrl` / `window.activityFetchInit`.
 */
async function fetchBalanceFromActivity() {
  const url =
    typeof window.buildActivityFetchUrl === "function" ? window.buildActivityFetchUrl() : "/activity";
  const init = window.activityFetchInit || { cache: "no-store", credentials: "same-origin" };
  const res = await fetch(url, init);
  const text = await res.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch (_parseErr) {
    return { ok: res.ok, status: res.status, data: {}, parseError: true };
  }
  return { ok: res.ok, status: res.status, data, parseError: false };
}
let lastBufferedPrice = null;
let priceAtLastSecondTick = null;
let cockpitHeartbeatTimer = null;
let wsRef = null;
let priceChart = null;
let priceSeries = null;
let predictionOverlaySeries = null;
let markerSeries = null;
let whaleDangerPriceLineHandle = null;
let priceSeriesIsCandle = false;
let priceChartResizeObserver = null;
let equityCurveChart = null;
let winLossChart = null;
let sentimentOutcomeChart = null;
let latestNewsItems = [];
let brainRewardChart = null;
let brainEpisodeChart = null;
let brainEntropyChart = null;
let brainLossChart = null;
let brainNewsLagChart = null;
let brainBenchmarkChart = null;
let brainCorrelationChart = null;
let activeTab = "terminal";
let systemLogsSocket = null;
/** Redis-bridge: worker → portal WebSocket `/ws/trading-updates`. */
let tradingUpdatesSocket = null;
let tradingUpdatesLive = false;
const TRADING_UPDATES_WS_RECONNECT_MS = 5000;
let tradingUpdatesReconnectTimer = null;
/** Laatste Redis/activity-payload (ms); WS kan open staan zonder berichten — dan blijft `/activity` nodig. */
let lastTradingWsActivityAtMs = 0;
let systemLogsReconnectTimer = null;
let systemStatsSocket = null;
let systemStatsReconnectTimer = null;
let brainTabTrainingLossChart = null;
let brainTabRewardErrorChart = null;
let brainTabFeatureChart = null;
let brainStatsSocket = null;
let brainStatsReconnectTimer = null;
const MAX_SYSTEM_LOG_LINES = 500;
let systemLogsPaused = false;
let systemLogsMuted = false;
let systemLogsBuffer = [];
let lastLedgerCycleSeq = null;
let rawChartLineData = [];
let rawChartMarkers = [];
let latestBrainStatsPayload = null;
let markerRenderState = { thresholdPx: 34, hideText: false, sizeMode: "normal" };
/** Chart.js: historische prijs + AI-voorspelling (`/api/v1/predictions?symbol=...`). */
let tradingChart = null;

/** Donkere gridlijnen zodat witte assen en neon datasets contrast houden */
const CHART_GRID_DARK = "#222222";
const CHART_AXIS_WHITE = "#FFFFFF";

const HINT_PORTAL_ID = "brainHintPortal";
const CHART_INTERVAL_STORAGE_KEY = "trading_timeframe";
const CHART_INTERVAL_ALLOWED = new Set(["ai", "1m", "5m", "15m", "60m"]);

function normalizeChartIntervalSelection(raw) {
  const v = String(raw || "").toLowerCase();
  return CHART_INTERVAL_ALLOWED.has(v) ? v : "ai";
}

function chartIntervalFromSelection(sel) {
  const s = normalizeChartIntervalSelection(sel);
  return s === "ai" ? CHART_CANDLE_INTERVAL : s;
}

function updateChartIntervalUi(activeSelection) {
  const chosen = normalizeChartIntervalSelection(activeSelection);
  try {
    const btns = document.querySelectorAll("#chartTimeframeControls .chart-timeframe-btn[data-timeframe]");
    btns.forEach((btn) => {
      const tf = normalizeChartIntervalSelection(btn.getAttribute("data-timeframe"));
      btn.classList.toggle("is-active", tf === chosen);
      btn.setAttribute("aria-pressed", tf === chosen ? "true" : "false");
    });
  } catch (_e) {
    /* ignore */
  }
}

let chartIntervalSelection = (() => {
  try {
    return normalizeChartIntervalSelection(window.localStorage.getItem(CHART_INTERVAL_STORAGE_KEY));
  } catch (_e) {
    return "ai";
  }
})();
let chartInterval = chartIntervalFromSelection(chartIntervalSelection);
let headlessMode = false;
let __lastTerminalHwHud = "—";

function setTerminalDotState(dotId, state) {
  const el = document.getElementById(dotId);
  if (!el) return;
  el.classList.remove("is-ok", "is-error", "is-loading");
  if (state === "ok") el.classList.add("is-ok");
  else if (state === "error") el.classList.add("is-error");
  else el.classList.add("is-loading");
}

function updateTerminalHealthFromStats(stats) {
  if (!stats || typeof stats !== "object") {
    setTerminalDotState("termDotAiLive", "loading");
    setTerminalDotState("termDotGpuStatus", "loading");
    setTerminalDotState("termDotDbStatus", "loading");
    return;
  }
  const botStatus = String(stats.bot_status || "").toLowerCase();
  const aiFresh = stats.prediction_fresh === true;
  const aiOk = aiFresh && botStatus !== "panic_stop";
  const dev = String(stats.compute_device || "").toLowerCase();
  const gpuTemp = Number(stats.gpu_temp);
  const gpuOk = dev.includes("cuda") && Number.isFinite(gpuTemp) && gpuTemp > 0;
  const dbOk = !!(stats.paper_portfolio && typeof stats.paper_portfolio === "object");

  setTerminalDotState("termDotAiLive", aiOk ? "ok" : aiFresh ? "loading" : "error");
  setTerminalDotState("termDotGpuStatus", gpuOk ? "ok" : dev ? "error" : "loading");
  setTerminalDotState("termDotDbStatus", dbOk ? "ok" : "loading");

  const txtAi = document.getElementById("termTxtAiLive");
  const txtGpu = document.getElementById("termTxtGpuStatus");
  const txtDb = document.getElementById("termTxtDbStatus");
  if (txtAi) txtAi.textContent = aiOk ? "AI Live OK" : aiFresh ? "AI Warming" : "AI Stale";
  if (txtGpu) txtGpu.textContent = gpuOk ? "GPU OK" : dev === "cpu" ? "GPU Error" : "GPU Loading";
  if (txtDb) txtDb.textContent = dbOk ? "DB Connected" : "DB Syncing";

  const root = document.getElementById("cockpitTerminalHwMini");
  const hud = document.getElementById("hpMiniHud");
  const warn = document.getElementById("hpMiniWarnIcon");
  const cpu = Number(stats.cpu_load);
  const temp = Number(stats.gpu_temp);
  if (hud) {
    if (Number.isFinite(cpu) && cpu >= 0) {
      const tText = Number.isFinite(temp) && temp > 0 ? `${temp.toFixed(0)}°C` : "—";
      __lastTerminalHwHud = `CPU ${cpu.toFixed(0)}% | TMP ${tText}`;
      hud.textContent = __lastTerminalHwHud;
    } else if (!hud.textContent || hud.textContent.trim() === "—") {
      hud.textContent = __lastTerminalHwHud;
    }
  }
  const gpuFail = !gpuOk && dev.length > 0;
  if (root) root.classList.toggle("hp-hw-mini-monitor--alert", gpuFail);
  if (warn) {
    warn.textContent = gpuFail ? "⚠" : "●";
    warn.style.color = gpuFail ? "#ef4444" : "#6b7280";
  }
}

window.updateTerminalHealthFromStats = updateTerminalHealthFromStats;

/** Per-base accent (hex) voor actieve Elite-8 / scanner highlight */
const ELITE8_COIN_ACCENTS = Object.freeze({
  BTC: "#f7931a",
  ETH: "#627eea",
  SOL: "#9945ff",
  XRP: "#00aae4",
  DOGE: "#c2a633",
  ADA: "#0033ad",
  DOT: "#e6007a",
  AVAX: "#e84142",
  LTC: "#bfbbbb",
  LINK: "#2a5ada",
  MATIC: "#8247e5",
  BNB: "#f0b90b",
});

function baseAccentForMarket(market) {
  const raw = String(market || "").toUpperCase();
  const base = raw.includes("-") ? raw.split("-")[0] : raw;
  return ELITE8_COIN_ACCENTS[base] || "#00ff88";
}

// Helper: Veilig meerdere text/HTML elementen updaten op basis van classes
function setClassText(selector, text) {
  document.querySelectorAll(selector).forEach(el => el.textContent = text);
}
function setClassHTML(selector, html) {
  document.querySelectorAll(selector).forEach(el => el.innerHTML = html);
}

/**
 * window.botMetrics — enige bron voor DOM-id-registry + scalar/cache state (geen verborgen metric-DOM).
 * botState is een alias voor backwards compatibility / console.
 */
function initBotMetrics() {
  const base = {
    correlation: { sentimentPrice: 0, newsWeight: 0, priceWeight: 0 },
    marketState: {
      fearGreed: "",
      btcDom: "",
      whale: "",
      macro: "",
      rsi: "",
      macd: "",
    },
    focusLine: "",
    trainingStats: { learningRate: "", steps: "", exploration: "" },
    portfolio: {},
    sentiment: {},
    storage: {},
    reasoningText: "",
    wsPrices: {},
  };
  const ex = window.botMetrics && typeof window.botMetrics === "object" ? window.botMetrics : {};
  window.botMetrics = {
    ...base,
    ...ex,
    correlation: { ...base.correlation, ...(ex.correlation || {}) },
    marketState: { ...base.marketState, ...(ex.marketState || {}) },
    trainingStats: { ...base.trainingStats, ...(ex.trainingStats || {}) },
    portfolio: { ...base.portfolio, ...(ex.portfolio || {}) },
    sentiment: { ...base.sentiment, ...(ex.sentiment || {}) },
    storage: { ...base.storage, ...(ex.storage || {}) },
    wsPrices: { ...base.wsPrices, ...(ex.wsPrices || {}) },
  };
}

initBotMetrics();
window.botState = window.botMetrics;

/** Twee segmenten voor de ledger-strip; beide worden na fetch bijgewerkt. */
const cockpitLedgerStatusParts = { markets: "Markten: …", ledger: "Ledger: …" };

function paintCockpitLedgerStatus() {
  const el = document.getElementById("cockpitLedgerStatusText");
  if (!el) return;
  el.textContent = `${cockpitLedgerStatusParts.markets} · ${cockpitLedgerStatusParts.ledger}`;
}

function mergeContrastScale(scale = {}) {
  const prevTicks = scale.ticks || {};
  const prevFont =
    typeof prevTicks.font === "object" && prevTicks.font !== null ? prevTicks.font : {};
  const prevGrid = scale.grid || {};
  const prevBorder = scale.border || {};
  return {
    ...scale,
    ticks: {
      ...prevTicks,
      color: CHART_AXIS_WHITE,
      font: {
        ...prevFont,
        size: prevFont.size !== undefined ? prevFont.size : 14,
        family: prevFont.family || "'JetBrains Mono', ui-monospace, monospace",
        weight: prevFont.weight !== undefined ? prevFont.weight : "700",
      },
    },
    grid: {
      ...prevGrid,
      color: prevGrid.color !== undefined ? prevGrid.color : CHART_GRID_DARK,
    },
    border: {
      ...prevBorder,
      color: prevBorder.color !== undefined ? prevBorder.color : CHART_AXIS_WHITE,
    },
  };
}

function highContrastChartOptions(base = {}) {
  const options = { ...base };
  const existingPlugins = options.plugins || {};
  const existingLegend = existingPlugins.legend || {};
  const existingTooltip = existingPlugins.tooltip || {};
  options.plugins = {
    ...existingPlugins,
    legend: {
      ...existingLegend,
      labels: {
        ...(existingLegend.labels || {}),
        color: CHART_AXIS_WHITE,
        font: {
        size: 14,
        weight: "700",
        family: "'JetBrains Mono', ui-monospace, monospace",
          ...((existingLegend.labels && existingLegend.labels.font) || {}),
        },
      },
    },
    tooltip: {
      ...existingTooltip,
      backgroundColor: "#000000",
      titleColor: CHART_AXIS_WHITE,
      bodyColor: CHART_AXIS_WHITE,
      footerColor: CHART_AXIS_WHITE,
      titleFont: { weight: "bold", size: 14, ...(existingTooltip.titleFont || {}) },
      bodyFont: { weight: "bold", size: 13, ...(existingTooltip.bodyFont || {}) },
      footerFont: { weight: "bold", ...(existingTooltip.footerFont || {}) },
      borderColor: CHART_AXIS_WHITE,
      borderWidth: existingTooltip.borderWidth !== undefined ? existingTooltip.borderWidth : 1,
    },
  };

  options.scales = options.scales ? { ...options.scales } : {};
  const scaleKeys = Object.keys(options.scales);
  if (!scaleKeys.length) {
    options.scales.x = mergeContrastScale({
      ticks: { maxRotation: 0 },
    });
    options.scales.y = mergeContrastScale({});
  } else {
    for (const key of scaleKeys) {
      const prev = options.scales[key] || {};
      const extraTicks =
        key === "x" && !(prev.ticks && prev.ticks.maxRotation !== undefined)
          ? { maxRotation: 0 }
          : {};
      options.scales[key] = mergeContrastScale({
        ...prev,
        ticks: { ...(prev.ticks || {}), ...extraTicks },
      });
    }
  }

  return options;
}

(function applyChartJsDefaults() {
  if (typeof Chart === "undefined" || !Chart.defaults) return;
  Chart.defaults.color = CHART_AXIS_WHITE;
  Chart.defaults.font = Chart.defaults.font || {};
  Chart.defaults.font.family = "'JetBrains Mono', ui-monospace, monospace";
  Chart.defaults.font.size = 14;
  Chart.defaults.font.weight = "700";
})();

/** Chart.js x = ms; LWC `rawChartLineData.time` = zelfde schaal als toEpochSeconds (sec). */
function _normMkChart(s) {
  return String(s || "").toUpperCase().replace("/", "-");
}

function buildChartJsRealPriceFromMainLw(symbol) {
  const want = _normMkChart(symbol);
  const cur = _normMkChart(selectedChartPair || selectedMarket || "");
  if (!want || want !== cur) return null;
  const arr = Array.isArray(rawChartLineData) ? rawChartLineData : [];
  if (!arr.length) return null;
  const take = Math.min(arr.length, 96);
  const slice = arr.slice(-take);
  const out = [];
  for (let i = 0; i < slice.length; i += 1) {
    const p = slice[i];
    const t = Number(p.time);
    const y = Number(p.value);
    if (!Number.isFinite(t) || !Number.isFinite(y)) continue;
    out.push({ x: t * 1000, y });
  }
  return out.length ? out : null;
}

/** Chart.js lineaire x-as in ms: tick → HH:mm:ss (zelfde tijdlijn als worker candles). */
function formatPredictionChartTickTime(axisValue) {
  const n = Number(axisValue);
  if (!Number.isFinite(n)) return "";
  const ms = n >= 1e11 ? n : n * 1000;
  if (!Number.isFinite(ms) || ms <= 0) return "";
  try {
    return new Date(ms).toLocaleTimeString("nl-NL", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch (_e) {
    return "";
  }
}

function inferChartJsYDecimalsFromRange(yMin, yMax) {
  const lo = Math.min(yMin, yMax);
  const hi = Math.max(yMin, yMax);
  const span = Math.abs(hi - lo);
  if (!(span > 0) && hi > 0) return Math.max(4, Math.min(8, Math.ceil(-Math.log10(hi)) + 2));
  if (!(span > 0)) return 4;
  return Math.max(4, Math.min(8, Math.ceil(-Math.log10(span / 6)) + 1));
}
window.inferChartJsYDecimalsFromRange = inferChartJsYDecimalsFromRange;

/** Y-as alleen op recent venster + voorspelling → kleine prijsbewegingen blijven zichtbaar (geen platte lijn). */
function computePredictionChartYBounds(historicalPoints, predictedPoints) {
  const hp = Array.isArray(historicalPoints) ? historicalPoints : [];
  const pp = Array.isArray(predictedPoints) ? predictedPoints : [];
  const tailN = Math.max(12, Math.min(64, Number(window.PREDICTION_CHART_Y_TAIL || 32)));
  const histTail = hp.slice(-Math.min(tailN, Math.max(1, hp.length)));
  const ys = [...histTail, ...pp].map((p) => Number(p && p.y)).filter((y) => Number.isFinite(y) && y > 0);
  if (!ys.length) return null;
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  let spanY = yMax - yMin;
  const mid = (yMin + yMax) / 2;
  const minRel =
    Number.isFinite(Number(window.PREDICTION_CHART_MIN_REL_Y_SPAN)) && Number(window.PREDICTION_CHART_MIN_REL_Y_SPAN) > 0
      ? Number(window.PREDICTION_CHART_MIN_REL_Y_SPAN)
      : 0.0015;
  const minAbsSpan = Math.max(Math.abs(mid) * minRel, 1e-12);
  if (spanY < minAbsSpan) {
    const half = minAbsSpan / 2;
    yMin = mid - half;
    yMax = mid + half;
    spanY = minAbsSpan;
  }
  const yPad = Math.max(spanY * 0.14, Math.abs(mid) * 0.00025);
  return { min: yMin - yPad, max: yMax + yPad };
}

(function injectPredictionChartWaitingStyles() {
  if (typeof document === "undefined" || document.getElementById("predictionChartWaitingSpinStyle")) return;
  const s = document.createElement("style");
  s.id = "predictionChartWaitingSpinStyle";
  s.textContent =
    "@keyframes predChartSpin{to{transform:rotate(360deg)}} .prediction-chart-ai-waiting{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;background:rgba(0,0,0,0.62);color:#67e8f9;font-family:'JetBrains Mono',ui-monospace,monospace;font-size:14px;font-weight:600;z-index:4;pointer-events:none;letter-spacing:0.02em}" +
    ".prediction-chart-ai-waiting[hidden]{display:none!important}" +
    ".prediction-chart-ai-waiting .prediction-chart-ai-waiting__spin{width:26px;height:26px;border:3px solid rgba(103,232,249,0.22);border-top-color:#67e8f9;border-radius:50%;animation:predChartSpin 0.75s linear infinite}";
  document.head.appendChild(s);
})();

function ensurePredictionChartWaitingOverlay() {
  const canvas = document.getElementById("tradingPredictionChart");
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
}

function setPredictionChartWaitingVisible(show) {
  const el = ensurePredictionChartWaitingOverlay();
  if (!el) return;
  el.hidden = !show;
}

window.computePredictionChartYBounds = computePredictionChartYBounds;
window.ensurePredictionChartWaitingOverlay = ensurePredictionChartWaitingOverlay;
window.setPredictionChartWaitingVisible = setPredictionChartWaitingVisible;

/** Optioneel: RL buy/sell-kanteling op voorspellingspunten. Niet meer gebruikt op Chart.js — liever zelfde curve als LW-overlay (`refreshMainChartGhostLine`). */
function applyRlSellConfidenceToPredictedPoints(predictedPoints, dataPayload) {
  if (!Array.isArray(predictedPoints) || predictedPoints.length < 2) return predictedPoints;
  const ap = dataPayload && dataPayload.ai_action_probs;
  const rld = dataPayload && dataPayload.rl_last_decision;
  let buy = NaN;
  let sell = NaN;
  if (ap && typeof ap === "object") {
    buy = Number(ap.buy_pct);
    sell = Number(ap.sell_pct);
  }
  if (dataPayload && dataPayload.buy != null) {
    buy = Number(dataPayload.buy);
    sell = Number(dataPayload.sell);
  }
  if (!Number.isFinite(buy) && rld && typeof rld === "object") buy = Number(rld.prob_buy) * 100;
  if (!Number.isFinite(sell) && rld && typeof rld === "object") sell = Number(rld.prob_sell) * 100;
  if (!Number.isFinite(buy)) buy = 33.33;
  if (!Number.isFinite(sell)) sell = 33.33;
  let conf = Number(dataPayload && dataPayload.rl_confidence);
  if (!Number.isFinite(conf) && rld) conf = Number(rld.confidence);
  if (!Number.isFinite(conf)) conf = 0.5;
  conf = Math.min(1, Math.max(0, conf));
  const bear = (sell - buy) / 100;
  const mag = conf * 0.055;
  return predictedPoints.map((p, i) => {
    if (i === 0) return p;
    const t = i / (predictedPoints.length - 1);
    const mult = 1 - bear * mag * t;
    return { x: p.x, y: Math.max(1e-12, Number(p.y) * mult) };
  });
}

window.__terminalPredictionChartHelpers = {
  buildChartJsRealPriceFromMainLw,
  formatPredictionChartTickTime,
  inferChartJsYDecimalsFromRange,
  computePredictionChartYBounds,
  applyRlSellConfidenceToPredictedPoints,
};
window.formatPredictionChartTickTime = formatPredictionChartTickTime;

function initTradingPredictionChart() {
  const canvas = document.getElementById("tradingPredictionChart");
  if (!canvas || typeof Chart === "undefined") return null;
  const U = window.ChartUtils;
  if (U && U.registry && U.registry.tradingPredictionChart) {
    tradingChart = U.registry.tradingPredictionChart;
    window.tradingChart = tradingChart;
    return tradingChart;
  }
  if (tradingChart) {
    window.tradingChart = tradingChart;
    return tradingChart;
  }

  const lineData = {
    datasets: [
      {
        label: "Real Price",
        data: [],
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
        data: [],
        parsing: false,
        borderColor: "#67e8f9",
        backgroundColor: "transparent",
        borderWidth: 3,
        borderDash: [7, 5],
        borderCapStyle: "round",
        pointRadius: 0,
        tension: 0.18,
        fill: false,
      },
    ],
  };

  const baseOptions = highContrastChartOptions({
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
        labels: {
          boxWidth: 12,
          padding: 10,
          usePointStyle: false,
        },
      },
    },
    scales: {
      x: {
        type: "linear",
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
          autoSkip: true,
          padding: 6,
          callback(v) {
            return formatPredictionChartTickTime(v);
          },
        },
      },
      y: {
        beginAtZero: false,
        ticks: {
          maxTicksLimit: 10,
        },
      },
    },
  });

  if (U && typeof U.upsertChart === "function") {
    tradingChart = U.upsertChart("tradingPredictionChart", {
      type: "line",
      data: lineData,
      options: baseOptions,
    });
    window.tradingChart = tradingChart;
    return tradingChart;
  }

  const ctx = canvas.getContext("2d");
  tradingChart = new Chart(ctx, {
    type: "line",
    data: lineData,
    options: baseOptions,
  });
  window.tradingChart = tradingChart;
  return tradingChart;
}

async function updatePredictionChart(symbol) {
  const symU = _normMkChart(symbol || selectedMarket || "BTC-EUR");
  const sym = encodeURIComponent(symU);
  try {
    const response = await fetch(`/api/v1/predictions?symbol=${sym}`);
    if (!response.ok) throw new Error(`Netwerkfout bij ophalen predicties (${response.status})`);
    const data = await response.json();
    const dataPolicy = mergePolicySnapshotIntoPredictionData(symU, data);

    const hist = Array.isArray(dataPolicy.historical) ? dataPolicy.historical : [];
    const pred = Array.isArray(dataPolicy.predicted) ? dataPolicy.predicted : [];
    const lwSeries = buildChartJsRealPriceFromMainLw(symU);
    if (!hist.length && !(lwSeries && lwSeries.length)) {
      throw new Error("Geen historische punten in API-response");
    }

    let historicalPoints = hist.length
      ? hist
          .map((item) => {
            const x = new Date(item.timestamp);
            const y = Number(item.price);
            return { x: x.getTime(), y };
          })
          .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
      : [];

    if (lwSeries && lwSeries.length) {
      historicalPoints = lwSeries;
    } else if (historicalPoints.length && Array.isArray(rawChartLineData) && rawChartLineData.length && hist.length) {
      const cur = _normMkChart(selectedChartPair || selectedMarket || "");
      if (cur === symU) {
        const anchor = rawChartLineData[rawChartLineData.length - 1];
        const yA = Number(anchor && anchor.value);
        if (Number.isFinite(yA) && yA > 0) {
          const last = historicalPoints[historicalPoints.length - 1];
          historicalPoints[historicalPoints.length - 1] = { ...last, y: yA };
        }
      }
    }

    if (!historicalPoints.length) throw new Error("Geen bruikbare prijspunten voor voorspellingsgrafiek");

    const lastHistoricalPoint = historicalPoints[historicalPoints.length - 1];

    let predictedPoints = pred
      .map((item) => {
        const x = new Date(item.timestamp);
        const y = Number(item.predicted_price);
        return { x: x.getTime(), y };
      })
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));

    if (lastHistoricalPoint && predictedPoints.length) {
      predictedPoints.unshift({ x: lastHistoricalPoint.x, y: lastHistoricalPoint.y });
    }

    predictedPoints = applyRlSellConfidenceToPredictedPoints(predictedPoints, dataPolicy);
    setPredictionChartWaitingVisible(predictedPoints.length === 0);

    const chart = initTradingPredictionChart();
    if (chart) {
      chart.data.datasets[0].data = historicalPoints;
      chart.data.datasets[1].data = predictedPoints;
      if (chart.data.datasets[1]) chart.data.datasets[1].borderWidth = 3;
      const merged = [...historicalPoints, ...predictedPoints];
      const allXs = merged.map((p) => p.x).filter(Number.isFinite);
      if (allXs.length) {
        const tMin = Math.min(...allXs);
        const tMax = Math.max(...allXs);
        const span = Math.max(60_000, tMax - tMin);
        const pad = Math.max(60_000, span * 0.12);
        chart.options.scales = chart.options.scales || {};
        chart.options.scales.x = chart.options.scales.x || {};
        chart.options.scales.x.type = "linear";
        chart.options.scales.x.suggestedMin = tMin - pad * 0.05;
        chart.options.scales.x.suggestedMax = tMax + pad;
        chart.options.scales.x.ticks = chart.options.scales.x.ticks || {};
        chart.options.scales.x.ticks.maxTicksLimit = 12;
        chart.options.scales.x.ticks.callback = (v) => formatPredictionChartTickTime(v);
      }
      const yb = computePredictionChartYBounds(historicalPoints, predictedPoints);
      if (yb) {
        const prec = inferChartJsYDecimalsFromRange(yb.min, yb.max);
        chart.options.scales.y = {
          ...(chart.options.scales.y || {}),
          beginAtZero: false,
          min: yb.min,
          max: yb.max,
          suggestedMin: undefined,
          suggestedMax: undefined,
          ticks: {
            ...(chart.options.scales.y && chart.options.scales.y.ticks ? chart.options.scales.y.ticks : {}),
            maxTicksLimit: 8,
            callback(v) {
              const nv = Number(v);
              if (!Number.isFinite(nv)) return "";
              return nv.toLocaleString("nl-NL", { minimumFractionDigits: prec, maximumFractionDigits: prec });
            },
          },
        };
      }
      try {
        if (typeof chart.resize === "function") chart.resize();
      } catch (_e) {
        /* noop */
      }
      chart.update("none");
      scheduleSyncPredictionChartJsXFromLightweight();
    }

    const predEl = document.getElementById("prediction");
    if (predEl && historicalPoints.length) {
      const tx = String(predEl.textContent || "");
      if (/^laden/i.test(tx) || tx.includes("Laden…") || tx.includes("Laden...")) predEl.textContent = "";
    }
    if (typeof setChartTimeHint === "function" && historicalPoints.length) {
      const hiRow = hist.length ? hist[hist.length - 1] : null;
      const prRow = pred.length ? pred[pred.length - 1] : null;
      setChartTimeHint({
        market: symU,
        updated_at: hiRow && hiRow.timestamp != null ? String(hiRow.timestamp) : new Date().toISOString(),
        predicted_at: prRow && prRow.timestamp != null ? String(prRow.timestamp) : null,
      });
    }

    const accuracyElement = document.getElementById("ai-accuracy-score");
    if (accuracyElement) {
      const raw = data.accuracy_score;
      if (raw != null && Number.isFinite(Number(raw))) {
        const ac = Number(raw);
        accuracyElement.innerText = `${ac.toFixed(1)}%`;
        accuracyElement.style.color = ac > 80 ? "#00ff88" : "#ff3e3e";
      } else {
        accuracyElement.innerText = "—";
        accuracyElement.style.color = "#888888";
      }
    }
  } catch (error) {
    console.error("Fout bij updaten van de AI voorspellingsgrafiek:", error);
    try {
      setPredictionChartWaitingVisible(true);
    } catch (_e) {
      /* noop */
    }
  }
}

function ensureBrainHintPortal() {
  let el = document.getElementById(HINT_PORTAL_ID);
  if (!el) {
    el = document.createElement("div");
    el.id = HINT_PORTAL_ID;
    el.setAttribute("aria-hidden", "true");
    document.body.appendChild(el);
  }
  return el;
}

function hideBrainHintPortal() {
  const el = document.getElementById(HINT_PORTAL_ID);
  if (el) el.innerHTML = "";
}

function positionBrainHintPortal(wrap, bubbleEl) {
  const text = (bubbleEl && bubbleEl.textContent) ? bubbleEl.textContent.trim() : "";
  if (!text) return;
  const rect = wrap.getBoundingClientRect();
  const portal = ensureBrainHintPortal();
  portal.innerHTML = "";
  const div = document.createElement("div");
  div.className = "hint-bubble hint-bubble--portal";
  div.textContent = text;
  portal.appendChild(div);
  const vw = window.innerWidth;
  const maxW = Math.min(420, vw - 16);
  div.style.cssText = [
    "position:fixed",
    "z-index:99999",
    `max-width:${maxW}px`,
    "background:rgba(0,0,0,0.55)",
    "color:#ffffff",
    "border:1px solid rgba(255,255,255,0.2)",
    "border-radius:4px",
    "padding:8px 10px",
    "font-size:14px",
    "line-height:1.35",
    "backdrop-filter:blur(5px)",
    "-webkit-backdrop-filter:blur(5px)",
    "box-shadow:0 8px 28px rgba(0,0,0,0.45)",
    "pointer-events:none",
  ].join(";");
  const left = Math.min(Math.max(8, rect.left + rect.width / 2 - maxW / 2), vw - maxW - 8);
  let top = rect.bottom + 8;
  div.style.left = `${left}px`;
  div.style.top = `${top}px`;
  requestAnimationFrame(() => {
    const h = div.offsetHeight;
    if (top + h > window.innerHeight - 8) {
      top = Math.max(8, rect.top - h - 8);
      div.style.top = `${top}px`;
    }
  });
}

function initHintPortals() {
  const roots = [
    document.body,
    document.getElementById("globalEliteTickerStrip"),
    document.getElementById("tab-terminal"),
    document.getElementById("tab-aibrain"),
    document.getElementById("tab-ledger"),
    document.getElementById("tab-hardware"),
  ].filter(Boolean);
  for (const root of roots) {
    root.querySelectorAll(".hint-wrap").forEach((wrap) => {
      const bubble = wrap.querySelector(".hint-bubble");
      if (!bubble || wrap.dataset.hintPortalBound === "1") return;
      wrap.dataset.hintPortalBound = "1";
      const show = () => positionBrainHintPortal(wrap, bubble);
      const hide = () => hideBrainHintPortal();
      wrap.addEventListener("mouseenter", show);
      wrap.addEventListener("mouseleave", hide);
      wrap.addEventListener("focusin", show);
      wrap.addEventListener("focusout", hide);
    });
  }
  if (!window.__genesisHintScrollBound) {
    window.__genesisHintScrollBound = true;
    window.addEventListener(
      "scroll",
      () => {
        hideBrainHintPortal();
      },
      true
    );
  }
}

window.addEventListener("resize", () => hideBrainHintPortal());

function signalClass(signal) {
  const s = String(signal || "").toLowerCase();
  if (s === "buy") return "positive";
  if (s === "sell") return "negative";
  return "";
}

function setLiveUpdatedAt() {
  const now = new Date();
  setClassText(".js-live-updated-at", `Laatste update: ${now.toLocaleTimeString()}`);
}

function renderModeBanner(mode) {
  const el = document.getElementById("modeBanner");
  if (!el) return;
  const live = String(mode || "").toLowerCase() === "live";
  el.className = `cockpit-mode-pill mode-banner ${live ? "mode-live" : "mode-paper"}`;
  el.textContent = live ? "LIVE" : "PAPER";
}

function sentimentToneClass(value) {
  return Number(value) >= 0 ? "positive" : "negative";
}

function bytesToMB(v) {
  return (Number(v || 0) / (1024 * 1024)).toFixed(2);
}

function formatNewsTimeAmsterdam(ts) {
  if (!ts) return "-";
  const dt = new Date(ts);
  if (!Number.isFinite(dt.getTime())) return String(ts);
  const absolute = new Intl.DateTimeFormat("nl-NL", {
    timeZone: "Europe/Amsterdam",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(dt);
  const diffSec = Math.max(0, Math.floor((Date.now() - dt.getTime()) / 1000));
  if (diffSec < 60) return `${absolute} (zojuist)`;
  if (diffSec < 3600) return `${absolute} (${Math.floor(diffSec / 60)}m geleden)`;
  return `${absolute} (${Math.floor(diffSec / 3600)}u geleden)`;
}

/** Korte relatieve tijd voor cockpit-nieuws (bijv. "5m geleden"). */
function formatNewsRelativeShort(ts) {
  if (!ts) return "—";
  const dt = new Date(ts);
  if (!Number.isFinite(dt.getTime())) return "—";
  const diffSec = Math.max(0, Math.floor((Date.now() - dt.getTime()) / 1000));
  if (diffSec < 60) return "zojuist";
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m geleden`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}u geleden`;
  return `${Math.floor(diffSec / 86400)}d geleden`;
}

function escapeHtmlText(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/**
 * BTC-posities: soms `position_qty` in satoshi's i.p.v. hele BTC.
 * Corrigeer vóór allocatie: (qty × prijs) / saldo × 100.
 */
function normalizePositionQtyForAllocation(market, qtyRaw, priceEur, totalBalanceEur) {
  let q = Number(qtyRaw);
  if (!Number.isFinite(q) || q <= 0) return q;
  const mku = String(market || "").toUpperCase();
  if (!mku.startsWith("BTC-")) return q;
  const px = Number(priceEur);
  const bal = Number(totalBalanceEur);
  if (!Number.isFinite(px) || px <= 0 || !Number.isFinite(bal) || bal <= 0) return q;
  let actualAllocation = (q * px) / bal * 100;
  if (actualAllocation > 250 || (q >= 1e5 && actualAllocation > 150)) {
    q /= 1e8;
    actualAllocation = (q * px) / bal * 100;
  }
  return q;
}

/** Markten met open paper-positie (qty > 0) — scanner + positielampje in header. */
function buildMarketsInPositionSet(data) {
  const p = (data && data.paper_portfolio) || {};
  const out = new Set();
  const pbm = p.position_by_market && typeof p.position_by_market === "object" ? p.position_by_market : null;
  if (pbm && Object.keys(pbm).length) {
    for (const [k, raw] of Object.entries(pbm)) {
      const q = typeof raw === "number" ? raw : Number(raw);
      const mku = String(k || "").toUpperCase().trim();
      if (mku && Number.isFinite(q) && q > 0) out.add(mku);
    }
  }
  if (out.size === 0) {
    const sym = String(p.position_symbol || "").toUpperCase();
    const qty = Number(p.position_qty);
    if (sym && sym !== "NONE" && Number.isFinite(qty) && qty > 0) out.add(sym);
  }
  return out;
}

/**
 * Allocatie % = (Pos Qty × prijs EUR) / totaal saldo (equity) × 100.
 * Bij corrupte qty×prijs (>> saldo) val terug op serverregel voor die markt.
 */
function allocationDisplayLines(data) {
  const alloc = (data && data.allocation_snapshot) || {};
  const serverLines = Array.isArray(alloc.lines) ? alloc.lines : [];
  const p = (data && data.paper_portfolio) || {};
  const eq = Number(p.equity);
  const cashN = Number(p.cash);
  let totalSaldo = Number.isFinite(eq) && eq > 0 ? eq : 0;
  if (totalSaldo <= 0 && Number.isFinite(cashN) && cashN > 0) totalSaldo = cashN;
  if (totalSaldo <= 0) return serverLines;

  const lpm = p.last_prices_by_market && typeof p.last_prices_by_market === "object" ? p.last_prices_by_market : {};
  const posSym = String(p.position_symbol || "").toUpperCase();
  const am = Array.isArray(data.active_markets) ? data.active_markets : [];

  function priceForMarket(mku) {
    let px = Number(lpm[mku]);
    if (!Number.isFinite(px) || px <= 0) {
      if (mku === posSym) px = Number(p.last_price);
    }
    if (!Number.isFinite(px) || px <= 0) {
      for (let i = 0; i < am.length; i++) {
        const row = am[i];
        if (!row || String(row.market || "").toUpperCase() !== mku) continue;
        const lp = row.last_price != null ? row.last_price : row.price;
        const n = Number(lp);
        if (Number.isFinite(n) && n > 0) {
          px = n;
          break;
        }
      }
    }
    return Number.isFinite(px) && px > 0 ? px : 0;
  }

  const pbm = p.position_by_market && typeof p.position_by_market === "object" ? p.position_by_market : null;
  const entries =
    pbm && Object.keys(pbm).length
      ? Object.entries(pbm)
      : (() => {
          const qty = Number(p.position_qty);
          if (!posSym || posSym === "NONE" || !Number.isFinite(qty) || qty <= 0) return [];
          return [[posSym, qty]];
        })();

  const out = [];
  for (let i = 0; i < entries.length; i++) {
    const mku = String(entries[i][0] || "").toUpperCase();
    let qty = Number(entries[i][1]);
    const px = priceForMarket(mku);
    if (!Number.isFinite(qty) || qty <= 0 || px <= 0) continue;
    qty = normalizePositionQtyForAllocation(mku, qty, px, totalSaldo);
    const posEur = qty * px;
    let actualAllocation = (posEur / totalSaldo) * 100;
    if (!Number.isFinite(actualAllocation) || actualAllocation < 0) actualAllocation = 0;
    if (actualAllocation > 500) {
      const prev = serverLines.find((r) => String((r && r.market) || "").toUpperCase() === mku);
      if (prev && Number.isFinite(Number(prev.weight_pct))) {
        out.push({
          ...prev,
          weight_pct: Math.round(Math.min(100, Math.max(0, Number(prev.weight_pct))) * 100) / 100,
        });
      }
      continue;
    }
    const wPct = Math.round(actualAllocation * 100) / 100;
    const base = mku.includes("-") ? mku.split("-")[0] : mku;
    const prev = serverLines.find((r) => String((r && r.market) || "").toUpperCase() === mku);
    out.push({
      market: mku,
      coin: base,
      weight_pct: wPct,
      notional_eur: Math.round(posEur * 100) / 100,
      in_position: prev ? Boolean(prev.in_position) : true,
    });
  }
  return out.length ? out : serverLines;
}

/** Eén tijdstempel + tekst: strip herhaalde ``ISO [LEVEL]`` prefixes (Python logger + dubbele injectie). */
function bvWorkerHintDisplayParts(raw) {
  let s = String(raw || "").trim();
  const re = /^(\d{4}-\d{2}-\d{2}T[\d:+\-Z.]+)\s+\[[^\]]+\]\s*/i;
  let lastClock = null;
  for (let i = 0; i < 4; i += 1) {
    const m = s.match(re);
    if (!m) break;
    lastClock = m[1];
    s = s.slice(m[0].length).trim();
  }
  return { clock: lastClock, text: s || String(raw || "").trim() };
}

function formatBvAiTradeClock(d) {
  try {
    return new Intl.DateTimeFormat("nl-NL", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(d);
  } catch (_e) {
    return "—";
  }
}

function bvAiTradeRowClass(text) {
  const t = String(text || "").toUpperCase();
  if (/\b(BUY|BULL|LONG|KOOP|POSITIEF|STIJG|▲)\b/.test(t) || (t.includes("BUY") && !t.includes("SELL"))) {
    return "bv-ai-trade-row--pos";
  }
  if (/\b(SELL|BEAR|SHORT|VERKOOP|NEGATIEF|PANIC|DAL|▼)\b/.test(t) || /\bSELL\b/.test(t)) {
    return "bv-ai-trade-row--neg";
  }
  return "bv-ai-trade-row--neu";
}

function bvAiTradeSideGlyph(text) {
  const t = String(text || "").toUpperCase();
  if ((/\bBUY\b/.test(t) || t.includes("BUY")) && !t.includes("SELL")) return "▲";
  if (/\bSELL\b/.test(t) || t.includes("SELL")) return "▼";
  return "·";
}

// Freeze / auto-scroll / HOLD-groepering state voor bvAiThoughtFeed
let _bvLogsFrozen = false;
let _bvLogsPinned = false;
let _bvLogsPrevProbs = null; // { buy, hold, sell }

(function _initBvLogControls() {
  document.addEventListener("DOMContentLoaded", function () {
    const freezeBtn = document.getElementById("bvLogsFreezeBtn");
    if (freezeBtn && !freezeBtn.dataset.bvBound) {
      freezeBtn.dataset.bvBound = "1";
      freezeBtn.addEventListener("click", function () {
        _bvLogsFrozen = !_bvLogsFrozen;
        freezeBtn.textContent = _bvLogsFrozen ? "Unfreeze" : "Freeze";
        freezeBtn.style.color = _bvLogsFrozen ? "#39ff14" : "";
        freezeBtn.style.borderColor = _bvLogsFrozen ? "#39ff14" : "";
      });
    }
    const feed = document.getElementById("bvAiThoughtFeed");
    if (feed && !feed.dataset.scrollBound) {
      feed.dataset.scrollBound = "1";
      feed.addEventListener("scroll", function () {
        const nearBottom = feed.scrollHeight - (feed.scrollTop + feed.clientHeight) < 48;
        _bvLogsPinned = !nearBottom;
      });
    }
  });
})();

/** Actieve-berekeningen feed: layout + white-space via CSS (bv-ai-trades-feed--scroll-body). */
function renderBvThoughtFeed(data) {
  if (_bvLogsFrozen) return;
  const root = document.getElementById("bvAiThoughtFeed");
  if (!root) return;
  if (!payloadSymbolMatchesActive(data)) return;
  const activeMarket = getActiveMarketForPolicyUi();
  const activeU = normPairKey(activeMarket);
  const chunks = [];
  const hintsByMarket =
    data && data.worker_calc_hints_by_market && typeof data.worker_calc_hints_by_market === "object"
      ? data.worker_calc_hints_by_market
      : null;
  const perHints = resolveMultiMapEntry(hintsByMarket, activeMarket);
  let hintsRaw = [];
  if (Array.isArray(perHints) && perHints.length) {
    hintsRaw = perHints;
  } else if (Array.isArray(data && data.worker_calc_hints)) {
    hintsRaw = data.worker_calc_hints.filter((x) => workerHintLineMatchesPair(x, activeU));
  }
  const hints = hintsRaw
    .map((x) => String(x || "").trim())
    .filter((x) => x.length > 4);
  hints.slice(-6).forEach((h) => chunks.push(h));
  let rld = resolveMultiMapEntry(data && data.rl_multi_decisions, activeMarket);
  if (!rld || typeof rld !== "object") rld = data && data.rl_last_decision;
  if (rld && typeof rld === "object") {
    const dk = normPairKey(rld.market || rld.ticker || rld.symbol || "");
    if (dk && dk !== activeU) rld = null;
  }
  if (rld && typeof rld === "object" && typeof rld.reasoning === "string") {
    const raw = String(rld.reasoning || "").trim();
    if (raw) {
      const parts = raw.split(/\.\s+/).map((x) => x.trim()).filter((x) => x.length > 12);
      for (const p of parts.slice(-12)) {
        chunks.push(p.endsWith(".") ? p : `${p}.`);
      }
    }
  }
  if (!chunks.length && data && typeof data.decision_reasoning === "string") {
    const scopeMk = normPairKey(data.market || data.ticker || "");
    if (!scopeMk || scopeMk === activeU) {
      const dr = String(data.decision_reasoning || "").trim();
      if (dr.length > 20) {
        dr.split(/\n+/)
          .map((x) => x.trim())
          .filter((x) => x.length > 15)
          .slice(-6)
          .forEach((line) => chunks.push(line));
      }
    }
  }
  const lp = data && data.last_prediction;
  if (lp && typeof lp === "object") {
    const sig = String(lp.signal || "").toUpperCase();
    const tk = normPairKey(lp.ticker || "");
    if (sig && tk && tk === activeU) chunks.push(`Signaal ${sig} · ${tk}`);
  }
  // Importance filter: extract current policy probs and check for >5% shift
  let rldForProbs = data && data.rl_multi_decisions
    ? resolveMultiMapEntry(data.rl_multi_decisions, getActiveMarketForPolicyUi())
    : null;
  if (!rldForProbs) rldForProbs = data && data.rl_last_decision;
  const curProbs = rldForProbs ? {
    buy: Number(rldForProbs.prob_buy || 0),
    hold: Number(rldForProbs.prob_hold || 0),
    sell: Number(rldForProbs.prob_sell || 0),
  } : null;
  const sigFilterEl = document.getElementById("bvLogsSignificantOnly");
  const sigFilterOn = sigFilterEl && sigFilterEl.checked;
  if (sigFilterOn && curProbs && _bvLogsPrevProbs) {
    const delta = Math.max(
      Math.abs(curProbs.buy - _bvLogsPrevProbs.buy),
      Math.abs(curProbs.hold - _bvLogsPrevProbs.hold),
      Math.abs(curProbs.sell - _bvLogsPrevProbs.sell)
    );
    if (delta < 0.05) return; // minder dan 5% verschil — skip render
  }
  if (curProbs) _bvLogsPrevProbs = curProbs;

  // HOLD-grouping: verklein opeenvolgende HOLD-regels tot één met teller
  const rawChunks = chunks.slice(-20);
  const grouped = [];
  for (const t of rawChunks) {
    const isHold = /Uitgevoerde actie:\s*HOLD/i.test(t) || /^Signaal HOLD/i.test(t);
    const last = grouped[grouped.length - 1];
    if (isHold && last && last._holdGroup) {
      last._holdCount = (last._holdCount || 1) + 1;
      last.text = last._baseText + ` (×${last._holdCount})`;
    } else {
      const entry = { text: t, _holdGroup: isHold, _baseText: t, _holdCount: 1 };
      grouped.push(entry);
    }
  }
  const show = grouped.slice(-10).map(e => e.text);

  if (!show.length) {
    root.innerHTML = `<div class="bv-ai-trade-row bv-ai-trade-row--neu bv-ai-trade-row--empty"><span class="bv-ai-trade-ts">—</span><span class="bv-ai-trade-txt">Nog geen worker-updates.</span><span class="bv-ai-trade-side">·</span></div>`;
    if (typeof window.__rlHeartbeatRefresh === "function") window.__rlHeartbeatRefresh();
    return;
  }
  const baseMs = Date.now();
  root.innerHTML = show
    .map((t, idx) => {
      const parts = bvWorkerHintDisplayParts(t);
      const rowCls = bvAiTradeRowClass(parts.text);
      const txtRaw = parts.text;
      const hlCls =
        /\bconfidence\b/i.test(txtRaw) || /\bsoftmax\b/i.test(txtRaw) ? " bv-ai-trade-row--confidence-highlight" : "";
      let ts;
      try {
        ts = parts.clock
          ? formatBvAiTradeClock(new Date(parts.clock))
          : formatBvAiTradeClock(new Date(baseMs - (show.length - 1 - idx) * 900));
      } catch (_e) {
        ts = "—";
      }
      const side = escapeHtmlText(bvAiTradeSideGlyph(parts.text));
      return `<div class="bv-ai-trade-row ${rowCls}${hlCls}"><span class="bv-ai-trade-ts">${escapeHtmlText(ts)}</span><span class="bv-ai-trade-txt">${escapeHtmlText(parts.text)}</span><span class="bv-ai-trade-side">${side}</span></div>`;
    })
    .join("");
  // Auto-scroll: stop als gebruiker handmatig omhoog heeft gescrolld
  if (!_bvLogsPinned) root.scrollTop = root.scrollHeight;
  if (typeof window.__rlHeartbeatRefresh === "function") window.__rlHeartbeatRefresh();
}

function normalizeProbToPercent(x) {
  const n = Number(x);
  if (!Number.isFinite(n) || n < 0) return null;
  return n <= 1.0 + 1e-9 ? n * 100.0 : Math.min(100, n);
}

/** Zelfde sleutelvorm als backend (BTC-EUR). */
function normPairKey(s) {
  return String(s || "").toUpperCase().replace("/", "-");
}

/** Zoekt waarde in rl_multi_decisions / worker_calc_hints_by_market ongeacht slash/hoofdletters. */
function resolveMultiMapEntry(obj, market) {
  if (!obj || typeof obj !== "object") return null;
  const u = normPairKey(market);
  const raw = String(market || "").toUpperCase();
  if (Object.prototype.hasOwnProperty.call(obj, raw)) return obj[raw];
  if (Object.prototype.hasOwnProperty.call(obj, u)) return obj[u];
  for (const k of Object.keys(obj)) {
    if (normPairKey(k) === u) return obj[k];
  }
  return null;
}

/** Chart.js voorspellingslijn: gebruik actuele policy-softmax (dashboard) i.p.v. alleen poll-JSON — blijft meebewegen met koers/policy. */
function mergePolicySnapshotIntoPredictionData(symU, apiData) {
  const out = { ...(apiData && typeof apiData === "object" ? apiData : {}) };
  const st = window.__lastDashboardStats;
  if (!st || typeof st !== "object") return out;
  const m = normPairKey(symU);
  const multi = st.rl_multi_decisions;
  let d = null;
  if (multi && typeof multi === "object") {
    d = resolveMultiMapEntry(multi, m);
  }
  if (d && typeof d === "object") {
    out.rl_last_decision = { ...d, market: m, ticker: m };
    out.ai_action_probs = {
      buy_pct: Number(d.prob_buy) * 100,
      hold_pct: Number(d.prob_hold) * 100,
      sell_pct: Number(d.prob_sell) * 100,
      market: m,
      ticker: m,
    };
    return out;
  }
  if (st.rl_last_decision && typeof st.rl_last_decision === "object") {
    const g = st.rl_last_decision;
    const gk = normPairKey(g.market || g.ticker || g.symbol || "");
    if (!gk || gk === m) {
      out.rl_last_decision = g;
      if (st.ai_action_probs && typeof st.ai_action_probs === "object") out.ai_action_probs = st.ai_action_probs;
    }
  }
  return out;
}

/** Globale worker-hints: alleen tonen als regel naar dit paar verwijst (voorkomt dezelfde RSI-tekst op elke tab). */
function workerHintLineMatchesPair(text, activeU) {
  const t = String(text || "").toUpperCase();
  const u = normPairKey(activeU);
  if (!u || t.length < 5) return false;
  if (t.includes(u)) return true;
  const alt = u.replace(/-/g, "/");
  if (alt !== u && t.includes(alt.toUpperCase())) return true;
  return false;
}

function getActiveMarketForPolicyUi() {
  const fromSelect = document.getElementById("marketSelect")?.value;
  const fromState = window.AppCore && window.AppCore.state ? window.AppCore.state.selectedMarket : null;
  return String(fromSelect || fromState || selectedMarket || "BTC-EUR").toUpperCase();
}

function inferIncomingSymbol(raw) {
  if (!raw || typeof raw !== "object") return "";
  const activeU = normPairKey(getActiveMarketForPolicyUi());
  const multi = raw.rl_multi_decisions && typeof raw.rl_multi_decisions === "object" ? raw.rl_multi_decisions : null;
  if (multi && activeU) {
    if (multi[activeU] && typeof multi[activeU] === "object") return activeU;
    for (const k of Object.keys(multi)) {
      if (normPairKey(k) === activeU) return activeU;
    }
  }
  const direct = raw.symbol || raw.market || raw.ticker;
  if (direct) return normPairKey(direct);
  const lp = raw.last_prediction && typeof raw.last_prediction === "object" ? raw.last_prediction : null;
  if (lp && lp.ticker) return normPairKey(lp.ticker);
  const lo = raw.last_order && typeof raw.last_order === "object" ? raw.last_order : null;
  const ord = lo && lo.order && typeof lo.order === "object" ? lo.order : null;
  if (ord && (ord.ticker || ord.market)) return normPairKey(ord.ticker || ord.market);
  const ap = raw.ai_action_probs && typeof raw.ai_action_probs === "object" ? raw.ai_action_probs : null;
  if (ap && (ap.market || ap.ticker || ap.symbol)) return normPairKey(ap.market || ap.ticker || ap.symbol);
  return "";
}

function payloadHasActiveSymbolData(raw, activeU) {
  if (!raw || typeof raw !== "object") return false;
  const multi = raw.rl_multi_decisions && typeof raw.rl_multi_decisions === "object" ? raw.rl_multi_decisions : null;
  if (multi) {
    if (multi[activeU] && typeof multi[activeU] === "object") return true;
    for (const k of Object.keys(multi)) {
      if (normPairKey(k) === activeU && multi[k] && typeof multi[k] === "object") return true;
    }
  }
  const hints = raw.worker_calc_hints_by_market && typeof raw.worker_calc_hints_by_market === "object" ? raw.worker_calc_hints_by_market : null;
  if (hints) {
    if (Array.isArray(hints[activeU]) && hints[activeU].length) return true;
    for (const k of Object.keys(hints)) {
      if (normPairKey(k) === activeU && Array.isArray(hints[k]) && hints[k].length) return true;
    }
  }
  return false;
}

function payloadSymbolMatchesActive(raw) {
  const activeU = normPairKey(getActiveMarketForPolicyUi());
  const incomingU = inferIncomingSymbol(raw);
  if (!incomingU) return true;
  if (incomingU === activeU) return true;
  return payloadHasActiveSymbolData(raw, activeU);
}

function resetMarketScopedUi(market) {
  const mk = normPairKey(market || getActiveMarketForPolicyUi());
  const feed = document.getElementById("bvAiThoughtFeed");
  if (feed) {
    feed.innerHTML =
      `<div class="bv-ai-trade-row bv-ai-trade-row--neu bv-ai-trade-row--empty">` +
      `<span class="bv-ai-trade-ts">—</span>` +
      `<span class="bv-ai-trade-txt">Wisselen naar ${escapeHtmlText(mk)}… wacht op markt-specifieke updates.</span>` +
      `<span class="bv-ai-trade-side">·</span>` +
      `</div>`;
  }
  const setProb = (valueId, barId) => {
    const v = document.getElementById(valueId);
    const b = document.getElementById(barId);
    if (v) v.textContent = "—";
    if (b) b.style.width = "0%";
  };
  setProb("hpAiProbBuy", "hpAiProbBuyBar");
  setProb("hpAiProbHold", "hpAiProbHoldBar");
  setProb("hpAiProbSell", "hpAiProbSellBar");
  setClassText(".js-sentiment-value", "—");
  setClassText(".js-sentiment-confidence", "—");
  document.querySelectorAll(".js-sentiment-bar").forEach((el) => {
    el.style.width = "0%";
  });
  if (window.TerminalLiveTail && typeof window.TerminalLiveTail.clear === "function") {
    window.TerminalLiveTail.clear();
  }
  if (typeof window.updatePayloadFilterDebugBadge === "function") {
    window.updatePayloadFilterDebugBadge({ active: mk, incoming: "", dropped: false, src: "switch" });
  }
  const brainWrap = document.getElementById("hpAiBrainPulseWrap");
  const brainRo = document.getElementById("hpAiBrainConfReadout");
  if (brainWrap) brainWrap.classList.remove("hp-ai-brain-pulse--strong");
  if (brainRo) brainRo.textContent = "";
}

/** LW UTCTimestamp = unix seconden; voorspellings-API kan per ongeluk ms doorgeven. */
function normalizeChartTimeSeconds(t) {
  if (t == null) return t;
  const n = Number(t);
  if (!Number.isFinite(n)) return t;
  if (n > 1e12) return Math.floor(n / 1000);
  return n;
}

/** Prijsas: voorkomt dat alles als 0.08 wordt afgerond bij munten < €1. */
function inferLwPriceFormatFromSamples(values) {
  const finite = (values || []).filter((v) => Number.isFinite(v) && v > 0);
  if (!finite.length) return { type: "price", precision: 2, minMove: 0.01 };
  const sorted = finite.slice().sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)] || sorted[0];
  const abs = Math.abs(Number(med));
  if (abs >= 50000) return { type: "price", precision: 0, minMove: 1 };
  if (abs >= 1000) return { type: "price", precision: 2, minMove: 0.01 };
  if (abs >= 1) return { type: "price", precision: 4, minMove: 0.0001 };
  if (abs >= 0.01) return { type: "price", precision: 6, minMove: 1e-6 };
  return { type: "price", precision: 8, minMove: 1e-8 };
}

function applyLwSeriesPriceFormat(series, values) {
  if (!series || typeof series.applyOptions !== "function") return;
  try {
    series.applyOptions({ priceFormat: inferLwPriceFormatFromSamples(values) });
  } catch (_e) {
    /* oudere LW-build */
  }
}

/** Lightweight Charts: `Time` = unix seconden | BusinessDay-object | soms string. */
function lightweightChartsTimeToUnixSeconds(time) {
  if (time == null) return NaN;
  if (typeof time === "number" && Number.isFinite(time)) {
    /* UTCTimestamp in LW = unix seconden; sommige series geven per ongeluk ms door (>1e12). */
    if (time > 1e12) return time / 1000;
    return time;
  }
  if (typeof time === "string") {
    const n = Number(time);
    if (Number.isFinite(n) && n > 1e8) return n > 1e12 ? n / 1000 : n;
    const ms = Date.parse(time);
    if (Number.isFinite(ms)) return ms / 1000;
    return NaN;
  }
  if (typeof time === "object") {
    if (typeof time.timestamp === "number" && Number.isFinite(time.timestamp)) {
      const t = time.timestamp;
      return t > 1e12 ? t / 1000 : t;
    }
    const y = time.year;
    const m = time.month;
    const d = time.day;
    if (y != null && m != null && d != null) {
      return Date.UTC(Number(y), Number(m) - 1, Number(d), 12, 0, 0) / 1000;
    }
  }
  return NaN;
}

function formatChartAxisTimeHm(timeSecOrLwTime) {
  const sec = lightweightChartsTimeToUnixSeconds(timeSecOrLwTime);
  if (!Number.isFinite(sec)) return "";
  const dt = new Date(sec * 1000);
  if (!Number.isFinite(dt.getTime())) return "";
  return dt.toLocaleTimeString("nl-NL", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

let __lwPredXSyncRaf = null;

function scheduleSyncPredictionChartJsXFromLightweight() {
  if (__lwPredXSyncRaf != null) return;
  __lwPredXSyncRaf = requestAnimationFrame(() => {
    __lwPredXSyncRaf = null;
    syncPredictionChartJsXFromLightweight();
  });
}

/** Chart.js voorspellingspaneel: x-as = zichtbaar LW-time window (pan/zoom hoofdgrafiek). */
function syncPredictionChartJsXFromLightweight() {
  if (typeof activeTab !== "undefined" && activeTab !== "terminal") return;
  const chart =
    typeof initTradingPredictionChart === "function" ? initTradingPredictionChart() : window.tradingChart;
  if (!chart || !chart.options || !chart.options.scales || !chart.options.scales.x) return;
  if (!priceChart) return;
  let fromMs = NaN;
  let toMs = NaN;
  try {
    const ts = priceChart.timeScale();
    const tr = typeof ts.getVisibleRange === "function" ? ts.getVisibleRange() : null;
    if (tr && tr.from != null && tr.to != null) {
      const f = lightweightChartsTimeToUnixSeconds(tr.from);
      const t = lightweightChartsTimeToUnixSeconds(tr.to);
      if (Number.isFinite(f) && Number.isFinite(t) && t > f) {
        fromMs = f * 1000;
        toMs = t * 1000;
      }
    }
  } catch (_e) {
    /* ignore */
  }
  if (!Number.isFinite(fromMs) || !Number.isFinite(toMs)) {
    try {
      const lr = priceChart.timeScale().getVisibleLogicalRange?.();
      const arr = Array.isArray(rawChartLineData) ? rawChartLineData : [];
      if (!lr || !arr.length) return;
      const n = arr.length;
      const i0 = Math.max(0, Math.min(n - 1, Math.floor(Number(lr.from))));
      const i1 = Math.max(0, Math.min(n - 1, Math.ceil(Number(lr.to))));
      const tA = normalizeChartTimeSeconds(Number(arr[i0].time));
      const tB = normalizeChartTimeSeconds(Number(arr[i1].time));
      if (!Number.isFinite(tA) || !Number.isFinite(tB) || tB <= tA) return;
      fromMs = tA * 1000;
      toMs = tB * 1000;
    } catch (_e2) {
      return;
    }
  }
  if (!Number.isFinite(fromMs) || !Number.isFinite(toMs) || toMs <= fromMs) return;
  const pad = (toMs - fromMs) * 0.02;
  const xScale = chart.options.scales.x;
  xScale.type = "linear";
  xScale.min = fromMs - pad;
  xScale.max = toMs + pad;
  delete xScale.suggestedMin;
  delete xScale.suggestedMax;
  chart.update("none");
}

window.syncPredictionChartJsXFromLightweight = syncPredictionChartJsXFromLightweight;
window.scheduleSyncPredictionChartJsXFromLightweight = scheduleSyncPredictionChartJsXFromLightweight;

function predictionRowToEpochSeconds(row) {
  if (!row || typeof row !== "object") return null;
  const raw = row.timestamp ?? row.ts ?? row.time ?? row.t ?? row.generated_at;
  if (raw == null) return null;
  if (typeof raw === "number" && Number.isFinite(raw)) {
    const s = raw > 1e12 ? Math.floor(raw / 1000) : Math.floor(raw);
    return toEpochSeconds(new Date(s * 1000).toISOString());
  }
  return toEpochSeconds(String(raw));
}

function decisionMatchesMarket(decision, market) {
  if (!decision || typeof decision !== "object") return false;
  const norm = (s) => String(s || "").toUpperCase().replace("/", "-");
  const m = norm(market);
  if (!m) return false;
  const dk = norm(decision.market || decision.ticker || decision.symbol || "");
  return !!dk && dk === m;
}

function hpBotAppearsRunningPayload(data) {
  const bs = String((data && data.bot_status) || "").toLowerCase();
  if (!bs) return true;
  return bs !== "paused" && bs !== "panic_stop" && bs !== "stopped" && bs !== "stop";
}

function hpPolicyThreePercentsNearlyZero(b, h, s) {
  const toPct = (x) => {
    const n = Number(x);
    if (!Number.isFinite(n) || n < 0) return NaN;
    return n <= 1.0 + 1e-9 ? n * 100.0 : Math.min(100, n);
  };
  const pb = toPct(b);
  const ph = toPct(h);
  const ps = toPct(s);
  if (![pb, ph, ps].every((v) => Number.isFinite(v))) return true;
  return pb <= 0.05 && ph <= 0.05 && ps <= 0.05;
}

function hpPaintProbBarsCalculating(elB, elH, elS, barB, barH, barS) {
  if (elB) elB.textContent = "Calculating…";
  if (elH) elH.textContent = "Calculating…";
  if (elS) elS.textContent = "Calculating…";
  if (barB) barB.style.width = "0%";
  if (barH) barH.style.width = "0%";
  if (barS) barS.style.width = "0%";
}

function paintAiPolicyProbsFromPayload(data) {
  try {
  const elB = document.getElementById("hpAiProbBuy");
  const elH = document.getElementById("hpAiProbHold");
  const elS = document.getElementById("hpAiProbSell");
  const barB = document.getElementById("hpAiProbBuyBar");
  const barH = document.getElementById("hpAiProbHoldBar");
  const barS = document.getElementById("hpAiProbSellBar");
  if (!elB || !elH || !elS) return;
  /* Geen globale payloadSymbolMatchesActive-blokkade: branches filteren op actieve munt (ap/rl_last/dMulti). */
  const setRow = (valEl, barEl, raw) => {
    const p = normalizeProbToPercent(raw);
    if (valEl) valEl.textContent = p == null ? "—" : `${p.toFixed(1)}%`;
    if (barEl) barEl.style.width = p == null ? "0%" : `${Math.min(100, Math.max(0, p)).toFixed(2)}%`;
  };
  const normMk = (s) => String(s || "").toUpperCase().replace("/", "-");
  const activeMarket = getActiveMarketForPolicyUi();
  const activeU = normMk(activeMarket);

  const multi = data && data.rl_multi_decisions;
  let dMulti = null;
  if (multi && typeof multi === "object") {
    const raw = multi[activeMarket] || multi[activeU];
    if (raw && typeof raw === "object") dMulti = raw;
    if (!dMulti) {
      for (const k of Object.keys(multi)) {
        if (normMk(k) === activeU) {
          dMulti = multi[k];
          break;
        }
      }
    }
  }
  if (dMulti && typeof dMulti === "object") {
    const bM = normalizeProbToPercent(dMulti.prob_buy);
    const hM = normalizeProbToPercent(dMulti.prob_hold);
    const sM = normalizeProbToPercent(dMulti.prob_sell);
    const hasMulti = [bM, hM, sM].some((v) => v != null && v > 0.01);
    const taggedOk = decisionMatchesMarket(dMulti, activeU);
    const untagged = !String(dMulti.market || dMulti.ticker || dMulti.symbol || "").trim();
    /* Object komt uit rl_multi_decisions[actieve markt]: altijd tekenen bij geldige probs (ticker-tag kan achterlopen). */
    if (hasMulti) {
      if (hpPolicyThreePercentsNearlyZero(dMulti.prob_buy, dMulti.prob_hold, dMulti.prob_sell) && hpBotAppearsRunningPayload(data)) {
        hpPaintProbBarsCalculating(elB, elH, elS, barB, barH, barS);
        return;
      }
      setRow(elB, barB, dMulti.prob_buy);
      setRow(elH, barH, dMulti.prob_hold);
      setRow(elS, barS, dMulti.prob_sell);
      if (typeof window.applyHpAiProbThresholdMarkers === "function") {
        window.applyHpAiProbThresholdMarkers(
          data && data.rl_decision_threshold_pct != null ? data : window.__lastDashboardStats || data
        );
      }
      return;
    }
    if ((taggedOk || untagged) && hpBotAppearsRunningPayload(data)) {
      hpPaintProbBarsCalculating(elB, elH, elS, barB, barH, barS);
      return;
    }
  }

  const ap = data && data.ai_action_probs;
  const apUseful =
    ap &&
    typeof ap === "object" &&
    (ap.buy_pct != null || ap.hold_pct != null || ap.sell_pct != null);
  if (apUseful) {
    const apMarket = normMk(ap.market || ap.ticker || ap.symbol || data?.market || data?.ticker || "");
    if (apMarket && apMarket !== activeU) {
      return;
    }
    if (apMarket) {
      const b = normalizeProbToPercent(ap.buy_pct);
      const h = normalizeProbToPercent(ap.hold_pct);
      const s = normalizeProbToPercent(ap.sell_pct);
      const hasAny = [b, h, s].some((v) => v != null && v > 0.01);
      if (!hasAny) {
        if (hpBotAppearsRunningPayload(data)) hpPaintProbBarsCalculating(elB, elH, elS, barB, barH, barS);
        else {
          if (elB) elB.textContent = "Thinking...";
          if (elH) elH.textContent = "Thinking...";
          if (elS) elS.textContent = "Thinking...";
          if (barB) barB.style.width = "0%";
          if (barH) barH.style.width = "0%";
          if (barS) barS.style.width = "0%";
        }
        return;
      }
      setRow(elB, barB, ap.buy_pct);
      setRow(elH, barH, ap.hold_pct);
      setRow(elS, barS, ap.sell_pct);
      if (typeof window.applyHpAiProbThresholdMarkers === "function") {
        window.applyHpAiProbThresholdMarkers(
          data && data.rl_decision_threshold_pct != null ? data : window.__lastDashboardStats || data
        );
      }
      return;
    }
  }
  let d = data && data.rl_last_decision;
  if (!d && dMulti && typeof dMulti === "object") d = dMulti;
  if (d && typeof d === "object") {
    const dk = normMk(d.market || d.ticker || d.symbol || "");
    if (dk && dk !== activeU) {
      setRow(elB, barB, null);
      setRow(elH, barH, null);
      setRow(elS, barS, null);
      if (typeof window.applyHpAiProbThresholdMarkers === "function") {
        window.applyHpAiProbThresholdMarkers(window.__lastDashboardStats || data || {});
      }
      return;
    }
  }
  if (!d || typeof d !== "object") {
    setRow(elB, barB, null);
    setRow(elH, barH, null);
    setRow(elS, barS, null);
    if (typeof window.applyHpAiProbThresholdMarkers === "function") {
      window.applyHpAiProbThresholdMarkers(window.__lastDashboardStats || data || {});
    }
    return;
  }
  if (hpPolicyThreePercentsNearlyZero(d.prob_buy, d.prob_hold, d.prob_sell) && hpBotAppearsRunningPayload(data)) {
    hpPaintProbBarsCalculating(elB, elH, elS, barB, barH, barS);
    if (typeof window.applyHpAiProbThresholdMarkers === "function") {
      window.applyHpAiProbThresholdMarkers(data && data.rl_decision_threshold_pct != null ? data : window.__lastDashboardStats || data);
    }
    return;
  }
  setRow(elB, barB, d.prob_buy);
  setRow(elH, barH, d.prob_hold);
  setRow(elS, barS, d.prob_sell);
  if (typeof window.applyHpAiProbThresholdMarkers === "function") {
    window.applyHpAiProbThresholdMarkers(
      data && data.rl_decision_threshold_pct != null ? data : window.__lastDashboardStats || data
    );
  }
  } finally {
    if (typeof window.syncHpAiBrainConfidenceGlow === "function") {
      try {
        window.syncHpAiBrainConfidenceGlow(data);
      } catch (_e) {
        /* noop */
      }
    }
  }
}

window.paintAiPolicyProbsFromPayload = paintAiPolicyProbsFromPayload;

function formatBrainReasoningBodyHtml(raw) {
  let t = escapeHtmlText(String(raw || ""));
  t = t.replace(/(Besluit:\s*(?:HOLD|BUY|SELL)\.)/gi, '<span class="cockpit-besluit-callout">$1</span>');
  t = t.replace(/\n/g, "<br>");
  return t;
}

/** Badge + label: Scanner vs RL policy vs wachten (zelfde logica als module_brain.js). */
function brainReasoningSourceMeta(metaCtx, bodyLower) {
  const st = String((metaCtx && metaCtx.status) || "").toLowerCase();
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
  if (t.includes("geen rl-policytekst nog") || t.includes("model aan het inladen") || t.includes("wachten op eerste besluit")) {
    return { badge: "Wacht", variant: "wait", note: "Nog geen volledige policy-tekst" };
  }
  return { badge: "Context", variant: "mixed", note: "Payload of mix" };
}

function formatBrainReasoningPanelHtml(raw, metaCtx) {
  const body = formatBrainReasoningBodyHtml(raw);
  const meta = brainReasoningSourceMeta(metaCtx || {}, String(raw || "").toLowerCase());
  return (
    `<div class="brain-reasoning-stack">` +
    `<header class="brain-reasoning-kicker" aria-label="Bron van de tekst">` +
    `<span class="brain-reasoning-badge brain-reasoning-badge--${meta.variant}">${escapeHtmlText(meta.badge)}</span>` +
    `<span class="brain-reasoning-kicker-note">${escapeHtmlText(meta.note)}</span>` +
    `</header>` +
    `<div class="brain-reasoning-body">${body}</div>` +
    `</div>`
  );
}

function classifyLogLine(line) {
  const text = String(line || "").toUpperCase();
  if (text.includes("[WHALE-SYNC]")) return "log-whale-sync";
  if (text.includes("[RL-BRAIN]")) return "log-rl-brain";
  if (text.includes("ERROR")) return "log-error";
  if (text.includes("WARNING") || text.includes("WARN")) return "log-warning";
  if (text.includes("SUCCESS")) return "log-success";
  return "log-info";
}

function formatSystemLogLineForDisplay(line) {
  // Behoud exact de logger-regel; geen extra timestamp-transformaties op de client.
  let s = String(line || "");
  // Cleanup: soms komt "[HH:MM:SS] HH:MM:SS ..." binnen; bewaar dan één tijd.
  s = s.replace(/^\[(\d{2}:\d{2}:\d{2})\]\s+\1\b/, "[$1]");
  s = s.replace(/^(\d{2}:\d{2}:\d{2})\s+\1\b/, "$1");
  return s;
}

function shouldAutoScroll(consoleEl) {
  const threshold = 40;
  return consoleEl.scrollHeight - (consoleEl.scrollTop + consoleEl.clientHeight) < threshold;
}

/** Dedupe-key: zelfde logpayload zonder leidende timestamp / [LEVEL]-prefixen. */
function cockpitLogMessageKey(line) {
  let s = String(line || "");
  // Strip alle datum/tijd-fragmenten zodat anti-spam op inhoud dedupet.
  s = s.replace(/\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b/gi, "");
  s = s.replace(/\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\b/g, "");
  while (/^\[[^\]]+\]\s+/.test(s)) s = s.replace(/^\[[^\]]+\]\s+/, "");
  return s.replace(/\s+/g, " ").trim();
}

if (typeof window.__terminalLogMessageKey !== "function") {
  window.__terminalLogMessageKey = cockpitLogMessageKey;
}

function cockpitLogSplitTsRest(text) {
  const s = String(text);
  let m = s.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+(.*)$/i);
  if (m) return { ts: m[1], rest: m[2] };
  m = s.match(/^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(.*)$/);
  if (m) return { ts: m[1], rest: m[2] };
  return { ts: "", rest: s };
}

function cockpitLogTailKind(line) {
  const s = String(line || "");
  const u = s.toUpperCase();
  if (u.includes("EXCEPTION") || u.includes("TRACEBACK") || u.includes("CRITICAL") || /\bERROR\b/.test(s)) return "error";
  if (/\bINFO\b/.test(s)) return "info";
  return "system";
}

function flashCockpitTailRow(el) {
  el.classList.remove("tail-row--flash");
  void el.offsetWidth;
  el.classList.add("tail-row--flash");
  window.setTimeout(() => {
    try {
      el.classList.remove("tail-row--flash");
    } catch (_) {}
  }, 480);
}

function pushLogsTabLineFallback(line) {
  const root = document.getElementById("cockpitLogConsole");
  if (!root || systemLogsMuted) return;
  const displayText = formatSystemLogLineForDisplay(line);
  const key = cockpitLogMessageKey(displayText);
  const last = root.lastElementChild;
  if (last && key.length > 0 && last.dataset.dedupeKey === key) {
    const n = Number(last.dataset.dedupeCount || "1") + 1;
    last.dataset.dedupeCount = String(n);
    let badge = last.querySelector(".tail-dup-badge");
    if (!badge) {
      badge = document.createElement("span");
      badge.className = "tail-dup-badge";
      badge.setAttribute("aria-label", `${n} keer herhaald`);
      const tsEl = last.querySelector(".tail-ts");
      if (tsEl) tsEl.insertAdjacentElement("afterend", badge);
      else last.insertBefore(badge, last.firstChild);
    }
    badge.textContent = `[x${n}]`;
    if (cockpitLogTailKind(displayText) === "error") {
      last.className = "tail-row tail-row--error";
    }
    const prevTop = root.scrollTop;
    flashCockpitTailRow(last);
    root.scrollTop = prevTop;
    return;
  }

  const kind = cockpitLogTailKind(displayText);
  const div = document.createElement("div");
  div.className = `tail-row tail-row--${kind}`;
  div.dataset.dedupeKey = key;
  div.dataset.dedupeCount = "1";
  const { ts, rest } = cockpitLogSplitTsRest(displayText);
  if (ts) {
    const tsSpan = document.createElement("span");
    tsSpan.className = "tail-ts";
    tsSpan.textContent = `${ts} `;
    div.appendChild(tsSpan);
  }
  const msg = document.createElement("span");
  msg.className = "tail-msg";
  msg.textContent = ts ? rest : displayText;
  div.appendChild(msg);
  root.appendChild(div);
  while (root.children.length > 100) root.removeChild(root.firstChild);
  root.scrollTop = root.scrollHeight;
}

function appendSystemLogLine(line) {
  if (systemLogsMuted) return;
  const raw = String(line || "");
  const upper = raw.toUpperCase();
  // Actieve ruisfilter: onderdruk observatie-spam op INFO-niveau.
  if (raw.includes("_build_observation completed") && !(/\bERROR\b|\bCRITICAL\b|\bWARN(?:ING)?\b/i.test(raw))) {
    return;
  }
  if (activeTab === "logs") {
    if (window.TerminalLiveTail && typeof window.TerminalLiveTail.push === "function") {
      window.TerminalLiveTail.push(raw);
      return;
    }
    pushLogsTabLineFallback(raw);
    return;
  }
  let consoleEl = document.getElementById("systemLogConsole");
  
  // Auto-inject container als deze in de HTML dev-tab ontbreekt
  if (!consoleEl) {
      const hardwareTab = document.getElementById("tab-hardware");
      if (!hardwareTab) return;
      consoleEl = document.createElement("div");
      consoleEl.id = "systemLogConsole";
      consoleEl.style.flex = "1";
      consoleEl.style.overflowY = "auto";
      consoleEl.style.backgroundColor = "#0b0e11";
      consoleEl.style.border = "2px solid #333";
      consoleEl.style.borderRadius = "8px";
      consoleEl.style.padding = "10px";
      consoleEl.style.marginTop = "20px";
      consoleEl.style.minHeight = "400px";
      consoleEl.style.maxHeight = "600px";
      hardwareTab.appendChild(consoleEl);
  }

  const autoscroll = shouldAutoScroll(consoleEl);
  const displayText = formatSystemLogLineForDisplay(raw);
  const dedupeKey = cockpitLogMessageKey(displayText);
  const last = consoleEl.lastElementChild;
  if (last && dedupeKey.length > 0 && last.dataset && last.dataset.dedupeKey === dedupeKey) {
    const n = Number(last.dataset.dedupeCount || "1") + 1;
    last.dataset.dedupeCount = String(n);
    let badge = last.querySelector(".tail-dup-badge");
    if (!badge) {
      badge = document.createElement("span");
      badge.className = "tail-dup-badge";
      badge.style.marginLeft = "8px";
      last.appendChild(badge);
    }
    badge.textContent = `[x${n}]`;
    if (autoscroll) consoleEl.scrollTop = consoleEl.scrollHeight;
    return;
  }
  const div = document.createElement("div");
  div.className = `system-log-line ${classifyLogLine(line)}`;
  div.dataset.dedupeKey = dedupeKey;
  div.dataset.dedupeCount = "1";
  
  // Fallback styling voor het geval dev-tab CSS mist
  div.style.fontFamily = "'JetBrains Mono', ui-monospace, monospace";
  div.style.fontSize = "13px";
  div.style.padding = "2px 4px";
  div.style.borderBottom = "1px solid rgba(255,255,255,0.05)";
  div.style.wordBreak = "break-all";
  
  if (upper.includes("ERROR") || upper.includes("CRITICAL")) div.style.color = "#ff3131";
  else if (upper.includes("WARN")) div.style.color = "#ffb84d";
  else if (upper.includes("SUCCESS") || upper.includes("OK")) div.style.color = "#39ff14";
  else if (upper.includes("[RL-BRAIN]")) div.style.color = "#00f8ff";
  else div.style.color = "#cccccc";

  div.textContent = displayText;
  consoleEl.appendChild(div);
  while (consoleEl.children.length > MAX_SYSTEM_LOG_LINES) {
    consoleEl.removeChild(consoleEl.firstChild);
  }
  if (autoscroll) {
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }
}

function clearSystemLogConsole() {
  const consoleEl = document.getElementById("systemLogConsole");
  if (consoleEl) consoleEl.innerHTML = "";
  systemLogsBuffer = [];
  if (window.TerminalLiveTail && typeof window.TerminalLiveTail.clear === "function") {
    window.TerminalLiveTail.clear();
  } else {
    const cockpit = document.getElementById("cockpitLogConsole");
    if (cockpit) cockpit.innerHTML = "";
  }
}

function updateSystemPauseButton() {
  const btn = document.getElementById("systemLogPauseBtn");
  if (!btn) return;
  btn.textContent = systemLogsPaused ? "Resume" : "Pause";
  btn.classList.toggle("active", systemLogsPaused);
}

function updateSystemMuteButton() {
  const btn = document.getElementById("systemLogMuteBtn");
  if (!btn) return;
  btn.textContent = systemLogsMuted ? "Unmute" : "Mute";
  btn.classList.toggle("active", systemLogsMuted);
}

function toggleSystemLogPause() {
  systemLogsPaused = !systemLogsPaused;
  updateSystemPauseButton();
  if (!systemLogsPaused && systemLogsBuffer.length) {
    const flush = systemLogsBuffer.slice(-MAX_SYSTEM_LOG_LINES);
    systemLogsBuffer = [];
    for (const line of flush) {
      appendSystemLogLine(line);
    }
  }
}

function toggleSystemLogMute() {
  systemLogsMuted = !systemLogsMuted;
  updateSystemMuteButton();
}

function coinBadgeClass(coin) {
  const c = String(coin || "").toLowerCase();
  if (c === "btc") return "tag-btc";
  if (c === "eth") return "tag-eth";
  if (c === "sol") return "tag-sol";
  if (c === "xrp") return "tag-xrp";
  if (c === "ada") return "tag-ada";
  return "tag-default";
}

function sourceBadge(icon, source) {
  return "";
}

function toShortSummary(raw) {
  const text = String(raw || "")
    .replace(/<[^>]*>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!text) return "Geen samenvatting beschikbaar.";
  const maxChars = 420;
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars).trimEnd()}...`;
}

function openNewsModal(item) {
  document.getElementById("newsModalTitle").textContent = item.title || item.headline || "-";
  document.getElementById("newsModalMeta").textContent =
    `${item.source || "Unknown source"} | ${formatNewsTimeAmsterdam(item.published_at || item.ts)}`;
  document.getElementById("newsModalSummary").textContent = toShortSummary(item.summary || item.description || "");
  const tickers = Array.isArray(item.affected_tickers) ? item.affected_tickers : [item.ticker_tag || item.coin].filter(Boolean);
  document.getElementById("newsModalTickers").textContent = `Affected tickers: ${tickers.join(", ") || "-"}`;
  document.getElementById("newsModalAi").textContent =
    `${item.ai_reasoning || item.explanation || "No AI reasoning available."}`;
  const link = document.getElementById("newsModalLink");
  link.href = item.url || "#";
  link.style.pointerEvents = item.url ? "auto" : "none";
  document.getElementById("newsModal").classList.remove("hidden");
}

function closeNewsModal() {
  document.getElementById("newsModal").classList.add("hidden");
}

function noteLivePriceFromWs(rawPrice, marketHint) {
  const n = Number(rawPrice);
  if (!Number.isFinite(n)) return;
  const mkt = String(marketHint || selectedMarket || "BTC-EUR").toUpperCase();
  const critical = window.__eliteCriticalStream === true;
  const nowMs = performance.now();
  window.__lastPricePushByMkt = window.__lastPricePushByMkt || Object.create(null);
  if (!critical) {
    const last = window.__lastPricePushByMkt[mkt];
    if (last != null && nowMs - last < 500) return;
  }
  window.__lastPricePushByMkt[mkt] = nowMs;
  if (!window.botMetrics.wsPrices || typeof window.botMetrics.wsPrices !== "object") {
    window.botMetrics.wsPrices = {};
  }
  window.botMetrics.wsPrices[mkt] = n;
  const sel = String(selectedMarket || "BTC-EUR").toUpperCase();
  if (mkt !== sel) return;
  lastBufferedPrice = { n, mkt };
  // Senior Fix: Haper-beveiliging (Throttle). We renderen niet direct op inkomend verkeer.
  // De 1Hz heartbeat (startCockpitHeartbeat) pakt lastBufferedPrice automatisch op!
}

/** WebSocket UI: batch DOM updates in één animation frame (minder flicker). */
let cockpitWsRafPending = false;
const cockpitWsRafQueue = [];

function enqueueCockpitWsRender(fn) {
  if (typeof fn !== "function") return;
  cockpitWsRafQueue.push(fn);
  if (cockpitWsRafPending) return;
  cockpitWsRafPending = true;
  requestAnimationFrame(() => {
    cockpitWsRafPending = false;
    const batch = cockpitWsRafQueue.splice(0, cockpitWsRafQueue.length);
    for (const f of batch) {
      try {
        f();
      } catch (_) {}
    }
  });
}

function normalizeSystemStatsWsPayload(raw) {
  let d = raw;
  if (typeof raw === "string") {
    try {
      d = JSON.parse(raw);
    } catch (_) {
      return null;
    }
  }
  if (!d) return null;
  if (d.t === "hb") return { __ws: "hb" };
  if (d.t === "system_stats" && d.c != null && d.cpu_pct == null) {
    return {
      topic: "system_stats",
      cpu_pct: d.c,
      ram_pct: d.r,
      disk_pct: d.d,
      gpu_util_pct: d.g,
      gpu_mem_util_pct: d.gm,
      gpu_util_effective: d.ge,
      vram_used_mb: d.vu,
      vram_total_mb: d.vt,
      gpu_ok: d.gk === 1,
      gpu_name: d.gn,
      gpu_index: d.gi,
    };
  }
  return d;
}

function normalizeBrainWsPayload(raw) {
  let d = raw;
  if (typeof raw === "string") {
    try {
      d = JSON.parse(raw);
    } catch (_) {
      return null;
    }
  }
  if (!d) return null;
  if (d.t === "hb") return { __ws: "hb" };
  if (d.v === 1 && d.t === "brain_stats") {
    if (Array.isArray(d.L) && window.botMetrics) {
      window.botMetrics.eliteLiteStream = d.L;
      window.botMetrics.focusMarket = d.f;
    }
    return {
      topic: "brain_stats",
      training_monitor: d.tm,
      feature_weights: d.fw,
      feature_weights_policy: d.fwp,
      rl_observation: d.rl,
      social_buzz: d.sb,
      reasoning: d.reasoning,
      generated_at: d.generated_at,
    };
  }
  return d;
}

/** Prijs uit Redis/worker snapshot (/activity + WS); voorkomt vastlopende header bij market-switch. Gebruikt `active_markets` als wallet-map per pair incompleet of corrupt is. */
function applySelectedMarketPriceFromPaperPortfolio(p, data) {
  if (!p || typeof p !== "object") p = {};
  const mk = String(selectedMarket || "BTC-EUR").toUpperCase();
  const lpm = p.last_prices_by_market;
  let raw =
    lpm && typeof lpm === "object" && lpm[mk] != null && lpm[mk] !== "" ? lpm[mk] : null;
  const numOk = (x) => {
    const v = Number(x);
    return Number.isFinite(v) && v > 0 ? v : null;
  };
  let n = numOk(raw);
  if (n == null && data && Array.isArray(data.active_markets)) {
    for (const row of data.active_markets) {
      if (String((row && row.market) || "").toUpperCase() !== mk) continue;
      const lp = row.last_price != null ? row.last_price : row.price;
      n = numOk(lp);
      if (n != null) break;
    }
  }
  if (n == null) return;
  lastBufferedPrice = { n, mkt: mk };
  window.__lastPricePushByMkt = window.__lastPricePushByMkt || Object.create(null);
  window.__lastPricePushByMkt[mk] = performance.now();
  flushHeaderPriceTick();
}

function flushHeaderPriceTick() {
  const lp = document.getElementById("btc-price");
  if (!lp || lastBufferedPrice === null) return;
  const { n, mkt } = lastBufferedPrice;
  const prev = priceAtLastSecondTick;
  lp.textContent = `${mkt}  ${n.toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 6 })}`;
  if (prev != null && Number.isFinite(prev)) {
    if (n > prev + 1e-12) {
      lp.classList.remove("cockpit-btc-price--down");
      lp.classList.add("cockpit-btc-price--up");
      window.setTimeout(() => {
        try {
          lp.classList.remove("cockpit-btc-price--up");
        } catch (_) {}
      }, 420);
    } else if (n < prev - 1e-12) {
      lp.classList.remove("cockpit-btc-price--up");
      lp.classList.add("cockpit-btc-price--down");
      window.setTimeout(() => {
        try {
          lp.classList.remove("cockpit-btc-price--down");
        } catch (_) {}
      }, 420);
    }
  }
  priceAtLastSecondTick = n;
  if (typeof window.tickTerminalPredictionChartLivePrice === "function") {
    window.tickTerminalPredictionChartLivePrice(n);
  }
}

function updateLastScanLabel() {
  const el = document.getElementById("brainLastScanLine");
  const db = document.getElementById("tab-terminal"); // of je root dashboard container
  if (!el) return;
  const iso = window.__lastEngineTickIso;
  if (!iso) {
    el.textContent = "Laatste scan: wacht op engine-tick…";
    el.classList.remove("text-red-500", "animate-pulse");
    return;
  }
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t)) {
    el.textContent = "Laatste scan: —";
    return;
  }
  const sec = Math.max(0, Math.floor((Date.now() - t) / 1000));
  
  if (sec > 30) {
    // Stale Data UI Waarschuwing
    el.textContent = `⚠️ WAARSCHUWING: Data is verouderd (${sec}s geleden)`;
    el.classList.add("text-red-500", "font-bold", "animate-pulse");
    el.style.color = "#ff3131"; 
    if (db) {
      db.classList.add("opacity-75", "grayscale-[30%]");
      db.style.transition = "all 0.5s ease";
    }
  } else {
    // Gezonde UI
    el.textContent = sec === 0 ? "Laatste scan: zojuist" : `Laatste scan: ${sec} seconden geleden`;
    el.classList.remove("text-red-500", "font-bold", "animate-pulse");
    el.style.color = "";
    if (db) {
      db.classList.remove("opacity-75", "grayscale-[30%]");
    }
  }
}

function startCockpitHeartbeat() {
  if (cockpitHeartbeatTimer) return;
  cockpitHeartbeatTimer = setInterval(() => {
    flushHeaderPriceTick();
    updateLastScanLabel();
  }, 1000);
}

function connectBitvavoPriceStream() {
  if (wsRef && [WebSocket.OPEN, WebSocket.CONNECTING].includes(wsRef.readyState)) return;
  priceAtLastSecondTick = null;
  lastBufferedPrice = null;
  const mkt = String(selectedMarket || "BTC-EUR").toUpperCase();
  /** Alleen geselecteerde markt + Elite-8 snapshot — niet alle dropdown-opties (veel minder WS-traffic). */
  const eliteMk = (Array.isArray(lastActiveMarketsRows) ? lastActiveMarketsRows : [])
    .slice(0, 8)
    .map((row) => String(row.market || "").toUpperCase())
    .filter(Boolean);
  const subscribedMarkets = Array.from(new Set([mkt, ...eliteMk]));
  const ws = new WebSocket("wss://ws.bitvavo.com/v2/");
  wsRef = ws;
  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        action: "subscribe",
        channels: [
          { name: "ticker24h", markets: subscribedMarkets },
          { name: "trades", markets: subscribedMarkets },
        ],
      })
    );
  };
  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload?.event === "ticker24h" && Array.isArray(payload.data) && payload.data.length) {
        for (const row of payload.data) {
          const pm = String(row.market || mkt).toUpperCase();
          noteLivePriceFromWs(row.last ?? row.close ?? row.closePrice, pm);
        }
      }
      if (payload?.event === "trade" && Array.isArray(payload.data)) {
        for (const t of payload.data) {
          const pm = String(t.market || mkt).toUpperCase();
          noteLivePriceFromWs(t.price, pm);
        }
      } else if (payload?.event === "trade") {
        const pm = String(payload.market || "").toUpperCase();
        noteLivePriceFromWs(payload.price, pm || mkt);
      }
    } catch (_) {}
  };
  ws.onclose = () => setTimeout(connectBitvavoPriceStream, 2000);
}

async function refreshHealthMode() {
  try {
    const res = await fetch("/health");
    if (!res.ok) return;
    const data = await res.json();
    renderModeBanner(data.mode);
  } catch (err) {
    console.warn("[health] Fetch failed:", err);
  }
}

function syncHeaderMarketChip() {
  const chip = document.getElementById("headerPairDisplay");
  if (chip) chip.textContent = String(selectedMarket || selectedChartPair || "BTC-EUR").toUpperCase();
  syncHeaderPositionLamp();
}

/** Positie-indicator in cockpit-header (open positie op geselecteerde markt). */
function syncHeaderPositionLamp() {
  const lamp = document.getElementById("headerMarketPosLamp");
  if (!lamp) return;
  const mk = String(
    document.getElementById("marketSelect")?.value || selectedMarket || selectedChartPair || "BTC-EUR"
  ).toUpperCase();
  const held = window.__marketsInPosition instanceof Set ? window.__marketsInPosition : new Set();
  const on = held.has(mk);
  lamp.classList.toggle("cockpit-pos-lamp--on", on);
  lamp.classList.toggle("cockpit-pos-lamp--off", !on);
  const msgOn = `Open paper-positie op ${mk}`;
  const msgOff = `Geen open positie op ${mk}`;
  lamp.setAttribute("aria-label", on ? msgOn : msgOff);
  lamp.title = on ? msgOn : msgOff;
}

window.buildMarketsInPositionSet = buildMarketsInPositionSet;
window.syncHeaderPositionLamp = syncHeaderPositionLamp;

async function refreshSelectedMarket() {
  try {
    const res = await fetch("/markets/selected");
    if (!res.ok) return;
    const data = await res.json();
    selectedMarket = data.selected_market || "BTC-EUR";
    syncHeaderMarketChip();
    syncElite8AssetToolbarSelection();
  } catch (err) {
    console.warn("[markets/selected] Fetch failed:", err);
  }
}

async function refreshMarkets() {
  const select = document.getElementById("marketSelect");
  if (!select) return;
  try {
    const res = await fetch("/markets/active");
    const text = await res.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch (_parseErr) {
      cockpitLedgerStatusParts.markets = `Markten: ongeldige response (${res.status})`;
      paintCockpitLedgerStatus();
      return;
    }
    if (!res.ok) {
      const det = typeof data.detail === "string" ? data.detail.slice(0, 72) : "";
      cockpitLedgerStatusParts.markets = det
        ? `Markten: fout (${res.status}) — ${det}`
        : `Markten: fout (${res.status})`;
      paintCockpitLedgerStatus();
      select.innerHTML = "";
      lastActiveMarketsRows = [];
      renderScannerTickerBar([]);
      return;
    }
    select.innerHTML = "";
    const rows = Array.isArray(data.markets) ? [...data.markets] : [];
    lastActiveMarketsRows = rows;
    if (!rows.length) {
      cockpitLedgerStatusParts.markets = "Markten: leeg (scanner/Bitvavo)";
    } else {
      const n = rows.length;
      cockpitLedgerStatusParts.markets = `Markten: ${n} geladen`;
    }
    paintCockpitLedgerStatus();
    const bitvavoVolTop =
      rows.some((r) => r && r.list_profile === "bitvavo_eur_volume_top") ||
      rows.some((r) => String(r && r.selection_reason).includes("Bitvavo EUR #"));
    if (bitvavoVolTop) {
      rows.sort((a, b) => Number(b.volume_quote_24h || 0) - Number(a.volume_quote_24h || 0));
    } else {
      rows.sort((a, b) => {
        const pa = a.is_pillar === true ? 0 : 1;
        const pb = b.is_pillar === true ? 0 : 1;
        if (pa !== pb) return pa - pb;
        return String(a.market || "").localeCompare(String(b.market || ""));
      });
    }
    for (const m of rows) {
      const opt = document.createElement("option");
      opt.value = m.market;
      const star = m.is_pillar === true ? "★ " : "";
      const q = Number(m.quality_score || 0);
      opt.textContent = `${star}${m.market}`;
      const hintParts = [];
      if (m.selection_reason) hintParts.push(String(m.selection_reason));
      if (m.volume_quote_24h != null) hintParts.push(`Vol24h: ${m.volume_quote_24h}`);
      hintParts.push(`Quality: ${q}/3`);
      opt.title = hintParts.join(" · ");
      if (m.market === selectedMarket) opt.selected = true;
      select.appendChild(opt);
    }
    void refreshActivity();
    renderScannerTickerBar(rows);
    syncHeaderMarketChip();
    syncElite8AssetToolbarSelection();
    if (wsRef) connectBitvavoPriceStream();
  } catch (err) {
    console.warn("[markets/active]", err);
    cockpitLedgerStatusParts.markets = "Markten: netwerkfout";
    paintCockpitLedgerStatus();
  }
}

function normalizeElite8PairKey(x) {
  return String(x || "")
    .toUpperCase()
    .replace("/", "-");
}

function syncScannerTickerBarSelection() {
  const root = document.getElementById("scannerTickerBar");
  if (!root) return;
  const sm = normalizeElite8PairKey(
    document.getElementById("marketSelect")?.value || selectedMarket || selectedChartPair || "BTC-EUR"
  );
  root.querySelectorAll("button.scanner-ticker-badge[data-market]").forEach((btn) => {
    const mk = normalizeElite8PairKey(btn.getAttribute("data-market"));
    btn.classList.toggle("active-asset", mk === sm);
  });
}

function syncElite8AssetToolbarSelection() {
  const root = document.getElementById("elite8AiStatusBar");
  if (!root) return;
  const sm = normalizeElite8PairKey(
    document.getElementById("marketSelect")?.value || selectedMarket || selectedChartPair || "BTC-EUR"
  );
  root.querySelectorAll("button.elite8-ai-pill[data-market]").forEach((btn) => {
    const mk = normalizeElite8PairKey(btn.getAttribute("data-market"));
    const on = mk === sm;
    btn.classList.toggle("active-asset", on);
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  });
  syncScannerTickerBarSelection();
}

window.syncElite8AssetToolbarSelection = syncElite8AssetToolbarSelection;

function refreshDashboardStatsForActiveMarket() {
  const mk = encodeURIComponent(
    String(
      document.getElementById("marketSelect")?.value ||
        selectedMarket ||
        selectedChartPair ||
        (window.AppCore && window.AppCore.state && window.AppCore.state.selectedMarket) ||
        "BTC-EUR"
    )
      .trim()
      .toUpperCase()
  );
  fetch(`/api/v1/stats?symbol=${mk}`)
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      if (d && typeof d === "object") {
        document.dispatchEvent(new CustomEvent("ai-trading-stats", { detail: d }));
      }
    })
    .catch(() => {});
}

window.refreshDashboardStatsForActiveMarket = refreshDashboardStatsForActiveMarket;

async function switchEliteMarket(market, marketsSnapshot) {
  const m = String(market || "").toUpperCase();
  if (!m) return;
  selectedMarket = m;
  selectedChartPair = m;
  lastBufferedPrice = null;
  priceAtLastSecondTick = null;
  const sel = document.getElementById("marketSelect");
  if (sel) {
    const match = Array.from(sel.options || []).find((o) => normalizeElite8PairKey(o.value) === normalizeElite8PairKey(m));
    if (match) sel.value = match.value;
  }
  syncElite8AssetToolbarSelection();
  const hdr = document.getElementById("headerPairDisplay");
  if (hdr) hdr.textContent = m;
  if (window.AppCore && window.AppCore.state) window.AppCore.state.selectedMarket = m;
  resetMarketScopedUi(m);
  void updateChart(m);
  if (typeof window.__reloadTerminalPredictionChart === "function") window.__reloadTerminalPredictionChart();
  const lpHdr = document.getElementById("btc-price");
  if (lpHdr) lpHdr.textContent = `${m}  …`;

  // Senior Fix: Zorg dat de frontend state daadwerkelijk de active market syncs heeft 
  if (window.botMetrics) {
    window.botMetrics.focusMarket = m;
    window.botMetrics.selectedMarket = m;
  }

  try {
    await fetch(`/markets/select?market=${encodeURIComponent(m)}`, { method: "POST" });
  } catch (_e) {}
  syncHeaderMarketChip();
  connectBitvavoPriceStream();
  refreshDashboardStatsForActiveMarket();
  await Promise.all([refreshBalanceCheck(), refreshBrainLab(), refreshNewsInsights()]);
  await refreshActivity();
  await refreshSentiment();
  const snap = marketsSnapshot && marketsSnapshot.length ? marketsSnapshot : lastActiveMarketsRows;
  if (snap && snap.length) renderScannerTickerBar(snap);
  syncElite8AssetToolbarSelection();
}

// Senior Fix: Expose de functie globaal als updateMarket voor universele DOM controls.
window.updateMarket = switchEliteMarket;

function renderElite8AiStatusBar(signals) {
  const root = document.getElementById("elite8AiStatusBar");
  if (!root) return;

  const list = Array.isArray(signals) ? signals : [];
  const sigHash = JSON.stringify(list);
  if (root.dataset.elite8SignalsHash !== sigHash) {
    root.dataset.elite8SignalsHash = sigHash;
    root.dataset.lastHash = sigHash;

    root.innerHTML = "";

    if (!list.length) {
      root.innerHTML = `<span class="elite8-ai-status-bar__empty">Elite-8 AI-status wordt geladen…</span>`;
    } else {
    for (const s of list) {
      const mk = String(s.market || "").toUpperCase();
      const base = String(s.base || (mk.includes("-") ? mk.split("-")[0] : mk) || "?").toUpperCase();
      const st = String(s.state || "neutral");
      const action = String(s.action || "").toUpperCase() || "…";
      const conf = Number(s.confidence || 0);
      const inPos = Boolean(s.in_position);
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = `elite8-ai-pill elite8-ai-pill--${st}${inPos ? " elite8-ai-pill--in-position" : ""}`;
      pill.setAttribute("data-market", mk);
      pill.style.setProperty("--coin-accent", baseAccentForMarket(mk));
      const dotClass = st === "panic" ? "panic" : st === "bear" ? "bear" : st === "bull" ? "bull" : "neutral";
      pill.innerHTML =
        `<span class="elite8-ai-pill__dot elite8-ai-pill__dot--${dotClass}" aria-hidden="true"></span>` +
        `<span class="elite8-ai-pill__sym">${escapeHtmlText(base)}</span>` +
        (inPos ? `<span class="elite8-ai-pill__pos" aria-label="In positie">POS</span>` : "");
      const extra =
        (inPos ? " | IN POSITIE (paper)" : "") +
        (s.whale_danger ? " | Whale danger zone" : "") +
        (s.panic_cooldown ? " | Panic cooldown (geen BUY)" : "");
      pill.title = `${mk} — AI ${action} (${conf.toFixed(2)})${extra}. Klik: volledige portal-switch.`;
      pill.addEventListener("click", async () => {
        await switchEliteMarket(mk, lastActiveMarketsRows);
      });
      root.appendChild(pill);
    }
    }
  }
  syncElite8AssetToolbarSelection();
}

function renderScannerTickerBar(markets) {
  const root = document.getElementById("scannerTickerBar");
  if (!root) return;

  const rows = Array.isArray(markets) ? markets.slice(0, 8) : [];
  const rowHash = JSON.stringify(rows);
  if (root.dataset.scannerRowsHash !== rowHash) {
    root.dataset.scannerRowsHash = rowHash;
    root.dataset.lastHash = rowHash;

    root.innerHTML = "";

    const held = window.__marketsInPosition instanceof Set ? window.__marketsInPosition : new Set();
    for (const row of rows) {
      const market = String(row.market || "-");
      const mku = market.toUpperCase();
      const inPos = held.has(mku);
      const pct = Number(row.price_change_pct_24h || 0);
      const el = document.createElement("button");
      el.type = "button";
      el.className = `scanner-ticker-badge ${pct >= 0 ? "is-up" : "is-down"}${inPos ? " scanner-ticker-badge--in-position" : ""}`;
      el.setAttribute("data-market", mku);
      el.style.setProperty("--coin-accent", baseAccentForMarket(mku));
      const star = row.is_pillar === true ? "* " : "";
      el.textContent = `${star}${market} ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%${inPos ? " ·POS" : ""}`;
      const baseTitle = row.selection_reason ? String(row.selection_reason) : "";
      el.title = (baseTitle ? baseTitle + " — " : "") + (inPos ? "IN POSITIE (paper wallet)" : "Geen paper-positie");
      el.addEventListener("click", async () => {
        await switchEliteMarket(market, markets);
      });
      root.appendChild(el);
    }
  }
  syncScannerTickerBarSelection();
}

async function selectMarketFromDropdown() {
  const market = document.getElementById("marketSelect").value;
  await switchEliteMarket(market, lastActiveMarketsRows);
  void updatePredictionChart(market);
}

async function refreshBalanceCheck() {
  try {
    const res = await fetch(`/vault/balance-check?market=${encodeURIComponent(selectedMarket)}`);
    const data = await res.json();
    if (!res.ok || data.available === false) {
      setClassHTML(".js-balance-check", `Vault status: <span class="status-disconnected">Disconnected</span> (${data.reason || "unknown"})`);
      window.botMetrics.vaultLine = "disconnected";
      setLiveUpdatedAt();
      return;
    }
    setClassHTML(".js-balance-check", `Vault status: <span class="status-connected">Connected</span> | (${selectedMarket}) Buy: ${data.sufficient_for_buy} | Sell: ${data.sufficient_for_sell} | CanTrade: ${data.can_trade}`);
    window.botMetrics.vaultLine = "connected";
    setLiveUpdatedAt();
  } catch (err) {
    setClassHTML(".js-balance-check", `Vault status: <span class="status-disconnected">Disconnected</span> (error: ${String(err)})`);
    window.botMetrics.vaultLine = "error";
    setLiveUpdatedAt();
  }
}

async function refreshSentiment() {
  try {
    const res = await fetch("/sentiment/current");
    if (!res.ok) return;
    const data = await res.json();
    const score = Number(data.sentiment_score ?? 0);
    const conf = Number(data.sentiment_confidence ?? 0);
    setClassText(".js-sentiment-value", score.toFixed(3));
    setClassText(".js-sentiment-confidence", conf.toFixed(3));
    const pct = Math.max(0, Math.min(100, ((score + 1) / 2) * 100));
    document.querySelectorAll(".js-sentiment-bar").forEach(el => el.style.width = `${pct}%`);
    window.botMetrics.sentiment = { score, confidence: conf, barPct: pct };
    setLiveUpdatedAt();
  } catch (err) {
    console.warn("[sentiment] Fetch failed:", err);
  }
}

async function refreshBotStatus() {
  try {
    const res = await fetch("/bot/status");
    if (!res.ok) return;
    const data = await res.json();
    setClassText(".js-bot-status", `Bot status: ${data.bot_status}`);
    window.botMetrics.botStatusLine = `Bot status: ${data.bot_status}`;
    setLiveUpdatedAt();
  } catch (err) {
    console.warn("[bot/status] Fetch failed:", err);
  }
}

async function refreshStorageHealth() {
  try {
    const res = await fetch("/api/v1/system/storage");
    const data = await res.json();
    if (!res.ok) return;
    const disk = data.disk || {};
    const stats = data.stats || {};
    const usagePct = Number(disk.usage_pct || 0);
    setClassText(".js-storage-usage-pct", `${usagePct.toFixed(2)}%`);
    document.querySelectorAll(".js-storage-usage-bar").forEach(bar => {
      bar.style.width = `${Math.max(0, Math.min(100, usagePct))}%`;
      bar.classList.toggle("storage-critical", usagePct >= 85);
      bar.classList.toggle("storage-warning", usagePct >= 70 && usagePct < 85);
    });
    setClassText(".js-storage-opt-stats", `Laatste opschoning bespaarde: ${bytesToMB(stats.saved_bytes)} MB`);
    setClassText(".js-storage-health", `Historie: ${Number(stats.history_days || 400)} dagen | Resolutie: ${stats.resolution || "Mixed (1s/1m)"}`);
    window.botMetrics.storage = {
      usagePct,
      savedBytes: stats.saved_bytes,
      historyDays: Number(stats.history_days || 400),
      resolution: stats.resolution || "Mixed (1s/1m)",
    };
  } catch (_err) {
    // keep prior values on transient fetch issues
  }
}

function toEpochSeconds(dateText) {
  const dtStr = String(dateText).endsWith('Z') || String(dateText).includes('+') ? dateText : String(dateText) + 'Z';
  const ms = new Date(dtStr).getTime();
  if (!Number.isFinite(ms) || ms <= 0) return null;
  // Forceer +2 uur (7200 seconden) offset voor de grafiek weergave zoals gevraagd
  return Math.floor(ms / 1000) + 7200;
}

function createLineSeries(chart, options) {
  if (chart && typeof chart.addLineSeries === "function") {
    return chart.addLineSeries(options);
  }
  if (chart && typeof chart.addSeries === "function" && LightweightCharts?.LineSeries) {
    return chart.addSeries(LightweightCharts.LineSeries, options);
  }
  return null;
}

function createCandlestickSeries(chart, options) {
  if (chart && typeof chart.addCandlestickSeries === "function") {
    return chart.addCandlestickSeries(options);
  }
  if (chart && typeof chart.addSeries === "function" && LightweightCharts?.CandlestickSeries) {
    return chart.addSeries(LightweightCharts.CandlestickSeries, options);
  }
  return null;
}

function setSeriesMarkers(series, markers) {
  if (!series) return;
  if (typeof series.setMarkers === "function") {
    series.setMarkers(markers);
    return;
  }
  if (LightweightCharts?.createSeriesMarkers) {
    LightweightCharts.createSeriesMarkers(series, markers);
  }
}

function toMarkerTime(ts) {
  return toEpochSeconds(ts);
}

function markerVisualProfile() {
  if (!priceChart || !rawChartLineData.length) return { thresholdPx: 34, hideText: false, sizeMode: "normal" };
  const range = priceChart.timeScale().getVisibleLogicalRange?.();
  if (!range || !Number.isFinite(range.from) || !Number.isFinite(range.to)) {
    return { thresholdPx: 34, hideText: false, sizeMode: "normal" };
  }
  const host = document.getElementById("priceChart");
  const hostWidth = Math.max(320, Number(host?.clientWidth || 0));
  const dpr = Math.max(1, Number(window.devicePixelRatio || 1));
  const widthFactor = hostWidth >= 2200 ? 1.35 : hostWidth >= 1600 ? 1.2 : hostWidth >= 1200 ? 1.08 : 1.0;
  const dprFactor = dpr >= 2.5 ? 1.35 : dpr >= 2 ? 1.2 : dpr >= 1.5 ? 1.1 : 1.0;
  const adaptiveFactor = widthFactor * dprFactor;
  const visibleBars = Math.max(1, range.to - range.from);
  if (visibleBars > 260) return { thresholdPx: Math.round(42 * adaptiveFactor), hideText: true, sizeMode: "small" };
  if (visibleBars > 160) return { thresholdPx: Math.round(38 * adaptiveFactor), hideText: true, sizeMode: "small" };
  if (visibleBars > 110) return { thresholdPx: Math.round(34 * adaptiveFactor), hideText: true, sizeMode: "normal" };
  return { thresholdPx: Math.round(30 * adaptiveFactor), hideText: false, sizeMode: "normal" };
}

function clusterMarkers(markers, thresholdPx) {
  if (!priceChart || !markerSeries || !Array.isArray(markers) || markers.length <= 1) return markers || [];
  const clustered = [];
  let bucket = [];
  const flush = () => {
    if (!bucket.length) return;
    if (bucket.length <= 3) {
      clustered.push(...bucket);
      bucket = [];
      return;
    }
    const first = bucket[0];
    const sellCount = bucket.filter((m) => m.position === "aboveBar").length;
    const sideSell = sellCount >= Math.ceil(bucket.length / 2);
    clustered.push({
      ...first,
      position: sideSell ? "aboveBar" : "belowBar",
      color: sideSell ? "#ff3131" : "#39ff14",
      shape: sideSell ? "arrowDown" : "arrowUp",
      text: `x${bucket.length}`,
      size: 2,
    });
    bucket = [];
  };

  for (const marker of markers) {
    const t = marker.time;
    const y = markerSeries.priceToCoordinate?.(Number(marker.price_hint));
    const x = priceChart.timeScale().timeToCoordinate?.(t);
    if (!Number.isFinite(x)) {
      flush();
      clustered.push(marker);
      continue;
    }
    if (!bucket.length) {
      bucket.push({ ...marker, _x: x, _y: y });
      continue;
    }
    const prev = bucket[bucket.length - 1];
    const dx = Math.abs(x - prev._x);
    const dy = Number.isFinite(y) && Number.isFinite(prev._y) ? Math.abs(y - prev._y) : 0;
    if (dx <= thresholdPx && dy <= thresholdPx * 0.75) bucket.push({ ...marker, _x: x, _y: y });
    else {
      flush();
      bucket.push({ ...marker, _x: x, _y: y });
    }
  }
  flush();
  return clustered.map((m) => {
    const out = { ...m };
    delete out._x;
    delete out._y;
    delete out.price_hint;
    return out;
  });
}

function renderAdaptiveMarkers() {
  if (!markerSeries) return;
  const profile = markerVisualProfile();
  markerRenderState = profile;
  const base = rawChartMarkers
    .map((m) => {
      const signal = String(m.signal || "").toUpperCase();
      if (signal !== "BUY" && signal !== "SELL") return null;
      return {
        time: toMarkerTime(m.ts),
        position: signal === "SELL" ? "aboveBar" : "belowBar",
        color: signal === "SELL" ? "#ff3131" : "#39ff14",
        shape: signal === "SELL" ? "arrowDown" : "arrowUp",
        text: "",
        size: profile.sizeMode === "small" ? 0 : 1,
        avoidOverlapping: true,
        price_hint: Number(m.price || m.marker_price || m.latest_close || 0),
      };
    })
    .filter((m) => m && m.time !== null);
  const clustered = clusterMarkers(base, profile.thresholdPx).sort((a, b) => Number(a.time) - Number(b.time));
  setSeriesMarkers(markerSeries, clustered);
}

function ensureChartAndSeries(host) {
  const rect0 = host.getBoundingClientRect();
  const hostWidth = Math.max(280, Math.floor(rect0.width || host.clientWidth || 0) || 320);
  let hostHeight = Math.floor(rect0.height || host.clientHeight || 0);
  if (!Number.isFinite(hostHeight) || hostHeight < 1) hostHeight = Math.floor(host.offsetHeight || 280);
  hostHeight = Math.max(200, hostHeight);

  if (!priceChart) {
    priceChart = LightweightCharts.createChart(host, {
      autoSize: false,
      layout: {
        background: { color: "#000000" },
        textColor: "#888888",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.2)" },
        horzLines: { color: "rgba(255, 255, 255, 0.2)" },
      },
      width: hostWidth,
      height: hostHeight,
      leftPriceScale: { visible: false },
      rightPriceScale: {
        borderColor: "#FFFFFF",
        textColor: "#FFFFFF",
        scaleMargins: { top: 0.06, bottom: 0.22 },
      },
      timeScale: {
        visible: true,
        ticksVisible: true,
        borderColor: "#FFFFFF",
        borderVisible: true,
        rightOffset: 20,
        timeVisible: true,
        secondsVisible: false,
        barSpacing: 16,
        tickMarkFormatter: (time) => formatChartAxisTimeHm(time),
        textColor: "#888888",
      },
      localization: { locale: "nl-NL" },
    });
    const onLwViewportChange = () => {
      renderAdaptiveMarkers();
      scheduleSyncPredictionChartJsXFromLightweight();
    };
    priceChart.timeScale().subscribeVisibleLogicalRangeChange?.(onLwViewportChange);
    priceChart.timeScale().subscribeVisibleTimeRangeChange?.(onLwViewportChange);
    window.addEventListener("resize", () => {
      resizePriceChart(host);
    });
  }

  if (!priceChartResizeObserver && typeof ResizeObserver !== "undefined") {
    priceChartResizeObserver = new ResizeObserver(() => {
      resizePriceChart(host);
      if (typeof window.resizeChartsAfterTerminalMidLayout === "function") {
        window.resizeChartsAfterTerminalMidLayout();
      }
    });
    priceChartResizeObserver.observe(host);
    const midCol = document.querySelector("#tab-terminal .dashboard-col-main.tcc-market-col");
    if (midCol) priceChartResizeObserver.observe(midCol);
    const cmd =
      document.querySelector("#tab-terminal #dashboard.tcc-command-grid") ||
      document.querySelector("#tab-terminal .dashboard-container.tcc-command-grid");
    if (cmd) priceChartResizeObserver.observe(cmd);
  }

  if (!priceSeries) {
    priceSeriesIsCandle = false;
    priceSeries = createCandlestickSeries(priceChart, {
      upColor: "#39ff14",
      downColor: "#ff3131",
      borderVisible: true,
      wickUpColor: "#39ff14",
      wickDownColor: "#ff3131",
    });
    if (priceSeries) {
      priceSeriesIsCandle = true;
    } else {
      priceSeries = createLineSeries(priceChart, { color: "#00f5ff", lineWidth: 4 });
    }
    if (priceSeries) {
      markerSeries = priceSeries;
    }
  }

  if (!predictionOverlaySeries && priceChart && window.LightweightCharts) {
    const LW = window.LightweightCharts;
    const dashStyle = LW.LineStyle !== undefined ? LW.LineStyle.Dashed : 2;
    predictionOverlaySeries = createLineSeries(priceChart, {
      color: "#38bdf8",
      lineWidth: 2,
      lineStyle: dashStyle,
      lastValueVisible: true,
      priceLineVisible: false,
    });
  }
}

function resizePriceChart(host) {
  if (!priceChart || !host) return;
  const rect = host.getBoundingClientRect();
  let w = Math.floor(rect.width || host.clientWidth || host.offsetWidth || 0);
  let h = Math.floor(rect.height || host.clientHeight || 0);
  if (!Number.isFinite(w) || w < 1) w = Math.floor(host.offsetWidth || 320);
  if (!Number.isFinite(h) || h < 1) h = Math.floor(host.offsetHeight || 280);
  w = Math.max(280, w);
  h = Math.max(200, h);
  priceChart.applyOptions({ width: w, height: h });
  updateChartTimeline();
}

function updateChartTimeline() {
  const el = document.getElementById("chartTimeline");
  if (!el) return;
  const pair = _normMkChart(selectedChartPair || selectedMarket || "");
  const arr = Array.isArray(rawChartLineData) ? rawChartLineData : [];
  if (!arr.length) {
    el.style.display = "none";
    return;
  }
  el.style.display = "block";
  const fmt = (t) => formatChartAxisTimeHm(t) || "--:--";
  const first = arr[0];
  const mid = arr[Math.floor(arr.length / 2)];
  const last = arr[arr.length - 1];
  el.textContent = `${pair} · ${fmt(first?.time)}  |  ${fmt(mid?.time)}  |  ${fmt(last?.time)}`;
}

function buildOrUpdateChart(labels, prices, markers, whaleDangerZone = null, isNewPair = false) {
  // Tab-Isolatie: Render overslaan als terminal niet zichtbaar is
  if (activeTab !== "terminal") return;

  const host = document.getElementById("priceChart");
  if (!host) return;
  try {
    ensureChartAndSeries(host);
  } catch (_err) {
    requestAnimationFrame(() => {
      try {
        ensureChartAndSeries(host);
      } catch (_) {}
    });
    return;
  }

  if (!priceSeries || !priceChart) {
    document.getElementById("prediction").textContent =
      "Chart init error: Lightweight Charts API mismatch. Reload after rebuild.";
    return;
  }
  resizePriceChart(host);

  let currentRange = null;
  let isAtEnd = true;
  if (priceChart && rawChartLineData.length > 0 && !isNewPair) {
    currentRange = priceChart.timeScale().getVisibleLogicalRange();
    if (currentRange) {
      isAtEnd = currentRange.to >= rawChartLineData.length - 3;
    }
  }

  const lineData = labels
    .map((d, i) => {
      const t0 = toEpochSeconds(d);
      const time = t0 === null || t0 === undefined ? null : normalizeChartTimeSeconds(t0);
      return { time, value: Number(prices[i]) };
    })
    .filter((p) => p.time !== null && Number.isFinite(p.value))
    .sort((a, b) => Number(a.time) - Number(b.time));
  if (!lineData.length) {
    const skelEmpty = document.getElementById("priceChartSkeleton");
    if (skelEmpty) skelEmpty.classList.remove("is-visible");
    updateChartTimeline();
    return;
  }
  rawChartLineData = lineData;
  updateChartTimeline();
  if (priceSeriesIsCandle) {
    const candles = [];
    for (let i = 0; i < lineData.length; i += 1) {
      const cur = lineData[i];
      const prevVal = i > 0 ? lineData[i - 1].value : cur.value;
      const o = Number(prevVal);
      const c = Number(cur.value);
      const wick = Math.max(Math.abs(c), Math.abs(o)) * 0.002 + 1e-8;
      candles.push({
        time: cur.time,
        open: o,
        high: Math.max(o, c) + wick,
        low: Math.min(o, c) - wick,
        close: c,
      });
    }
    priceSeries.setData(candles);
    applyLwSeriesPriceFormat(
      priceSeries,
      candles.map((c) => c.close)
    );
  } else {
    priceSeries.setData(lineData);
    applyLwSeriesPriceFormat(
      priceSeries,
      lineData.map((p) => p.value)
    );
  }
  rawChartMarkers = Array.isArray(markers) ? markers.slice() : [];
  renderAdaptiveMarkers();
  const skel = document.getElementById("priceChartSkeleton");
  if (skel && lineData.length) skel.classList.remove("is-visible");
  const hostEl = document.getElementById("priceChart");
  const danger = Boolean(whaleDangerZone && whaleDangerZone.active);
  if (hostEl) hostEl.classList.toggle("whale-danger-zone", danger);
  if (whaleDangerPriceLineHandle) {
    try {
      if (typeof whaleDangerPriceLineHandle.remove === "function") whaleDangerPriceLineHandle.remove();
      else if (priceSeries && typeof priceSeries.removePriceLine === "function") {
        priceSeries.removePriceLine(whaleDangerPriceLineHandle);
      }
    } catch (_e) {}
    whaleDangerPriceLineHandle = null;
  }
  if (
    danger &&
    priceSeries &&
    lineData.length &&
    typeof priceSeries.createPriceLine === "function"
  ) {
    const lastPx = Number(lineData[lineData.length - 1].value);
    if (Number.isFinite(lastPx) && lastPx > 0) {
      try {
        whaleDangerPriceLineHandle = priceSeries.createPriceLine({
          price: lastPx,
          color: "rgba(255, 49, 49, 0.92)",
          lineWidth: 2,
          lineStyle: 2,
          axisLabelVisible: true,
          title: "⚠ Whale Danger Zone",
        });
      } catch (_e) {
        whaleDangerPriceLineHandle = null;
      }
    }
  }

  // Zoom persistentie & Scroll-beveiliging
  if (isNewPair || !currentRange) {
    // Forceer een standaard inzoom-niveau (laatste 120 candles) in plaats van alles plat te drukken
    const totalBars = lineData.length;
    if (totalBars > 120) {
      priceChart.timeScale().setVisibleLogicalRange({
        from: totalBars - 120,
        to: totalBars + 2
      });
    } else {
      priceChart.timeScale().fitContent();
    }
  } else {
    if (!isAtEnd) {
      // Gebruiker is terug aan het scrollen, behoud de exacte viewport (Geen auto-shift)
      priceChart.timeScale().setVisibleLogicalRange(currentRange);
    } else {
      // Gebruiker is aan het einde, schuif viewport soepel mee met behoud van het zoomniveau
      const rangeWidth = currentRange.to - currentRange.from;
      const newMax = lineData.length;
      priceChart.timeScale().setVisibleLogicalRange({
        from: newMax - rangeWidth,
        to: newMax + 2
      });
    }
  }

  priceChart.applyOptions({
    rightPriceScale: {
      scaleMargins: { top: 0.06, bottom: 0.22 },
    },
    timeScale: {
      visible: true,
      ticksVisible: true,
      borderVisible: true,
      rightOffset: 20,
      timeVisible: true,
      secondsVisible: false,
      tickMarkFormatter: (time) => formatChartAxisTimeHm(time),
    },
  });

  void refreshMainChartGhostLine(selectedChartPair || selectedMarket || "BTC-EUR");
  void updatePredictionChart(String(selectedChartPair || selectedMarket || "BTC-EUR")).then(() => {
    scheduleSyncPredictionChartJsXFromLightweight();
  });
}

/** API predicted_price als blauwe stippellijn op de hoofdgrafiek (LightweightCharts). */
async function refreshMainChartGhostLine(ticker) {
  if (headlessMode || activeTab !== "terminal") return;
  const sym = encodeURIComponent(String(ticker || "BTC-EUR").toUpperCase());
  try {
    const response = await fetch(`/api/v1/predictions?symbol=${sym}`);
    if (!response.ok) return;
    const data = await response.json();
    const pred = Array.isArray(data.predicted) ? data.predicted : [];
    const host = document.getElementById("priceChart");
    if (!host || !priceChart) return;
    ensureChartAndSeries(host);
    if (!predictionOverlaySeries) return;

    const pts = [];
    for (let i = 0; i < pred.length; i += 1) {
      const row = pred[i];
      const tRaw = predictionRowToEpochSeconds(row);
      const t = tRaw == null ? null : normalizeChartTimeSeconds(tRaw);
      const v = Number(row.predicted_price ?? row.price ?? row.value);
      if (t !== null && Number.isFinite(v) && v > 0) {
        pts.push({ time: t, value: v });
      }
    }
    pts.sort((a, b) => a.time - b.time);
    const dedup = [];
    let lastT = null;
    for (let j = 0; j < pts.length; j += 1) {
      if (lastT !== null && pts[j].time === lastT) {
        dedup[dedup.length - 1] = pts[j];
      } else {
        dedup.push(pts[j]);
        lastT = pts[j].time;
      }
    }
    if (rawChartLineData.length && dedup.length) {
      const last = rawChartLineData[rawChartLineData.length - 1];
      if (dedup[0].time > last.time) {
        dedup.unshift({ time: last.time, value: last.value });
      }
    }
    try {
      predictionOverlaySeries.setData(dedup);
      const priceSample = (Array.isArray(rawChartLineData) ? rawChartLineData : [])
        .map((p) => p.value)
        .concat(dedup.map((p) => p.value));
      applyLwSeriesPriceFormat(priceSeries, priceSample);
      applyLwSeriesPriceFormat(predictionOverlaySeries, dedup.map((p) => p.value));
      const predLast = dedup.length ? dedup[dedup.length - 1] : null;
      setChartTimeHint({
        market: decodeURIComponent(sym),
        updated_at: window.__lastEngineTickIso || null,
        predicted_at: predLast ? new Date(predLast.time * 1000).toISOString() : null,
      });
      try {
        window.__lastApiPredictionResponse = Object.assign({}, data, {
          symbol: decodeURIComponent(sym),
        });
      } catch (_e3) {
        /* ignore */
      }
      if (typeof window.updateTerminalChartJsPredictionDataset === "function") {
        window.updateTerminalChartJsPredictionDataset(data);
      }
    } catch (_e) {
      /* tijd niet strikt oplopend i.v.m. history offset — leegmaken voorkomt crash */
      try {
        predictionOverlaySeries.setData([]);
      } catch (_e2) {}
    }
  } catch (_err) {
    /* stil falen: Chart.js-paneel heeft nog fallback */
  }
}

function toggleChartFullscreen() {
  document.body.classList.toggle("sniper-fullscreen");
  const host = document.getElementById("priceChart");
  if (host) resizePriceChart(host);
}

function bindSniperPanelControls() {
  const headlessBtn = document.getElementById("toggleHeadlessBtn");
  const fsBtn = document.getElementById("toggleChartFullscreenBtn");
  headlessBtn?.addEventListener("click", () => {
    headlessMode = !headlessMode;
    document.body.classList.toggle("headless-mode", headlessMode);
    headlessBtn.textContent = headlessMode ? "Headless ON" : "Headless";
    const predictionEl = document.getElementById("prediction");
    if (predictionEl && headlessMode) {
      predictionEl.textContent = "Headless mode actief: chart rendering uitgeschakeld, trading-engine blijft actief.";
    }
    if (!headlessMode) {
      void refreshChart();
    }
  });
  fsBtn?.addEventListener("click", toggleChartFullscreen);
}

async function refreshChart() {
  if (headlessMode) return;
  await updateChart(selectedChartPair || selectedMarket);
}

async function updateChart(newPair) {
  // Tab-Isolatie & Headless check
  if (headlessMode || activeTab !== "terminal") return;
  const newChartPair = String(newPair || selectedMarket || "BTC-EUR").toUpperCase();
  const isNewPair = (newChartPair !== selectedChartPair) || !priceChart;
  selectedChartPair = newChartPair;
  const predictionEl = document.getElementById("prediction");
  const skel = document.getElementById("priceChartSkeleton");
  if (skel) skel.classList.add("is-visible");
  if (predictionEl) {
    predictionEl.textContent = `Laden van ${selectedChartPair} data...`;
  }
  try {
    const res = await fetch(
      `/api/v1/history?pair=${encodeURIComponent(selectedChartPair)}&lookback_days=180&interval=${encodeURIComponent(chartInterval || CHART_CANDLE_INTERVAL)}`
    );
    const data = await res.json();
    if (!res.ok) {
      if (predictionEl) predictionEl.textContent = `Kon ${selectedChartPair} data niet laden.`;
      if (skel) skel.classList.remove("is-visible");
      return;
    }
    if (predictionEl && String(predictionEl.textContent || "").startsWith("Laden van ")) {
      const iv = data.interval || CHART_CANDLE_INTERVAL;
      predictionEl.textContent = `Chart geladen (${iv}) voor ${data.tv_symbol || selectedChartPair}.`;
    }
    buildOrUpdateChart(
      data.labels || [],
      data.prices || [],
      data.markers || [],
      data.whale_danger_zone || null,
      isNewPair
    );
  } catch (_err) {
    if (predictionEl) predictionEl.textContent = `Laden van ${selectedChartPair} data mislukt.`;
    if (skel) skel.classList.remove("is-visible");
  }
}

function sentimentTagClass(label) {
  const s = String(label || "neutral").toLowerCase();
  if (s === "positive") return "sentiment-tag sentiment-tag--pos";
  if (s === "negative") return "sentiment-tag sentiment-tag--neg";
  return "sentiment-tag sentiment-tag--neu";
}

function createIntelligenceTickerItemButton(item) {
  const score = Number(item.finbert_score ?? item.sentiment ?? 0);
  const bull = score >= 0;
  const badgeClass = bull ? "cockpit-ticker-item-badge--bull" : "cockpit-ticker-item-badge--bear";
  const badgeText = bull ? "BULL" : "BEAR";
  const sym = String(item.ticker_tag || item.coin || "MKT").toUpperCase().slice(0, 10);
  const headline = escapeHtmlText(item.headline || item.text || item.title || "—");
  const when = formatNewsRelativeShort(item.ts || item.published_at);
  const row = document.createElement("button");
  row.type = "button";
  row.className = `cockpit-ticker-item${item.is_urgent ? " cockpit-ticker-item--urgent" : ""}`;
  row.innerHTML =
    `<span class="cockpit-ticker-item-badge ${badgeClass}">${badgeText}</span>` +
    `<span class="cockpit-ticker-item-sep" aria-hidden="true">|</span>` +
    `<span class="cockpit-ticker-item-symbol">${sym}</span>` +
    `<span class="cockpit-ticker-item-sep" aria-hidden="true">|</span>` +
    `<span class="cockpit-ticker-item-main">` +
    `<span class="cockpit-ticker-item-headline">${headline}</span>` +
    `<span class="cockpit-ticker-item-time">${when}</span>` +
    `</span>`;
  row.addEventListener("click", () => {
    if (item.is_social_stub) return;
    openNewsModal(item);
  });
  return row;
}

/** Onder de hoofdgrafiek: horizontale marquee + badges (BULL/BEAR | ticker | headline). */
function renderIntelligenceTicker(items) {
  const root = document.getElementById("intelligenceTickerNews");
  const wrap = document.getElementById("intelligenceTickerMarquee");
  if (!root) return;
  root.innerHTML = "";
  if (wrap) wrap.classList.add("cockpit-ticker-marquee--static");
  const sorted = [...items].sort((a, b) => Number(Boolean(b.is_urgent)) - Number(Boolean(a.is_urgent)));
  if (!sorted.length) {
    root.innerHTML = `<div class="cockpit-ticker-empty">Geen nieuwsitems beschikbaar.</div>`;
    return;
  }
  if (wrap) wrap.classList.remove("cockpit-ticker-marquee--static");
  const slice = sorted.slice(0, 32);
  const dur = Math.min(160, Math.max(36, slice.length * 6));
  root.style.setProperty("--cockpit-marquee-duration", `${dur}s`);
  for (const item of slice) {
    root.appendChild(createIntelligenceTickerItemButton(item));
  }
  const reduceMotion =
    typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduceMotion) {
    for (const item of slice) {
      const dup = createIntelligenceTickerItemButton(item);
      dup.setAttribute("aria-hidden", "true");
      dup.tabIndex = -1;
      root.appendChild(dup);
    }
  }
}

/** Terminal nieuwskolom: max 3 headlines met sentiment-badges. */
function renderNewsStream(containerId, items) {
  const root = document.getElementById(containerId);
  if (!root) return;
  root.innerHTML = "";
  const sorted = [...items].sort((a, b) => Number(Boolean(b.is_urgent)) - Number(Boolean(a.is_urgent)));
  if (!sorted.length) {
    root.innerHTML = `<div class="terminal-news-empty">Geen nieuwsitems beschikbaar.</div>`;
    return;
  }
  for (const item of sorted.slice(0, 3)) {
    const score = Number(item.finbert_score ?? item.sentiment ?? 0);
    const pos = score >= 0;
    const badgeClass = pos ? "terminal-news-badge--bull" : "terminal-news-badge--bear";
    const badgeText = pos ? "BULL" : "BEAR";
    const div = document.createElement("div");
    div.className = `terminal-news-row ${item.is_urgent ? "terminal-news-row--urgent" : ""}`;
    div.innerHTML =
      `<span class="terminal-news-badge ${badgeClass}">${badgeText}</span>` +
      `<div class="terminal-news-body">` +
      `<div class="terminal-news-headline">${item.headline || item.text || "-"}</div>` +
      `<div class="terminal-news-meta">${item.source || "—"} · ${formatNewsTimeAmsterdam(item.ts || item.published_at)}</div>` +
      `</div>`;
    div.addEventListener("click", () => openNewsModal(item));
    root.appendChild(div);
  }
}

function renderNewsBlock(containerId, items, withInsights = false) {
  const root = document.getElementById(containerId);
  if (!root) return;
  root.innerHTML = "";
  const sorted = [...items].sort((a, b) => Number(Boolean(b.is_urgent)) - Number(Boolean(a.is_urgent)));
  for (const item of sorted.slice(0, 5)) {
    const div = document.createElement("div");
    div.className = `news-item ${item.is_urgent ? "news-item-urgent" : ""}`;
    div.innerHTML =
      `<div class="news-headline-line"><strong class="news-headline">${item.headline || "-"}</strong> ${item.is_urgent ? '<span class="urgent-chip">BREAKING</span>' : ""}</div>` +
      `<div class="muted">${item.source || "Unknown source"} | ${formatNewsTimeAmsterdam(item.ts)}</div>` +
      `<div class="impact-row">` +
      `${sourceBadge(item.source_icon, item.source)}` +
      `<span class="impact-left tag ${coinBadgeClass(item.ticker_tag)}">${item.ticker_tag || "-"}</span>` +
      `</div>` +
      (withInsights
        ? `<div class="muted">${item.explanation || "No AI explanation available."}</div>`
        : "");
    div.addEventListener("click", () => openNewsModal(item));
    root.appendChild(div);
  }
}

async function refreshNewsInsights() {
  const [newsRes, actRes] = await Promise.allSettled([
    fetch("/api/v1/news/ticker?elite_mix=1"),
    fetch(
      typeof window.buildActivityFetchUrl === "function" ? window.buildActivityFetchUrl() : "/activity",
      window.activityFetchInit || { cache: "no-store", credentials: "same-origin" }
    ),
  ]);
  const data = newsRes.status === "fulfilled" && newsRes.value.ok ? await newsRes.value.json() : [];
  const act = actRes.status === "fulfilled" && actRes.value.ok ? await actRes.value.json() : {};
  let buzzData = {};
  try {
    const buzzRes = await fetch("/api/v1/social/buzz");
    if (buzzRes.ok) buzzData = await buzzRes.json();
  } catch (_) {
    buzzData = {};
  }
  const items = Array.isArray(data) ? data : [];
  latestNewsItems = items;
  const scannerFeed = Array.isArray(act.scanner_intel_feed) ? act.scanner_intel_feed : [];
  const scannerTickerItems = scannerFeed.map((row) => {
    const headline = String(row.headline || row.text || row.title || "Scanner update");
    const coin = String(row.coin || row.ticker_tag || "MKT").toUpperCase();
    return {
      headline,
      title: headline,
      summary: String(row.summary || ""),
      url: String(row.url || ""),
      source: String(row.source || "Scanner"),
      ts: row.published_at || row.publishedAt || act.started_at,
      published_at: row.published_at || row.publishedAt,
      ticker_tag: coin,
      finbert_score: Number(row.sentiment || 0),
      finbert_label: "neutral",
      finbert_confidence: 0.5,
      source_icon: null,
      is_urgent: Boolean(row.is_urgent !== false),
      social_volume: 0,
      impacts_ticker: true,
      explanation: "",
      ai_reasoning: "",
      affected_tickers: [coin],
      is_scanner_stub: true,
    };
  });
  const mapped = items.map((i) => ({
    headline: i.text,
    title: i.title || i.text,
    summary: i.summary || "",
    url: i.url || "",
    source: (typeof i.source === "object" ? i.source.name : i.source) || "News",
    ts: i.published_at,
    published_at: i.published_at,
    ticker_tag: i.coin,
    finbert_score: i.sentiment,
    finbert_label: Number(i.sentiment) >= 0 ? "positive" : "negative",
    finbert_confidence: i.confidence,
    source_icon: i.source_icon,
    is_urgent: Boolean(i.is_urgent),
    social_volume: Number(i.social_volume || 0),
    impacts_ticker: true,
    explanation: i.explanation,
    ai_reasoning: i.ai_reasoning,
    affected_tickers: i.affected_tickers || [i.coin],
  }));
  const buzzLines = Array.isArray(buzzData.lines) ? buzzData.lines : [];
  const socialTickerItems = buzzLines.map((ln) => ({
    headline: ln.headline || ln.ticker_line || "Social update",
    title: ln.headline || ln.ticker_line || "Social update",
    summary: "",
    url: "",
    source: "Social Buzz",
    ts: buzzData.updated_at || new Date().toISOString(),
    published_at: buzzData.updated_at,
    ticker_tag: ln.symbol || "MKT",
    finbert_score: Number(ln.velocity_pct_1h || 0) >= 0 ? 0.22 : -0.22,
    finbert_label: Number(ln.velocity_pct_1h || 0) >= 0 ? "positive" : "negative",
    finbert_confidence: 0.5,
    source_icon: null,
    is_urgent: Boolean(ln.high_interest),
    social_volume: Number(ln.velocity_pct_1h || 0),
    impacts_ticker: true,
    explanation: ln.regime ? `Regime: ${ln.regime}` : "",
    ai_reasoning: "",
    affected_tickers: [ln.symbol || "MKT"],
    is_social_stub: true,
  }));
  renderNewsStream("terminalNewsStream", mapped);
  renderIntelligenceTicker([...scannerTickerItems, ...socialTickerItems, ...mapped]);
  renderTickerTape(items);
}

function renderTickerTrack(trackId, items) {
  const root = document.getElementById(trackId);
  if (!root) return;
  root.innerHTML = "";
  const feed = items.slice(0, 16);
  if (!feed.length) {
    root.innerHTML = `<span class="ticker-item">No mapped high-impact news available.</span>`;
    return;
  }
  for (const item of feed) {
    const sent = Number(item.sentiment || 0);
    const span = document.createElement("span");
    span.className = `ticker-item ${sentimentToneClass(sent)}`;
    span.innerHTML =
      `<span class="tag ${coinBadgeClass(item.coin)}">${item.coin}</span> ` +
      `${sent >= 0 ? "+" : ""}${sent.toFixed(2)}: ${item.text}`;
    span.addEventListener("click", () => openNewsModal(item));
    span.style.cursor = "pointer";
    root.appendChild(span);
  }
}

function renderTickerTape(items) {
  renderTickerTrack("tickerTrack", items);
}

function setChartTimeHint(meta) {
  const el = document.getElementById("chartTimeHint");
  if (!el) return;
  const m = meta && typeof meta === "object" ? meta : {};
  const market = String(m.market || getActiveMarketForPolicyUi() || "BTC-EUR").toUpperCase().replace("/", "-");
  const updRaw = m.updated_at || m.last_engine_tick_utc || m.generated_at || m.ts || null;
  const predRaw = m.predicted_at || m.prediction_timestamp || null;

  const parse = (v) => {
    if (!v) return null;
    const d = new Date(v);
    return Number.isFinite(d.getTime()) ? d : null;
  };
  const fmtClock = (d) => (d ? d.toLocaleTimeString("nl-NL", { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—");
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

  const ageMs = updDt ? Date.now() - updDt.getTime() : 0;
  el.classList.toggle("is-stale", ageMs > 5 * 60 * 1000);
  el.classList.toggle("is-very-stale", ageMs > 15 * 60 * 1000);
}
window.setChartTimeHint = setChartTimeHint;

/** Shallow merge activity/WS payloads zodat policy-balkjes blijven werken bij gedeeltelijke updates (HOLD incl.). */
function mergeTerminalActivityPayload(data) {
  if (!data || typeof data !== "object") return {};
  const prev = window.__lastDashboardStats && typeof window.__lastDashboardStats === "object" ? window.__lastDashboardStats : {};
  window.__lastDashboardStats = { ...prev, ...data };
  return window.__lastDashboardStats;
}

function applyActivityResponse(data) {
  if (!data || typeof data !== "object") return;
  const merged = mergeTerminalActivityPayload(data);
  const activeU = normPairKey(getActiveMarketForPolicyUi());
  const incomingU = inferIncomingSymbol(merged);
  const symOk = payloadSymbolMatchesActive(merged);
  if (typeof window.updatePayloadFilterDebugBadge === "function") {
    window.updatePayloadFilterDebugBadge({
      active: activeU,
      incoming: incomingU || "",
      dropped: !symOk,
      src: "activity",
    });
  }
  if (!symOk) {
    if (payloadHasActiveSymbolData(merged, activeU)) {
      setChartTimeHint({
        market: merged.market || merged.selected_market || getActiveMarketForPolicyUi(),
        updated_at: merged.last_engine_tick_utc || (merged.last_prediction && merged.last_prediction.generated_at) || null,
        predicted_at: (merged.last_prediction && merged.last_prediction.generated_at) || null,
      });
      paintAiPolicyProbsFromPayload(merged);
    }
    return;
  }
  const inferTgl = document.getElementById("rlInferenceGreedyToggle");
  if (inferTgl && typeof merged.rl_inference_greedy === "boolean") {
    inferTgl.checked = merged.rl_inference_greedy;
  }
  updateTerminalHealthFromStats(merged);
  if (window.TerminalLiveTail && typeof window.TerminalLiveTail.bootstrapFromServerTail === "function") {
    window.TerminalLiveTail.bootstrapFromServerTail(merged.cockpit_log_tail);
  }
  window.__lastEngineTickIso =
    merged.last_engine_tick_utc ||
    (merged.last_prediction && merged.last_prediction.generated_at) ||
    null;
  updateLastScanLabel();
  setChartTimeHint({
    market: merged.market || merged.selected_market || getActiveMarketForPolicyUi(),
    updated_at: merged.last_engine_tick_utc || (merged.last_prediction && merged.last_prediction.generated_at) || null,
    predicted_at: (merged.last_prediction && merged.last_prediction.generated_at) || null,
  });
  const ls = merged.last_scores;
  if (ls && typeof ls === "object") {
    const lsMarket = normPairKey(ls.market || ls.ticker || ls.symbol || merged.market || merged.symbol || "");
    const activeForScores = normPairKey(getActiveMarketForPolicyUi());
    /* Alleen sentiment overslaan bij mismatch — niet de hele cockpit (BUY/HOLD/SELL bleef dan “vast”). */
    if (!lsMarket || lsMarket === activeForScores) {
      const score = Number(ls.sentiment_score);
      const conf = Number(ls.sentiment_confidence ?? 0);
      if (Number.isFinite(score)) {
        setClassText(".js-sentiment-value", score.toFixed(3));
        setClassText(".js-sentiment-confidence", conf.toFixed(3));
        const pct = Math.max(0, Math.min(100, ((score + 1) / 2) * 100));
        document.querySelectorAll(".js-sentiment-bar").forEach((el) => {
          el.style.width = `${pct}%`;
        });
        window.botMetrics.sentiment = { score, confidence: conf, barPct: pct };
      }
    }
  }
  const p = merged.paper_portfolio || {};
  const alloc = merged.allocation_snapshot || {};
  const linesForUi = allocationDisplayLines(merged);
  window.__marketsInPosition = buildMarketsInPositionSet(merged);
  const allocRoot = document.getElementById("executiveAllocationSnapshot");
  if (allocRoot) {
    const sum = String(alloc.summary || "Allocatie: —");
    const lines = linesForUi;
    const rows = lines.map((r) => {
      const c = String(r.coin || "?").toUpperCase();
      const w = Number(r.weight_pct || 0);
      const wTxt = Number.isFinite(w) ? w.toFixed(2) : "—";
      const wClamp = Math.max(0, Math.min(100, Number.isFinite(w) ? w : 0));
      const pos = r.in_position ? ' <span class="alloc-chip">IN POSITIE</span>' : "";
      return `<li class="alloc-row" style="--weight:${wClamp}%">
        <span class="alloc-coin">${escapeHtmlText(c)}</span>
        <span class="alloc-weight">${wTxt}%</span>${pos}
      </li>`;
    });
    const list =
      rows.length > 0
        ? `<ul class="executive-allocation-coins">${rows.join("")}</ul>`
        : "<p class=\"executive-allocation-empty\">Geen actieve posities.</p>";
    allocRoot.innerHTML =
      `<p class="executive-allocation-summary">${escapeHtmlText(sum)}</p>` + list;
  }
  const fmtEur = (v) => {
    const n = Number(v);
    return Number.isFinite(n)
      ? `€${n.toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
      : "—";
  };
    setClassText(".js-portfolio-equity", p.equity != null ? fmtEur(p.equity) : "—");
    setClassText(".js-portfolio-cash", p.cash != null ? fmtEur(p.cash) : "—");
  const heq = document.getElementById("headerEquity");
  const hca = document.getElementById("headerCash");
  if (heq) heq.textContent = p.equity != null ? fmtEur(p.equity) : "—";
  if (hca) hca.textContent = p.cash != null ? fmtEur(p.cash) : "—";
  const hBal = document.getElementById("headerBalanceEuro");
  if (hBal && p.equity != null) {
    hBal.textContent = fmtEur(p.equity);
  }
    setClassText(".js-portfolio-stats-rest", `Pos Qty: ${p.position_qty ?? "-"} | Realized PnL: ${p.realized_pnl_eur ?? p.realized_pnl ?? "-"}`);
  const lastOrder = merged.last_order?.order || {};
  {
    const sig = (lastOrder.signal || "").toUpperCase();
    const posQty = Number(p.position_qty ?? 0);
    // Toon SELL/BUY alleen als het signaal ook daadwerkelijk een actieve positie weerspiegelt
    const sigDisplay = (sig === "SELL" && posQty <= 0) || sig === "HOLD" || !sig ? "—" : `${sig} ${lastOrder.ticker || ""} ${lastOrder.amount_quote_eur || lastOrder.amount_quote || ""}`.trim();
    setClassText(".js-active-orders", `Signaal: ${sigDisplay}`);
  }
  const fg = merged.fear_greed || {};
  const fgMain = document.getElementById("terminalFearGreed");
  const fgSub = document.getElementById("terminalFearGreedSub");
  const tickFg = document.getElementById("tickerMetricFg");
  const raw = fg.fear_greed_value;
  if (fgMain) {
    fgMain.textContent = raw != null && raw !== "" ? `${Number(raw).toFixed(0)} / 100` : "-";
  }
  if (fgSub) {
    const cls = String(fg.classification || "").trim();
    fgSub.textContent = cls ? cls : "";
  }
  if (tickFg && raw != null && raw !== "") {
    tickFg.textContent = Number(raw).toFixed(0);
  } else if (tickFg) {
    tickFg.textContent = "—";
  }
  const cyc = merged.last_order?.cycle_seq;
  if (cyc != null && cyc !== lastLedgerCycleSeq) {
    lastLedgerCycleSeq = cyc;
    const fs = String(merged.last_order?.engine_risk?.final_signal || "").toUpperCase();
  }
  const rp = merged.risk_profile;
  if (rp) {
      const bv = Number(rp.base_trade_eur);
      const mv = Number(rp.max_risk_pct);
      const sv = Number(rp.stop_loss_pct);
      setClassText(".js-risk-profile-base", Number.isFinite(bv) ? `€${bv.toFixed(0)}` : String(rp.base_trade_eur ?? "-"));
      setClassText(".js-risk-profile-max", Number.isFinite(mv) ? `${mv.toFixed(1)}%` : String(rp.max_risk_pct ?? "-"));
      setClassText(".js-risk-profile-sl", Number.isFinite(sv) ? `${sv.toFixed(1)}%` : String(rp.stop_loss_pct ?? "-"));
  }
  window.botMetrics.portfolio = {
    equity: p.equity ?? null,
    cash: p.cash ?? null,
    positionQty: p.position_qty ?? null,
    realizedPnlEur: p.realized_pnl_eur ?? p.realized_pnl ?? null,
    lastOrderSignal: lastOrder.signal ?? null,
    lastOrderTicker: lastOrder.ticker ?? null,
    fearGreedValue: raw,
    fearGreedClassification: String(fg.classification || "").trim(),
    riskProfile: rp || null,
  };
  window.__eliteCriticalStream = Boolean(
    Array.isArray(merged.elite_ai_signals) &&
      merged.elite_ai_signals.some((s) => s && (s.state === "panic" || s.whale_danger === true))
  );
  if (Array.isArray(merged.elite_ai_signals)) renderElite8AiStatusBar(merged.elite_ai_signals);
  void refreshWhaleRadar();
  applySelectedMarketPriceFromPaperPortfolio(p, merged);
  paintAiPolicyProbsFromPayload(merged);
  if (activeTab === "terminal" && !headlessMode) {
    void updatePredictionChart(getActiveMarketForPolicyUi());
  }
  renderBvThoughtFeed(merged);

  const sumEl = document.getElementById("hpDashAllocSummary");
  const chips = document.getElementById("hpDashAllocChips");
  if (sumEl) sumEl.textContent = String(alloc.summary || "Allocatie: —");
  if (chips) {
    const dashLines = linesForUi.slice(0, 10);
    chips.innerHTML = dashLines
      .map((r) => {
        const c = escapeHtmlText(String(r.coin || "?").toUpperCase());
        const w = Number(r.weight_pct || 0);
        const wTxt = Number.isFinite(w) ? `${w.toFixed(2)}%` : "—";
        const pos = r.in_position ? " in-pos" : "";
        return `<li class="hp-dash-chip${pos}"><span class="hp-dash-chip__sym">${c}</span><span class="hp-dash-chip__w">${wTxt}</span></li>`;
      })
      .join("");
  }
  const badgeEl = document.getElementById("hpDashAllocMainBadge");
  if (badgeEl) {
    const mkSel = String(document.getElementById("marketSelect")?.value || "BTC-EUR").toUpperCase();
    const line =
      linesForUi.find((r) => String((r && r.market) || "").toUpperCase() === mkSel) || linesForUi[0];
    const w = line ? Number(line.weight_pct) : NaN;
    badgeEl.textContent = Number.isFinite(w) ? `${w.toFixed(2)}%` : "—";
  }
  syncHeaderPositionLamp();
}

function executiveSnapshotIfStillLoading(msg) {
  const allocRoot = document.getElementById("executiveAllocationSnapshot");
  if (!allocRoot) return;
  const t = String(allocRoot.textContent || "").trim();
  if (t === "Laden…" || t === "Laden..." || /^laden/i.test(t)) {
    allocRoot.innerHTML = `<p class="executive-allocation-empty">${escapeHtmlText(msg)}</p>`;
  }
}

async function refreshActivity() {
  try {
    const { ok, status, data, parseError } = await fetchBalanceFromActivity();
    if (parseError) {
      executiveSnapshotIfStillLoading("Kon activity niet lezen (ongeldige JSON).");
      return;
    }
    if (!data || typeof data !== "object") {
      executiveSnapshotIfStillLoading("Kon activity niet lezen (ongeldige JSON).");
      return;
    }
    if (!ok) {
      executiveSnapshotIfStillLoading(`Activity niet beschikbaar (HTTP ${status}).`);
      return;
    }
    applyActivityResponse(data);
  } catch (err) {
    console.warn("[activity]", err);
    executiveSnapshotIfStillLoading("Activity netwerkfout — volgende poll opnieuw.");
  }
}

function applyTradingRedisPayload(raw) {
  if (!raw || typeof raw !== "object") return;
  if (raw.type === "error" || raw.type === "ping") return;
  const activeU = normPairKey(getActiveMarketForPolicyUi());
  const incomingU = inferIncomingSymbol(raw);
  if (incomingU && incomingU !== activeU && !payloadHasActiveSymbolData(raw, activeU)) {
    if (typeof window.updatePayloadFilterDebugBadge === "function") {
      window.updatePayloadFilterDebugBadge({ active: activeU, incoming: incomingU, dropped: true, src: "ws" });
    }
    return;
  }
  if (raw.type === "cockpit_log_line" && raw.line) {
    const ln = String(raw.line);
    const hasPair = /[A-Z0-9]+-[A-Z0-9]+/.test(ln.toUpperCase());
    if (hasPair && !ln.toUpperCase().includes(activeU)) {
      if (typeof window.updatePayloadFilterDebugBadge === "function") {
        window.updatePayloadFilterDebugBadge({
          active: activeU,
          incoming: incomingU || "",
          dropped: true,
          src: "ws-log",
        });
      }
      return;
    }
    if (typeof window.updatePayloadFilterDebugBadge === "function") {
      window.updatePayloadFilterDebugBadge({
        active: activeU,
        incoming: incomingU || "",
        dropped: false,
        src: "ws-log",
      });
    }
    if (window.TerminalLiveTail && typeof window.TerminalLiveTail.push === "function") {
      window.TerminalLiveTail.push(ln);
    } else if (activeTab === "logs") {
      pushLogsTabLineFallback(ln);
    }
    return;
  }
  const data = {
    last_engine_tick_utc: raw.last_engine_tick_utc || null,
    last_prediction: raw.last_prediction || null,
    paper_portfolio: raw.paper_portfolio || {},
    last_order: raw.last_order || {},
    fear_greed: raw.fear_greed || {},
    risk_profile: raw.risk_profile || null,
    elite_ai_signals: raw.elite_ai_signals,
    allocation_snapshot: raw.allocation_snapshot || {},
    last_scores: raw.last_scores,
    active_markets: Array.isArray(raw.active_markets) ? raw.active_markets : undefined,
    rl_inference_greedy: typeof raw.rl_inference_greedy === "boolean" ? raw.rl_inference_greedy : undefined,
    rl_last_decision:
      raw.rl_last_decision && typeof raw.rl_last_decision === "object" ? raw.rl_last_decision : undefined,
    rl_multi_decisions:
      raw.rl_multi_decisions && typeof raw.rl_multi_decisions === "object" ? raw.rl_multi_decisions : undefined,
    worker_calc_hints_by_market:
      raw.worker_calc_hints_by_market && typeof raw.worker_calc_hints_by_market === "object"
        ? raw.worker_calc_hints_by_market
        : undefined,
    ai_action_probs:
      raw.ai_action_probs && typeof raw.ai_action_probs === "object" ? raw.ai_action_probs : undefined,
    worker_calc_hints: Array.isArray(raw.worker_calc_hints) ? raw.worker_calc_hints : undefined,
  };
  if (!data.last_engine_tick_utc && data.last_prediction && data.last_prediction.generated_at) {
    data.last_engine_tick_utc = data.last_prediction.generated_at;
  }
  applyActivityResponse(data);
  lastTradingWsActivityAtMs = Date.now();
  tradingUpdatesLive = true;
}

function scheduleTradingUpdatesSocketReconnect() {
  if (tradingUpdatesReconnectTimer != null) return;
  tradingUpdatesReconnectTimer = window.setTimeout(() => {
    tradingUpdatesReconnectTimer = null;
    connectTradingUpdatesSocket();
  }, TRADING_UPDATES_WS_RECONNECT_MS);
}

function connectTradingUpdatesSocket() {
  if (tradingUpdatesSocket && tradingUpdatesSocket.readyState === WebSocket.CONNECTING) return;
  if (tradingUpdatesSocket && tradingUpdatesSocket.readyState === WebSocket.OPEN) return;
  if (tradingUpdatesReconnectTimer != null) {
    window.clearTimeout(tradingUpdatesReconnectTimer);
    tradingUpdatesReconnectTimer = null;
  }
  try {
    if (tradingUpdatesSocket) {
      try {
        tradingUpdatesSocket.onclose = null;
        tradingUpdatesSocket.onerror = null;
        tradingUpdatesSocket.onmessage = null;
        if (
          tradingUpdatesSocket.readyState === WebSocket.OPEN ||
          tradingUpdatesSocket.readyState === WebSocket.CONNECTING
        ) {
          tradingUpdatesSocket.close();
        }
      } catch (_) {}
    }
  } catch (_) {}
  tradingUpdatesSocket = null;
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}/ws/trading-updates`;
  let ws;
  try {
    ws = new WebSocket(wsUrl);
  } catch (_e) {
    scheduleTradingUpdatesSocketReconnect();
    return;
  }
  tradingUpdatesSocket = ws;
  ws.onopen = () => {
    if (tradingUpdatesReconnectTimer != null) {
      window.clearTimeout(tradingUpdatesReconnectTimer);
      tradingUpdatesReconnectTimer = null;
    }
  };
  ws.onmessage = (event) => {
    try {
      const raw = JSON.parse(String(event.data || "{}"));
      applyTradingRedisPayload(raw);
    } catch (_) {}
  };
  ws.onclose = () => {
    tradingUpdatesLive = false;
    lastTradingWsActivityAtMs = 0;
    tradingUpdatesSocket = null;
    scheduleTradingUpdatesSocketReconnect();
  };
  ws.onerror = () => {
    try {
      ws.close();
    } catch (_) {}
  };
}

function applySystemStatsPayload(data) {
  if (!data || data.__ws === "hb" || data.topic !== "system_stats") return;
  const cpu = Math.max(0, Math.min(100, Number(data.cpu_pct) || 0));
  const ramPct = Math.max(0, Math.min(100, Number(data.ram_pct) || 0));
  const gpuSm = Math.max(0, Math.min(100, Number(data.gpu_util_pct) || 0));
  const gpuEffIn = data.gpu_util_effective;
  const gpuEff = Math.max(0, Math.min(100, Number(gpuEffIn)));
  let gpu = gpuEffIn !== undefined && gpuEffIn !== null && Number.isFinite(gpuEff) ? gpuEff : gpuSm;
  if (data.gpu_ok && (!Number.isFinite(gpu) || gpu < 1)) {
    gpu = 1;
  }
  const disk = Math.max(0, Math.min(100, Number(data.disk_pct) || 0));
  const ringCpu = document.getElementById("ringMeterCpu");
  const ringRam = document.getElementById("ringMeterRam");
  const ringGpu = document.getElementById("ringMeterGpu");
  const ringDisk = document.getElementById("ringMeterDisk");
  if (ringCpu) ringCpu.style.setProperty("--pct", String(cpu));
  if (ringRam) ringRam.style.setProperty("--pct", String(ramPct));
  if (ringGpu) ringGpu.style.setProperty("--pct", String(gpu));
  if (ringDisk) ringDisk.style.setProperty("--pct", String(disk));
  const cpuVal = document.getElementById("ringCpuVal");
  const ramVal = document.getElementById("ringRamVal");
  const gpuVal = document.getElementById("ringGpuVal");
  const diskVal = document.getElementById("ringDiskVal");
  if (cpuVal) cpuVal.textContent = `${cpu.toFixed(0)}%`;
  if (ramVal) ramVal.textContent = `${ramPct.toFixed(0)}%`;
  if (gpuVal) gpuVal.textContent = `${gpu.toFixed(0)}%`;
  if (diskVal) diskVal.textContent = `${disk.toFixed(0)}%`;
  const hCpu = document.getElementById("headerStatCpu");
  const hRam = document.getElementById("headerStatRam");
  const hGpu = document.getElementById("headerStatGpu");
  const hDisk = document.getElementById("headerStatDisk");
  if (hCpu) hCpu.textContent = `🖥️ ${cpu.toFixed(0)}%`;
  if (hRam) hRam.textContent = `🧠 ${ramPct.toFixed(0)}%`;
  if (hGpu) {
    hGpu.textContent = `🎮 ${gpu.toFixed(0)}%`;
    hGpu.classList.toggle("cockpit-gpu-neon", Boolean(data.gpu_ok) && gpu > 50);
    hGpu.classList.toggle("cockpit-gpu-glow", Boolean(data.gpu_ok) && gpu > 0);
    hGpu.classList.toggle("header-gpu-sensor-pulse", Boolean(data.gpu_ok) && gpu <= 0);
  }
  if (hDisk) hDisk.textContent = `💾 ${disk.toFixed(0)}%`;
  const gpuLive = Boolean(data.gpu_ok);
  if (ringGpu) {
    ringGpu.classList.toggle("ring-meter--gpu-active", gpuLive);
    ringGpu.classList.toggle("ring-meter--gpu-idle", !gpuLive);
  }
  const gpuMetaEl = document.getElementById("ringGpuMeta");
  if (gpuMetaEl) {
    const used = Number(data.vram_used_mb);
    const tot = Number(data.vram_total_mb);
    const name = String(data.gpu_name || "").trim();
    const parts = [];
    if (name) parts.push(name);
    if (Number.isFinite(tot) && tot > 0) {
      parts.push(`VRAM ${Math.round(used)} / ${Math.round(tot)} MB`);
    } else if (gpuLive && name) {
      parts.push(`GPU load ${gpu.toFixed(0)}%`);
    } else if (!gpuLive && !name) {
      parts.push("GPU niet beschikbaar (geen nvidia-smi)");
    }
    gpuMetaEl.textContent = parts.join(" · ");
  }

}

function stopSystemStatsSocket() {
  if (systemStatsReconnectTimer) {
    clearTimeout(systemStatsReconnectTimer);
    systemStatsReconnectTimer = null;
  }
  if (systemStatsSocket) {
    const sock = systemStatsSocket;
    try {
      if (sock.readyState === WebSocket.OPEN) {
        sock.close();
      } else if (sock.readyState === WebSocket.CONNECTING) {
        sock.onopen = () => sock.close();
      }
    } catch (_) {}
    systemStatsSocket = null;
  }
}

function connectSystemStatsSocket() {
  if (systemStatsSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(systemStatsSocket.readyState)) return;
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}/ws/system-stats`;
  systemStatsSocket = new WebSocket(wsUrl);
  systemStatsSocket.onmessage = (event) => {
    try {
      const data = normalizeSystemStatsWsPayload(event.data);
      if (!data) return;
      if (data.__ws === "hb") {
        try {
          systemStatsSocket.send("hb_ack");
        } catch (_) {}
        return;
      }
      enqueueCockpitWsRender(() => applySystemStatsPayload(data));
    } catch (_) {}
  };
  systemStatsSocket.onclose = () => {
    systemStatsSocket = null;
    systemStatsReconnectTimer = setTimeout(connectSystemStatsSocket, 4000);
  };
  systemStatsSocket.onerror = () => {
    try {
      systemStatsSocket.close();
    } catch (_) {}
  };
}

const BRAIN_NEON_LOSS_POLICY = "#00f8ff";
const BRAIN_NEON_LOSS_VALUE = "#ff3131";
const BRAIN_NEON_REWARD = "#39ff14";
const BRAIN_NEON_REWARD_MA = "#fff176";
const BRAIN_MIN_LR = 1.0e-5;
const BRAIN_MIN_EPS_PCT = 5.0;
/** Zelfde default als backend (`trading_core`) en `module_brain.js` bij ontbrekende reasoning/API. */
const BRAIN_REASONING_WAIT = "Wachten op eerste besluit (RL inferentie)...";
const REWARD_MA_WINDOW = 100;

function rewardMovingAverage(series, windowSize = REWARD_MA_WINDOW) {
  const src = Array.isArray(series) ? series.map((v) => Number(v || 0)) : [];
  const out = [];
  let sum = 0;
  const w = Math.max(1, Number(windowSize || 1));
  for (let i = 0; i < src.length; i += 1) {
    sum += src[i];
    if (i >= w) sum -= src[i - w];
    const n = Math.min(i + 1, w);
    out.push(n > 0 ? sum / n : src[i]);
  }
  return out;
}

/** UI reward-as: schaal naar [-1, 1] (fallback als backend geen reward_normalized stuurt). */
function normalizeRewardClient(series) {
  const src = Array.isArray(series) ? series.map((v) => Number(v) || 0) : [];
  if (!src.length) return [];
  const m = Math.max(...src.map((x) => Math.abs(x)), 1e-9);
  return src.map((x) => Math.max(-1, Math.min(1, x / m)));
}

function brainFeatureBucket(key) {
  const u = String(key || "").toLowerCase();
  if (u.includes("whale")) return "whale";
  if (/sentiment|news|social|fear|greed|nlp|buzz|finbert|macro_vol|btc_dom|dominance/.test(u)) return "social";
  return "technical";
}

function featureWeightsGroupedBarConfig(mergedFw, rawObs, fi) {
  const mandatorySignalKeys = ["sentiment_score", "news_confidence", "whale_pressure"];
  const merged = { ...Object.fromEntries(mandatorySignalKeys.map((k) => [k, Number(mergedFw[k] || 0)])), ...mergedFw };
  const pairs = Object.keys(merged)
    .map((k) => ({ k, v: Number(merged[k]) || 0 }))
    .sort((a, b) => b.v - a.v);
  const sortedKeys = pairs.map((p) => p.k);
  const sortedValsZoom = pairs.map((p) => Math.log10(1 + Math.max(0, Number(p.v) || 0) * 1000));
  const dTech = sortedKeys.map((k, i) => (brainFeatureBucket(k) === "technical" ? sortedValsZoom[i] : null));
  const dSoc = sortedKeys.map((k, i) => (brainFeatureBucket(k) === "social" ? sortedValsZoom[i] : null));
  const dWhale = sortedKeys.map((k, i) => (brainFeatureBucket(k) === "whale" ? sortedValsZoom[i] : null));
  return {
    type: "bar",
    data: {
      labels: sortedKeys.length ? sortedKeys : ["—"],
      datasets: [
        {
          label: "Technisch",
          data: sortedKeys.length ? dTech : [0],
          backgroundColor: "#00f8ff",
          stack: "w",
        },
        {
          label: "Sociaal",
          data: sortedKeys.length ? dSoc : [0],
          backgroundColor: "#fff176",
          stack: "w",
        },
        {
          label: "Whale",
          data: sortedKeys.length ? dWhale : [0],
          backgroundColor: "#39ff14",
          stack: "w",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: "top" },
        tooltip: {
          filter: (item) => {
            const y = item.parsed?.y;
            return y != null && Number(y) !== 0;
          },
          callbacks: {
            label(ctx) {
              const k = ctx.dataset.label || "";
              const lab = String(ctx.label || "");
              const y = ctx.parsed != null ? ctx.parsed.y : ctx.raw;
              if (y == null || Number(y) === 0) return "";
              const r = rawObs[lab];
              const pol = (fi.feature_weights_policy || {})[lab];
              const rawVal = Number(merged[lab] || 0);
              let line = `${lab} [${k}]: raw=${rawVal.toFixed(4)} | zoom=${Number(y).toFixed(4)}`;
              if (r !== undefined && r !== null && Number.isFinite(Number(r))) line += ` | RL-vector=${Number(r).toFixed(4)}`;
              if (pol !== undefined && pol !== null && Number.isFinite(Number(pol))) line += ` | policy-mix=${Number(pol).toFixed(4)}`;
              return line;
            },
          },
        },
      },
      scales: {
        x: {
          stacked: true,
          ticks: {
            maxRotation: 45,
            minRotation: 35,
            color: CHART_AXIS_WHITE,
          },
        },
        y: {
          stacked: true,
          beginAtZero: true,
        },
      },
    },
  };
}

function withAmbientSignalWeights(rawWeights, rawObs) {
  const out = { ...(rawWeights || {}) };
  const obs = rawObs || {};
  const ambientFloor = Math.max(0.0008, Number(window.BRAIN_AMBIENT_WEIGHT || 0.0012));
  const ambientKeys = ["sentiment_score", "news_confidence", "social_volume", "whale_pressure"];
  for (const k of ambientKeys) {
    const v = Number(out[k] || 0);
    const obsSeen = Object.prototype.hasOwnProperty.call(obs, k);
    if ((!Number.isFinite(v) || Math.abs(v) < 1e-12) && obsSeen) out[k] = ambientFloor;
  }
  return out;
}

function updateBrainTabFallback(hasData) {
  const el = document.getElementById("brainTabFallback");
  if (!el) return;
  if (hasData) el.classList.add("hidden");
  else el.classList.remove("hidden");
}

function paintBrainTabCharts(monitorData, fiData) {
  // Tab-Isolatie: negeer zware DOM-updates voor deze grafieken als tabs verborgen zijn
  if (activeTab !== "aibrain") return;

  if (!document.getElementById("brainTabTrainingLossChart")) return;
  const monitor = monitorData || {};
  const net = monitor.network_logs || {};
  const stats = monitor.stats || {};
  const policyLoss = Array.isArray(monitor.loss) ? monitor.loss : [];
  const valueLoss = Array.isArray(net.value_loss) ? net.value_loss : [];
  const rawRewards = Array.isArray(monitor.reward) ? monitor.reward : [];
  const fi = fiData || {};
  const fw = withAmbientSignalWeights(fi.feature_weights || {}, fi.rl_observation || {});
  const rawObs = fi.rl_observation || {};
  const mandatorySignalKeys = ["sentiment_score", "news_confidence", "whale_pressure"];
  const mergedFw = { ...Object.fromEntries(mandatorySignalKeys.map((k) => [k, Number(fw[k] || 0)])), ...fw };
  const pairs = Object.keys(mergedFw)
    .map((k) => ({ k, v: Number(mergedFw[k]) || 0 }))
    .sort((a, b) => b.v - a.v);
  const steps = Number(stats.global_step_count || 0);
  const hasRewardReal =
    rawRewards.length > 0 &&
    !(rawRewards.length === 1 && Math.abs(Number(rawRewards[0]) || 0) < 1e-12);
  const hasFeatures = pairs.length > 0;
  const hasLoss = policyLoss.length > 0 || valueLoss.length > 0;
  const hasData = hasLoss || hasRewardReal || hasFeatures || steps > 0;
  updateBrainTabFallback(hasData);
  const calibrationEl = document.getElementById("strategyCalibrationStatus");
  if (calibrationEl) {
    const mk = String(fi.market || selectedMarket || selectedChartPair || "BTC-EUR").toUpperCase();
    const calibrating = Boolean(fi.calibrating) || steps < 10 || !hasFeatures;
    const isTraining = Boolean(stats.is_training_active);
    const trainingBadge = isTraining 
        ? `<span style="color:#39ff14; text-shadow: 0 0 5px #39ff14; margin-left:8px;">[TRAINING: ACTIVE]</span>` 
        : `<span style="color:#888; margin-left:8px;" title="Zet RL_BACKGROUND_TRAIN=1 in je vault om continue te leren">[TRAINING: PAUSED]</span>`;
        
    calibrationEl.innerHTML = calibrating
      ? `Bezig met kalibreren... (${mk}) ${trainingBadge}`
      : `Strategy actief voor ${mk} ${trainingBadge}`;
  }
  const socialStrip = document.getElementById("socialBuzzStrip");
  if (socialStrip) {
    const buzz = fi.social_buzz || {};
    const lines = Array.isArray(buzz.lines) ? buzz.lines : [];
    const top = lines[0];
    const mk = String(fi.market || selectedMarket || selectedChartPair || "BTC-EUR").toUpperCase();
    const sym = mk.split("-")[0] || "MKT";
    const mine = lines.find((r) => String(r.market || "").toUpperCase() === mk) || top;
    const vel = Number(mine?.velocity_pct_1h || 0);
    const fill = Math.min(100, Math.max(4, 45 + Math.min(55, vel / 6)));
    const hi = Boolean(mine?.high_interest);
    const label = mine?.headline || mine?.ticker_line || `Social Buzz voor ${sym}: ${vel.toFixed(1)}% vs 1h baseline`;
    socialStrip.innerHTML =
      `<span class="social-buzz-strip__icon" title="Social momentum">📡</span>` +
      `<div class="social-buzz-strip__bar" aria-hidden="true"><div class="social-buzz-strip__fill" style="width:${fill}%"></div></div>` +
      `<span class="social-buzz-strip__text"${hi ? ' style="color:#39ff14"' : ""}>${escapeHtmlText(label)}</span>`;
  }

  let n = hasLoss ? Math.max(policyLoss.length, valueLoss.length, 1) : 0;
  const labels = n > 0 ? Array.from({ length: n }, (_, i) => String(i + 1)) : [];
  const pl = [];
  const vl = [];
  for (let i = 0; i < n; i += 1) {
    pl.push(i < policyLoss.length ? policyLoss[i] : null);
    vl.push(i < valueLoss.length ? valueLoss[i] : null);
  }
  const lossPointR = n <= 1 && hasLoss ? 5 : n <= 4 && hasLoss ? 3 : 0;
  const lossCfg = {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Policy loss",
          data: pl,
          borderColor: BRAIN_NEON_LOSS_POLICY,
          backgroundColor: "rgba(0, 248, 255, 0.06)",
          borderWidth: 3,
          pointRadius: lossPointR,
          tension: hasLoss ? 0.25 : 0,
          spanGaps: true,
        },
        {
          label: "Value loss",
          data: vl,
          borderColor: BRAIN_NEON_LOSS_VALUE,
          backgroundColor: "rgba(255, 49, 49, 0.08)",
          borderWidth: 3,
          pointRadius: lossPointR,
          tension: hasLoss ? 0.25 : 0,
          spanGaps: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: "top",
        },
      },
    },
  };
  if (window.ChartUtils) window.ChartUtils.upsertChart("brainTabTrainingLossChart", lossCfg);

  const featCfg = featureWeightsGroupedBarConfig(mergedFw, rawObs, fi);
  if (window.ChartUtils) window.ChartUtils.upsertChart("brainTabFeatureChart", featCfg);

  const normRewards =
    Array.isArray(monitor.reward_normalized) && monitor.reward_normalized.length
      ? monitor.reward_normalized.map((v) => Number(v) || 0)
      : normalizeRewardClient(rawRewards);
  let rewards = normRewards.slice();
  let rewardLabels = rewards.map((_, idx) => String(idx + 1));
  if (!rewards.length) {
    rewards = [0];
    rewardLabels = ["0"];
  }
  const rewardMa = rewardMovingAverage(rewards, REWARD_MA_WINDOW);
  const rewardCfg = {
    type: "line",
    data: {
      labels: rewardLabels,
      datasets: [
        {
          label: "Reward (genorm. −1…1)",
          data: rewards,
          borderColor: BRAIN_NEON_REWARD,
          backgroundColor: "rgba(57, 255, 20, 0.08)",
          borderWidth: 3,
          pointRadius: 0,
        },
        {
          label: `Reward MA(${REWARD_MA_WINDOW})`,
          data: rewardMa,
          borderColor: BRAIN_NEON_REWARD_MA,
          backgroundColor: "rgba(255, 241, 118, 0.03)",
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: true } },
      scales: {
        x: mergeContrastScale({ ticks: { maxRotation: 0 } }),
        y: mergeContrastScale({ min: -1, max: 1 }),
      },
    },
  };
  if (window.ChartUtils) window.ChartUtils.upsertChart("brainTabRewardErrorChart", rewardCfg);

  const entropySeries = Array.isArray(monitor.policy_entropy) ? monitor.policy_entropy : [];
  const lastEntropy = entropySeries.length ? entropySeries[entropySeries.length - 1] : null;
  
  const flashText = (elId, text) => {
    const el = document.getElementById(elId);
    if (!el) return;
    const old = el.getAttribute("data-val");
    if (old !== null && old !== text) {
      el.style.color = "#39ff14";
      el.style.textShadow = "0 0 10px #39ff14";
      setTimeout(() => { el.style.color = ""; el.style.textShadow = ""; }, 1500);
    }
    el.textContent = text;
    el.setAttribute("data-val", text);
  };

  flashText("brainTabStatLR", Math.max(BRAIN_MIN_LR, Number(stats.learning_rate || 0)).toExponential(2));
  flashText("brainTabStatEntropy", lastEntropy !== null && Number.isFinite(Number(lastEntropy)) ? Number(lastEntropy).toFixed(4) : "—");
  flashText("brainTabStatExplore", `${Math.max(BRAIN_MIN_EPS_PCT, Number(stats.exploration_rate_pct || 0)).toFixed(2)}%`);
  flashText("brainTabStatDiscount", Number(stats.discount_factor || 0.99).toFixed(3));
  flashText("brainTabStatBatch", String(Number(stats.batch_size || 128).toFixed(0)));
  flashText("brainTabStatSteps", Number(stats.global_step_count || 0).toLocaleString());
}

function applyBrainDataPayload(raw) {
  let data = raw;
  if (typeof data === "string") {
    try {
      data = JSON.parse(data);
    } catch (_) {
      return;
    }
  }
  data = normalizeBrainWsPayload(data);
  if (!data || data.__ws === "hb") return;
  if (data.topic !== "brain_stats" && data.topic !== "brain_data") return;
  
  if (data.generated_at) {
    window.__lastEngineTickIso = data.generated_at;
    updateLastScanLabel();
  }
  // Altijd reasoning-box bijwerken op WS-tick (ook bij lege reasoning), anders lijkt de tab "dood"
  // terwijl `tm`/`fw` wél binnenkomen. Gebruik dezelfde wachttext als de API.
  const rtxt = String(data.reasoning || "").trim();
  const baseReason = rtxt || BRAIN_REASONING_WAIT;
  const net = data.training_monitor && data.training_monitor.network_logs ? data.training_monitor.network_logs : {};
  const latestKl = (net.approx_kl || []).slice(-1)[0];
  const latestValueLoss = (net.value_loss || []).slice(-1)[0];
  const reasoningText =
    baseReason +
    (latestKl !== undefined || latestValueLoss !== undefined
      ? `\nNetwork health: approx_kl=${Number(latestKl || 0).toFixed(6)}, value_loss=${Number(latestValueLoss || 0).toFixed(6)}.`
      : "");
  const rb = document.getElementById("brainReasoningBox");
  window.botMetrics.reasoningText = reasoningText;
  if (rb) rb.innerHTML = formatBrainReasoningPanelHtml(reasoningText, { status: data.status });

  paintBrainTabCharts(data.training_monitor, {
    feature_weights: data.feature_weights || {},
    rl_observation: data.rl_observation || {},
    feature_weights_policy: data.feature_weights_policy || {},
    social_buzz: data.social_buzz || {},
    market: String(selectedMarket || selectedChartPair || "BTC-EUR").toUpperCase(),
    calibrating: false,
  });
}

function stopBrainStatsSocket() {
  if (brainStatsReconnectTimer) {
    clearTimeout(brainStatsReconnectTimer);
    brainStatsReconnectTimer = null;
  }
  if (brainStatsSocket) {
    const sock = brainStatsSocket;
    try {
      if (sock.readyState === WebSocket.OPEN) {
        sock.close();
      } else if (sock.readyState === WebSocket.CONNECTING) {
        sock.onopen = () => sock.close();
      }
    } catch (_) {}
    brainStatsSocket = null;
  }
}

function connectBrainStatsSocket() {
  if (brainStatsSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(brainStatsSocket.readyState)) return;
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}/ws/brain-stats`;
  brainStatsSocket = new WebSocket(wsUrl);
  brainStatsSocket.onmessage = (event) => {
    if (document.hidden || activeTab !== "aibrain") {
      latestBrainStatsPayload = event.data;
      return;
    }
    try {
      const data = normalizeBrainWsPayload(event.data);
      if (!data) return;
      if (data.__ws === "hb") {
        try {
          brainStatsSocket.send("hb_ack");
        } catch (_) {}
        return;
      }
      enqueueCockpitWsRender(() => applyBrainDataPayload(data));
    } catch (_) {}
  };
  brainStatsSocket.onclose = () => {
    brainStatsSocket = null;
    if (activeTab === "aibrain") {
      brainStatsReconnectTimer = setTimeout(connectBrainStatsSocket, 5000);
    }
  };
  brainStatsSocket.onerror = () => {
    try {
      brainStatsSocket.close();
    } catch (_) {}
  };
}

async function refreshWhaleRadar() {
  const ul = document.getElementById("whaleRadarList");
  if (!ul) return;
  try {
    const res = await fetch("/api/v1/whale/radar");
    const data = await res.json();
    const moves = Array.isArray(data.moves) ? data.moves.slice(0, 3) : [];
    ul.innerHTML = "";
    if (!moves.length) {
      ul.innerHTML = '<li class="whale-radar-widget__empty">Geen grote whale-moves (≥1M USD) voor Elite-8.</li>';
      return;
    }
    for (const m of moves) {
      const li = document.createElement("li");
      li.className = "whale-radar-widget__item";
      const sym = escapeHtmlText(String(m.symbol || "?"));
      const dir = escapeHtmlText(String(m.direction || "?"));
      const usd = Number(m.usd_notional_est || 0);
      const h = escapeHtmlText(String(m.headline || "").slice(0, 140));
      li.innerHTML = `<strong>${sym}</strong> <span class="whale-radar-widget__dir">${dir}</span> · $${usd.toLocaleString("en-US")} · ${h}`;
      ul.appendChild(li);
    }
  } catch (_e) {
    ul.innerHTML = '<li class="whale-radar-widget__empty">Whale radar niet beschikbaar.</li>';
  }
}

async function refreshHistoryTrades() {
  const body = document.getElementById("cockpitLedgerBody");
  if (!body) return;
  if (window.ModuleLedger && typeof window.ModuleLedger.refresh === "function") {
    try {
      await window.ModuleLedger.refresh();
      const empty = body.querySelector("td.cockpit-ledger-empty");
      const n = empty ? 0 : body.querySelectorAll("tr").length;
      cockpitLedgerStatusParts.ledger = n ? `Ledger: ${n} trade(s)` : "Ledger: 0 trades";
      paintCockpitLedgerStatus();
    } catch (_e) {
      cockpitLedgerStatusParts.ledger = "Ledger: netwerkfout";
      paintCockpitLedgerStatus();
    }
    return;
  }
  try {
    // Expliciet ophalen van ALLE markten, ongeacht de geselecteerde grafiek
    const res = await fetch(
      `/api/v1/trades?limit=${LEDGER_ROUNDTRIP_FETCH_LIMIT}&view=events&market=all`
    );
    const data = await res.json();
    if (!res.ok) {
      cockpitLedgerStatusParts.ledger = `Ledger: API-fout (${res.status})`;
      paintCockpitLedgerStatus();
      return;
    }
    const rows = Array.isArray(data.trades) ? data.trades.slice() : [];
    sortLedgerTradesChronoDesc(rows);
    if (window.ModuleLedger && typeof window.ModuleLedger.renderTable === "function") {
      let riskProfile = null;
      try {
        const ar = await fetch(
          typeof window.buildActivityFetchUrl === "function" ? window.buildActivityFetchUrl() : "/activity",
          window.activityFetchInit || { cache: "no-store", credentials: "same-origin" }
        );
        if (ar.ok) {
          const aj = await ar.json();
          if (aj.risk_profile && typeof aj.risk_profile === "object") riskProfile = aj.risk_profile;
        }
      } catch (_e) {}
      window.ModuleLedger.renderTable({ trades: rows, risk_profile: riskProfile || undefined });
      cockpitLedgerStatusParts.ledger = rows.length
        ? `Ledger: ${rows.length} trade(s)`
        : "Ledger: 0 trades";
      paintCockpitLedgerStatus();
      return;
    }
  } catch (_err) {
    cockpitLedgerStatusParts.ledger = "Ledger: netwerkfout";
    paintCockpitLedgerStatus();
    // Keep existing rows on network error — do not clear
  }
}

async function refreshLedgerTab() {
  if (activeTab !== "ledger") return;
  await refreshHistoryTrades();
  await refreshPerformanceAnalytics();
}

/**
 * Senior Cleanup Fix: Vernietig alle grafieken om z-index 'ghosting' 
 * en canvas overlaps te voorkomen tijdens tab-wissels.
 */
function clearAllCharts() {
  if (window.ChartUtils && typeof window.ChartUtils.clearAllCharts === 'function') {
      window.ChartUtils.clearAllCharts();
  }

  const chartIds = [
      "equityCurveChart", "winLossChart", "sentimentOutcomeChart",
      "brainTabTrainingLossChart", "brainTabRewardErrorChart", "brainTabFeatureChart",
      "brainBenchmarkChart", "brainCorrelationChart", "brainEpisodeChart",
      "brainEntropyChart", "brainLossChart", "brainNewsLagChart",
      "terminalBenchmarkChart", "terminalCorrelationChart"
  ];
  chartIds.forEach(canvasId => {
      const oldCanvas = document.getElementById(canvasId);
      if (oldCanvas && oldCanvas.parentElement) {
        const wrapper = oldCanvas.parentElement;
        const newCanvas = document.createElement("canvas");
        newCanvas.id = canvasId;
        newCanvas.className = oldCanvas.className;
        oldCanvas.remove();
        wrapper.appendChild(newCanvas);
      }
  });

  // Lightweight charts cleanup
  const priceHost = document.getElementById("priceChart");
  if (priceChart && typeof priceChart.remove === 'function') {
    priceChart.remove();
    priceChart = null;
    priceSeries = null;
    predictionOverlaySeries = null;
    markerSeries = null;
    whaleDangerPriceLineHandle = null;
    priceSeriesIsCandle = false;
    if (priceChartResizeObserver) {
      priceChartResizeObserver.disconnect();
      priceChartResizeObserver = null;
    }
  }
  if (priceHost) {
    priceHost.innerHTML = ""; // DOM Cleanup
  }
}

async function refreshSystemLogsSnapshot() {
  if (activeTab !== "hardware") return;
  const toggle = document.getElementById("systemLogAutoRefresh");
  if (toggle && !toggle.checked) return;
  try {
    const res = await fetch("/api/v1/system/logs?limit=200");
    let data = {};
    try {
      data = await res.json();
    } catch {
      data = {};
    }
    let consoleEl = document.getElementById("systemLogConsole");
    if (consoleEl) {
      consoleEl.innerHTML = "";
    }
    if (!res.ok) {
      appendSystemLogLine(
        `[Hardware] /api/v1/system/logs HTTP ${res.status} — controleer portal-logs en permissies op _logs_hub.`
      );
      return;
    }
    const lines = Array.isArray(data.lines) ? data.lines : [];
    for (const line of lines) {
      appendSystemLogLine(line);
    }
    if (!lines.length) {
      appendSystemLogLine("[Hardware] API gaf geen regels terug (lege array).");
    }
  } catch (err) {
    appendSystemLogLine(`[Hardware] Snapshot mislukt: ${String(err)}`);
  }
}

function connectSystemLogsSocket() {
  if (activeTab !== "hardware" && activeTab !== "logs") return;
  if (systemLogsSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(systemLogsSocket.readyState)) return;
  const status = document.getElementById("systemLogStatus");
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}/ws/logs`;
  systemLogsSocket = new WebSocket(wsUrl);

  systemLogsSocket.onopen = () => {
    if (status) {
      status.textContent = "Live";
      status.className = "status-connected genesis-mono-strong";
    }
  };
  systemLogsSocket.onmessage = (event) => {
    const incoming = String(event.data || "");
    if (systemLogsPaused) {
      systemLogsBuffer.push(incoming);
      if (systemLogsBuffer.length > 1000) {
        systemLogsBuffer = systemLogsBuffer.slice(-1000);
      }
      return;
    }
    if (window.TerminalLiveTail && typeof window.TerminalLiveTail.push === "function" && activeTab === "logs") {
      window.TerminalLiveTail.push(incoming);
      return;
    }
    appendSystemLogLine(incoming);
  };
  systemLogsSocket.onclose = () => {
    if (status) {
      status.textContent = "Reconnecting...";
      status.className = "status-disconnected genesis-mono-strong";
    }
    if (activeTab === "hardware" || activeTab === "logs") {
      systemLogsReconnectTimer = setTimeout(connectSystemLogsSocket, 2000);
    }
  };
  systemLogsSocket.onerror = () => {
    try { systemLogsSocket.close(); } catch (_) {}
  };
}

function stopSystemLogsSocket() {
  if (systemLogsReconnectTimer) {
    clearTimeout(systemLogsReconnectTimer);
    systemLogsReconnectTimer = null;
  }
  if (systemLogsSocket) {
    const sock = systemLogsSocket;
    try {
      if (sock.readyState === WebSocket.OPEN) {
        sock.close();
      } else if (sock.readyState === WebSocket.CONNECTING) {
        sock.onopen = () => sock.close();
      }
    } catch (_) {}
    systemLogsSocket = null;
  }
}

/** Brain API: JSON alleen bij 2xx; anders fallback (zelfde gedrag als `module_brain.js`). */
async function brainApiSafeJson(res, fallback) {
  if (!res || !res.ok) return fallback;
  try {
    return await res.json();
  } catch (_) {
    return fallback;
  }
}

async function refreshBrainLab() {
  if (activeTab !== "aibrain") return;
  try {
    const marketFocus = selectedMarket || selectedChartPair || "BTC-EUR";
    const marketParam = encodeURIComponent(marketFocus);
    const [reasoningRes, fiRes, monitorRes, stateRes, lagRes, statsRes, statusRes] = await Promise.all([
      fetch("/api/v1/brain/reasoning"),
      fetch(`/api/v1/brain/feature-importance?market=${marketParam}`),
      fetch("/api/v1/brain/training-monitor"),
      fetch("/api/v1/brain/state-overview"),
      fetch("/api/v1/brain/news-lag"),
      fetch(`/api/v1/stats?symbol=${marketParam}`),
      fetch("/api/v1/status"),
    ]);
    if ([reasoningRes, fiRes, monitorRes, stateRes, lagRes].some((r) => !r || !r.ok)) {
      console.warn("[BrainLab] Een of meerdere brain-API endpoints gaven geen 2xx. Fallbacks worden gebruikt.");
    }
    const reasoningData = await brainApiSafeJson(reasoningRes, { reasoning: BRAIN_REASONING_WAIT });
    const fiData = await brainApiSafeJson(fiRes, {
      market: marketFocus,
      feature_weights: {},
      rl_observation: {},
    });
    const monitorData = await brainApiSafeJson(monitorRes, { stats: {} });
    const stateData = await brainApiSafeJson(stateRes, { state: {}, weight_focus: {} });
    const lagData = await brainApiSafeJson(lagRes, { items: [] });
    const statsData = await brainApiSafeJson(statsRes, {});
    const statusData = await brainApiSafeJson(statusRes, {});

    const hasSeriesData = (arr) => Array.isArray(arr) && arr.some((v) => Number.isFinite(Number(v)));
    const hasMonitorActivity =
      hasSeriesData(monitorData.reward) ||
      hasSeriesData(monitorData.reward_normalized) ||
      hasSeriesData(monitorData.loss) ||
      hasSeriesData(monitorData.episode_length);

    // Brain tab bruikbaar houden bij opstart: gebruik live policy-probs als feature fallback.
    const ap = statsData && typeof statsData.ai_action_probs === "object" ? statsData.ai_action_probs : {};
    const fallbackPolicyWeights = {
      policy_buy: Number(ap.buy_pct || 0) / 100.0,
      policy_hold: Number(ap.hold_pct || 0) / 100.0,
      policy_sell: Number(ap.sell_pct || 0) / 100.0,
    };
    if ((!fiData.feature_weights || !Object.keys(fiData.feature_weights).length) && Object.values(fallbackPolicyWeights).some((v) => v > 0)) {
      fiData.feature_weights = fallbackPolicyWeights;
    }

    const engine = String(statusData.ai_engine || "").toUpperCase() || "THINKING";
    const worker = String(statsData.worker_status || statusData.worker_status || "loading").toLowerCase();
    const cal = document.getElementById("strategyCalibrationStatus");
    if (cal) {
      cal.textContent = hasMonitorActivity
        ? `Live telemetry actief (${marketFocus}) — engine ${engine}`
        : `Wachten op RL telemetry (${marketFocus}) — engine ${engine}, worker ${worker}`;
    }

    paintBrainTabCharts(monitorData, fiData);

    const net = monitorData.network_logs || {};
    const latestKl = (net.approx_kl || []).slice(-1)[0];
    const latestValueLoss = (net.value_loss || []).slice(-1)[0];
    const reasoningText =
      (reasoningData.reasoning || BRAIN_REASONING_WAIT) +
      (latestKl !== undefined || latestValueLoss !== undefined
        ? `\nNetwork health: approx_kl=${Number(latestKl || 0).toFixed(6)}, value_loss=${Number(latestValueLoss || 0).toFixed(6)}.`
        : "") +
      (!hasMonitorActivity
        ? `\nStatus: ai_engine=${engine}, worker=${worker}, last_inference=${String(
            statsData.last_inference_time || statusData.last_inference_time || "—"
          )}.`
        : "");
    const rb = document.getElementById("brainReasoningBox");
    window.botMetrics.reasoningText = reasoningText;
    if (rb) rb.innerHTML = formatBrainReasoningPanelHtml(reasoningText, reasoningData);

    if (reasoningData && reasoningData.generated_at) {
        window.__lastEngineTickIso = reasoningData.generated_at;
        updateLastScanLabel();
    }

    const fw = withAmbientSignalWeights(fiData.feature_weights || {}, fiData.rl_observation || {});
    const featureChartConfig = featureWeightsGroupedBarConfig(fw, fiData.rl_observation || {}, {
      feature_weights_policy: fiData.feature_weights_policy || {},
    });

    const rawRw = monitorData.reward || [];
    const rewards =
      Array.isArray(monitorData.reward_normalized) && monitorData.reward_normalized.length
        ? monitorData.reward_normalized.map((v) => Number(v) || 0)
        : normalizeRewardClient(rawRw);
    const episodeLen = monitorData.episode_length || [];
    const entropy = monitorData.policy_entropy || [];
    const loss = monitorData.loss || [];
    const lagRows = Array.isArray(lagData.items) ? lagData.items : [];
    const lineOpts = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } };
    const rewardMa = rewardMovingAverage(rewards.length ? rewards : [0], REWARD_MA_WINDOW);
    const rewardCfg = {
      type: "line",
      data: {
        labels: (rewards.length ? rewards : [0]).map((_, idx) => idx + 1),
        datasets: [
          {
            label: "Reward (genorm. −1…1)",
            data: rewards.length ? rewards : [0],
            borderColor: "#66ff66",
            backgroundColor: "rgba(102, 255, 102, 0.06)",
            borderWidth: 4,
            pointRadius: 0,
          },
          {
            label: `Reward MA(${REWARD_MA_WINDOW})`,
            data: rewardMa,
            borderColor: BRAIN_NEON_REWARD_MA,
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 0,
          },
        ],
      },
      options: {
        ...lineOpts,
        plugins: { legend: { display: true } },
        scales: {
          x: mergeContrastScale({ ticks: { maxRotation: 0 } }),
          y: mergeContrastScale({ min: -1, max: 1 }),
        },
      },
    };
    if (document.getElementById("brainRewardChart")) {
      if (window.ChartUtils) window.ChartUtils.upsertChart(
        "brainRewardChart",
        rewardCfg
      );
    }
    const lagCfg = {
      type: "line",
      data: {
        labels: lagRows.map((_, idx) => idx + 1),
        datasets: [
          {
            label: "News Lag (sec)",
            data: lagRows.map((r) => Number(r.news_lag_sec || 0)),
            borderColor: "#fff176",
            borderWidth: 4,
            pointRadius: 0,
          },
        ],
      },
      options: lineOpts,
    };
    if (document.getElementById("brainNewsLagChart")) {
      if (window.ChartUtils) window.ChartUtils.upsertChart(
        "brainNewsLagChart",
        lagCfg
      );
    }
    const epCfg = {
      type: "line",
      data: {
        labels: episodeLen.map((_, idx) => idx + 1),
        datasets: [
          {
            label: "Episode Length",
            data: episodeLen,
            borderColor: "#66e8ff",
            borderWidth: 4,
            pointRadius: 0,
          },
        ],
      },
      options: lineOpts,
    };
    if (document.getElementById("brainEpisodeChart")) {
      if (window.ChartUtils) window.ChartUtils.upsertChart(
        "brainEpisodeChart",
        epCfg
      );
    }
    const entCfg = {
      type: "line",
      data: {
        labels: entropy.map((_, idx) => idx + 1),
        datasets: [
          {
            label: "Policy Entropy",
            data: entropy,
            borderColor: "#ff66ff",
            borderWidth: 4,
            pointRadius: 0,
          },
        ],
      },
      options: lineOpts,
    };
    if (document.getElementById("brainEntropyChart")) {
      if (window.ChartUtils) window.ChartUtils.upsertChart(
        "brainEntropyChart",
        entCfg
      );
    }
    const vl = net.value_loss || loss;
    const lossCfg = {
      type: "line",
      data: {
        labels: vl.map((_, idx) => idx + 1),
        datasets: [
          {
            label: "Value Loss",
            data: vl,
            borderColor: "#ffb84d",
            borderWidth: 4,
            pointRadius: 0,
          },
        ],
      },
      options: lineOpts,
    };
    if (document.getElementById("brainLossChart")) {
      if (window.ChartUtils) window.ChartUtils.upsertChart("brainLossChart", lossCfg);
    }

    const stats = monitorData.stats || {};
    window.botMetrics.trainingStats = {
      learningRate: Number(stats.learning_rate || 0).toExponential(2),
      steps: Number(stats.global_step_count || 0).toLocaleString(),
      exploration: `${Number(stats.exploration_rate_pct || 0).toFixed(2)}%`,
    };

    const benchmark = monitorData.benchmark || {};
    const benchCfg = {
      type: "bar",
      data: {
        labels: ["RL Agent", "Buy & Hold", "Alpha"],
        datasets: [
          {
            label: "PnL %",
            data: [
              Number(benchmark.rl_pnl_pct || 0),
              Number(benchmark.buy_hold_pnl_pct || 0),
              Number(benchmark.alpha_pct || 0),
            ],
            backgroundColor: ["#66ff66", "#00f8ff", "#ffff66"],
          },
        ],
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } },
    };
    if (window.ChartUtils) {
        window.ChartUtils.upsertChart("brainBenchmarkChart", benchCfg);
        window.ChartUtils.upsertChart("terminalBenchmarkChart", benchCfg);
    }

    const benchRows =
      `<tr><td>RL Agent</td><td>${Number(benchmark.rl_pnl_pct || 0).toFixed(3)}%</td></tr>` +
      `<tr><td>Buy & Hold</td><td>${Number(benchmark.buy_hold_pnl_pct || 0).toFixed(3)}%</td></tr>` +
      `<tr><td>Alpha</td><td>${Number(benchmark.alpha_pct || 0).toFixed(3)}%</td></tr>`;
    const body = document.getElementById("brainBenchmarkBody");
    if (body) body.innerHTML = benchRows;
    const tBenchBody = document.getElementById("terminalBenchmarkBody");
    if (tBenchBody) tBenchBody.innerHTML = benchRows;

    const corr = monitorData.correlation || {};
    const corrValue = Number(corr.sentiment_price_correlation || 0);
    const newsWeight = Number(corr.news_weight || 0);
    const priceWeight = Number(corr.price_weight || 0);
    window.botMetrics.correlation = {
      sentimentPrice: corrValue,
      newsWeight,
      priceWeight,
    };
    const corrCfg = {
      type: "bar",
      data: {
        labels: ["Sentiment/Price Corr", "News Weight", "Price Weight"],
        datasets: [
          {
            label: "Correlation & Decision Mix",
            data: [corrValue, newsWeight, priceWeight],
            backgroundColor: ["#ff4dff", "#39ff14", "#00f0ff"],
          },
        ],
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } },
    };
    if (window.ChartUtils) {
        window.ChartUtils.upsertChart("brainCorrelationChart", corrCfg);
        window.ChartUtils.upsertChart("terminalCorrelationChart", corrCfg);
    }
    setClassText(".js-corr-value", corrValue.toFixed(4));
    setClassText(".js-news-weight", newsWeight.toFixed(4));
    setClassText(".js-price-weight", priceWeight.toFixed(4));

    const st = stateData.state || {};
    const focus = stateData.weight_focus || {};
    const fg = Number(st.fear_greed_score || 0).toFixed(3);
    const btc = `${Number(st.btc_dominance_pct || 0).toFixed(2)}%`;
    const wh = Number(st.whale_pressure || 0).toFixed(3);
    const macro = Boolean(st.macro_volatility_window) ? "Yes" : "No";
    const rsi = Number(st.rsi_14 || 0).toFixed(2);
    const macdVal = Number(st.macd || 0).toFixed(4);
    const focusLine = `Focus - BTC Dom: ${Number(focus.btc_dominance || 0).toFixed(3)}, Whales: ${Number(focus.whales || 0).toFixed(3)}, Macro: ${Number(focus.macro || 0).toFixed(3)}, RSI: ${Number(focus.rsi || 0).toFixed(3)}`;
    const tBtc = document.getElementById("tickerMetricBtcDom");
    if (tBtc) tBtc.textContent = btc.replace(/\s/g, "") || "—";
    const tWh = document.getElementById("tickerMetricWhale");
    if (tWh) tWh.textContent = wh;
    const bsw = document.getElementById("brainStateWeights");
    if (bsw) bsw.textContent = focusLine;
    window.botMetrics.focusLine = focusLine;
    window.botMetrics.marketState = {
      fearGreed: fg,
      btcDom: btc,
      whale: wh,
      macro,
      rsi,
      macd: macdVal,
    };
  } catch (_err) {
    const fail = "AI Brain data kon niet geladen worden.";
    window.botMetrics.reasoningText = fail;
    const b = document.getElementById("brainReasoningBox");
    if (b) b.textContent = fail;
  }
}

function noirLedgerLineBarOptions(extra = {}) {
  const gridOff = { display: false, color: "transparent" };
  return highContrastChartOptions({
    responsive: true,
    maintainAspectRatio: false,
    ...extra,
    plugins: { legend: { display: false }, ...(extra.plugins || {}) },
    scales: {
      x: {
        ticks: { color: CHART_AXIS_WHITE },
        grid: gridOff,
        border: { color: CHART_AXIS_WHITE, display: true },
      },
      y: {
        ticks: { color: CHART_AXIS_WHITE },
        grid: gridOff,
        border: { color: CHART_AXIS_WHITE, display: true },
      },
      ...(extra.scales || {}),
    },
  });
}

function formatHoldHours(hours) {
  const n = Number(hours);
  if (!Number.isFinite(n) || n < 0) return "—";
  if (n === 0) return "0.0 u";
  if (n < 48) return `${n.toFixed(1)} u`;
  const d = Math.floor(n / 24);
  const r = n - d * 24;
  return `${d}d ${r.toFixed(1)}u`;
}

function renderLedgerPerformanceSummary(data) {
  const wl = data.analytics?.win_loss_ratio || {};
  let ps = data.analytics?.performance_summary;
  if (!ps) {
    const wins = Number(wl.wins || 0);
    const losses = Number(wl.losses || 0);
    const closed = wins + losses;
    const total = Math.max(1, closed);
    ps = {
      total_pnl_eur: Number(data.wallet?.realized_pnl_eur ?? 0),
      total_pnl_pct: Number(data.wallet?.realized_pnl_pct ?? 0),
      win_rate_pct: Number(wl.win_rate_pct ?? (closed ? (wins / total) * 100 : 0)),
      wins,
      losses,
      closed_trades: closed,
      max_win_eur: 0,
      max_loss_eur: 0,
      avg_hold_hours: 0,
    };
  }
  const eurEl = document.getElementById("ledgerPerfPnlEur") || document.getElementById("pnl-total");
  const pctEl = document.getElementById("ledgerPerfPnlPct");
  const wrEl = document.getElementById("ledgerPerfWinRate");
  const clEl = document.getElementById("ledgerPerfClosed") || document.getElementById("trade-count");
  const mwEl = document.getElementById("ledgerPerfMaxWin");
  const mlEl = document.getElementById("ledgerPerfMaxLoss");
  const hdEl = document.getElementById("ledgerPerfHold");
  const pnlEur = Number(ps.total_pnl_eur ?? data.wallet?.realized_pnl_eur ?? 0);
  const pnlPct = Number(ps.total_pnl_pct ?? data.wallet?.realized_pnl_pct ?? 0);
  const sign = pnlEur >= 0 ? "+" : "";
  if (eurEl) eurEl.textContent = `${sign}€${Math.abs(pnlEur).toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  if (pctEl) pctEl.textContent = `(${sign}${pnlPct.toFixed(2)}%)`;
  const wins = Number(ps.wins ?? wl.wins ?? 0);
  const losses = Number(ps.losses ?? wl.losses ?? 0);
  const closed = Number(ps.closed_trades ?? wins + losses);
  const wr = Number(ps.win_rate_pct ?? wl.win_rate_pct ?? 0);
  if (wrEl) wrEl.textContent = `${wr.toFixed(1)}% (${wins}W / ${closed}T)`;
  if (clEl) clEl.textContent = `${closed}`;
  if (mwEl) {
    const v = Number(ps.max_win_eur || 0);
    mwEl.textContent = `+€${v.toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  if (mlEl) {
    const v = Number(ps.max_loss_eur || 0);
    mlEl.textContent = `€${v.toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  if (hdEl) hdEl.textContent = formatHoldHours(ps.avg_hold_hours);
}

async function refreshPerformanceAnalytics() {
  // Tab-Isolatie
  if (activeTab !== "ledger") return;

  if (!document.getElementById("equityCurveChart")) return;
  try {
    const res = await fetch("/api/v1/performance/analytics");
    if (!res.ok) return;
    const data = await res.json();

    const equity = data.equity_curve || [];
    const equityLabels = equity.map((r) => (r.ts || "").slice(11, 19));
    const equityValues = equity.map((r) => Number(r.equity || 0));
    if (window.ChartUtils) {
        window.ChartUtils.upsertChart("equityCurveChart", {
          type: "line",
          data: {
            labels: equityLabels,
            datasets: [
              {
                label: "Equity",
                data: equityValues,
                borderColor: "#39ff14",
                borderWidth: 3,
                pointRadius: 0,
              },
            ],
          },
          options: noirLedgerLineBarOptions(),
        });
    }

    const wl = data.analytics?.win_loss_ratio || {};
    if (window.ChartUtils) {
        window.ChartUtils.upsertChart("winLossChart", {
          type: "doughnut",
          data: {
            labels: ["Wins", "Losses"],
            datasets: [
              {
                data: [Number(wl.wins || 0), Number(wl.losses || 0)],
                backgroundColor: ["#39ff14", "#ff3131"],
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                display: true,
                position: "bottom",
                labels: {
                  color: CHART_AXIS_WHITE,
                  font: { size: 13, weight: "700", family: "'JetBrains Mono', ui-monospace, monospace" },
                },
              },
              tooltip: {
                backgroundColor: "#000000",
                titleColor: CHART_AXIS_WHITE,
                bodyColor: CHART_AXIS_WHITE,
                borderColor: CHART_AXIS_WHITE,
                borderWidth: 1,
              },
            },
          },
        });
    }

    const svo = data.analytics?.sentiment_vs_outcome || [];
    if (window.ChartUtils) {
        window.ChartUtils.upsertChart("sentimentOutcomeChart", {
          type: "bar",
          data: {
            labels: svo.map((r) => r.bucket),
            datasets: [
              {
                label: "Avg PnL EUR",
                data: svo.map((r) => Number(r.avg_pnl_eur || 0)),
                backgroundColor: ["#39ff14", "#f8d04f", "#ff3131"],
              },
            ],
          },
          options: noirLedgerLineBarOptions(),
        });
    }

    renderLedgerPerformanceSummary(data);
    renderTradeHistory(data.recent_actions || []);
  } catch (err) {
    console.warn("[performance/analytics] Fetch failed:", err);
  }
}

function renderTradeHistory(trades) {
  const root = document.getElementById("tradeHistoryList");
  if (!root) return;
  root.innerHTML = "";
  const rows = trades.slice(0, 20);
  if (!rows.length) {
    root.innerHTML = `<div class="trade-list-empty">Nog geen trades.</div>`;
    return;
  }
  let rendered = 0;
  for (const t of rows) {
    const rt = String(t.row_type || "").toUpperCase();
    let action = String(t.action || t.type || "").toUpperCase();
    if (rt === "ROUND_TRIP") action = "CLOSE";
    if (!["BUY", "SELL", "CLOSE"].includes(action)) continue;
    const pnl = Number(t.pnl_eur || 0);
    const qty = Number(t.qty || 0);
    const ep = Number(t.entry_price || t.price || 0);
    const xp = Number(t.exit_price || 0);
    const fees = Number(t.fees_eur != null && t.fees_eur !== "" ? t.fees_eur : t.fee_eur) || 0;
    const stake = action === "BUY" && qty > 0 && ep > 0 ? qty * ep : null;
    let winEur = null;
    if ((action === "SELL" || action === "CLOSE") && rt === "ROUND_TRIP" && qty > 0 && ep > 0 && xp > 0) {
      winEur = xp * qty - ep * qty - fees;
    } else if ((action === "SELL" || action === "CLOSE") && Number.isFinite(pnl)) {
      winEur = pnl;
    }
    let euroLine = "";
    if (stake != null && Number.isFinite(stake)) {
      euroLine += `Werkelijke inleg: <b>${formatEurNlTerminal(stake)}</b> · `;
    }
    if (winEur != null && Number.isFinite(winEur)) {
      euroLine += `Gerealiseerde winst: <b>${formatEurNlTerminal(winEur)}</b> · `;
    }
    const pxStr = Number.isFinite(ep)
      ? ep.toLocaleString("nl-NL", { minimumFractionDigits: 4, maximumFractionDigits: 4 })
      : "—";
    const side = action === "BUY" ? "buy" : "sell";
    const row = document.createElement("div");
    row.className = `trade-list-row trade-list-row--${side}`;
    const tstr = String(t.ts || t.exit_ts_utc || "").slice(11, 19);
    row.innerHTML =
      `<span class="trade-list-dot" aria-hidden="true"></span>` +
      `<div class="trade-list-body">` +
      `<div class="trade-list-line1">${tstr} · ${t.pair || t.market || "-"} · <span class="${action === "BUY" ? "positive" : "negative"}">${action}</span></div>` +
      `<div class="trade-list-line2">${euroLine}Koers ${pxStr} · sent ${Number(t.sentiment_score || 0).toFixed(3)} · PnL <span class="${pnl >= 0 ? "positive" : "negative"}">${formatEurNlTerminal(pnl)}</span></div>` +
      `</div>`;
    root.appendChild(row);
    rendered += 1;
  }
  if (!rendered) {
    root.innerHTML = `<div class="trade-list-empty">Nog geen trades.</div>`;
  }
}

async function refreshTradesTable() {
  try {
    const res = await fetch(`/api/v1/trades?limit=${SIDEBAR_TRADES_FETCH_LIMIT}&view=events&market=all`);
    const data = await res.json();
    if (!res.ok) return;
    const rows = (data.trades || []).slice(0, 20).map((t) => {
      const rt = String(t.row_type || "").toUpperCase();
      const pair = String(t.pair || t.market || "").toUpperCase();
      let action = String(t.action || t.type || "").toUpperCase();
      if (rt === "ROUND_TRIP") action = "CLOSE";
      return {
        ts: t.exit_ts_utc || t.entry_ts_utc,
        market: pair,
        pair,
        row_type: rt,
        action,
        entry_price: t.entry_price,
        exit_price: t.exit_price,
        qty: t.qty,
        fees_eur: t.fees_eur != null && t.fees_eur !== "" ? t.fees_eur : t.fee_eur,
        sentiment_score: t.sentiment_score,
        pnl_eur: t.pnl_eur,
      };
    });
    renderTradeHistory(rows);
  } catch (_err) {
    // keep prior table on transient fetch errors
  }
}

async function postBotAction(path) {
  try {
    await fetch(path, { method: "POST" });
    await refreshBotStatus();
    await refreshStorageHealth();
    await refreshActivity();
  } catch (err) {
    console.warn(`[bot_action] ${path} failed:`, err);
  }
}

async function runPaperTrade() {
  const pair = String(selectedMarket || document.getElementById("marketSelect")?.value || "BTC-EUR").toUpperCase();
  const btn = document.getElementById("paperBtn");
  const originalText = btn ? btn.textContent : "Paper";
  
  if (btn) {
      btn.textContent = "Bezig...";
      btn.disabled = true;
  }
  
  try {
    const res = await fetch(`/paper/run?ticker=${encodeURIComponent(pair)}`, { method: "POST" });
    if (!res.ok) {
      console.warn("[paper/run] Server returned error", res.status);
      if (btn) { btn.textContent = "Fout!"; setTimeout(() => { btn.textContent = originalText; btn.disabled = false; }, 2000); }
      return;
    }
    const data = await res.json();
    const pEl = document.getElementById("prediction");
    if (pEl) {
        if (data.status === "command_sent") {
            pEl.innerHTML = `<span style="color:#39ff14">${data.message}</span>`;
        } else {
            pEl.textContent = JSON.stringify(data, null, 2);
        }
    }
    
    await refreshSentiment();
    await refreshNewsInsights();
    await refreshActivity();
    await refreshChart();
    await refreshHistoryTrades();
    await refreshTradesTable();
    await refreshPerformanceAnalytics();
    await refreshBrainLab();
    
    if (btn) {
        btn.textContent = "Succes!";
        setTimeout(() => { btn.textContent = originalText; btn.disabled = false; }, 2000);
    }
  } catch (err) {
    console.warn("[paper/run] Fetch failed:", err);
    const pEl = document.getElementById("prediction");
    if (pEl) pEl.textContent = "Fout bij uitvoeren paper trade (Network error).";
    if (btn) {
        btn.textContent = "Fout!";
        setTimeout(() => { btn.textContent = originalText; btn.disabled = false; }, 2000);
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  updateTerminalHealthFromStats(null);
  document.addEventListener("ai-trading-stats", (ev) => {
    const d = ev && ev.detail;
    if (!d || typeof d !== "object") return;
    updateTerminalHealthFromStats(d);
    renderBvThoughtFeed(d);
  });

  document.getElementById("paperBtn")?.addEventListener("click", runPaperTrade);
  document.getElementById("marketSelect")?.addEventListener("change", selectMarketFromDropdown);
  document.getElementById("refreshMarketsBtn")?.addEventListener("click", refreshMarkets);
  document.getElementById("stopBotBtn")?.addEventListener("click", () => postBotAction("/bot/panic"));
  document.getElementById("newsModalClose")?.addEventListener("click", closeNewsModal);
  document.getElementById("btn-terminal")?.addEventListener("click", () => switchTab("terminal"));
  document.getElementById("btn-aibrain")?.addEventListener("click", () => switchTab("aibrain"));
  document.getElementById("btn-ledger")?.addEventListener("click", () => switchTab("ledger"));
  document.getElementById("btn-logs")?.addEventListener("click", () => switchTab("logs"));
  document.getElementById("rlInferenceGreedyToggle")?.addEventListener("change", async (e) => {
    const el = e.target;
    const greedy = !!(el && el.checked);
    try {
      const res = await fetch("/api/v1/rl-inference-mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ greedy }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      console.warn("[rl-inference-mode]", err);
      if (el) el.checked = !greedy;
    }
  });
  document.getElementById("btnSystemLogsFromLedger")?.addEventListener("click", (e) => {
    e.preventDefault();
    const top = document.getElementById("btn-logs");
    if (top) top.click();
    else switchTab("logs");
  });
  document.getElementById("systemLogPauseBtn")?.addEventListener("click", toggleSystemLogPause);
  document.getElementById("systemLogClearBtn")?.addEventListener("click", clearSystemLogConsole);
  document.getElementById("systemLogMuteBtn")?.addEventListener("click", toggleSystemLogMute);
  document.getElementById("newsModal")?.addEventListener("click", (event) => {
    if (event.target.id === "newsModal") closeNewsModal();
  });
});

window.setChartInterval = async function setChartInterval(next) {
  const sel = normalizeChartIntervalSelection(next);
  chartIntervalSelection = sel;
  chartInterval = chartIntervalFromSelection(sel);
  try {
    window.localStorage.setItem(CHART_INTERVAL_STORAGE_KEY, sel);
  } catch (_e) {
    /* ignore */
  }
  updateChartIntervalUi(sel);
  const foot = document.getElementById("chartTimeFootnote");
  if (foot) foot.textContent = `Timeframes: AI | 1m | 5m | 15m | 60m | active=${sel.toUpperCase()}`;
  await updateChart(selectedChartPair || selectedMarket);
};

/**
 * Sequenced startup loader — voorkomt 'Network Storm' / ERR_EMPTY_RESPONSE na FinBERT/PPO-load.
 * Fase 1: health + markten (+ geselecteerde markt) → korte pauze.
 * Fase 2: activity + bot status → korte pauze.
 * Fase 3: zware endpoints (trades/ledger + performance analytics).
 * Fase 4: chart + nieuws + brain + WebSockets.
 */
window.runSequencedStartup = async function runSequencedStartup() {
  updateSystemPauseButton();
  updateSystemMuteButton();
  cockpitLedgerStatusParts.markets = "Markten: ophalen…";
  cockpitLedgerStatusParts.ledger = "Ledger: ophalen…";
  paintCockpitLedgerStatus();

  await Promise.allSettled([refreshHealthMode(), refreshSelectedMarket(), refreshMarkets()]);
  updateChartIntervalUi(chartIntervalSelection);
  const foot = document.getElementById("chartTimeFootnote");
  if (foot) foot.textContent = `Timeframes: AI | 1m | 5m | 15m | 60m | active=${chartIntervalSelection.toUpperCase()}`;
  selectedChartPair = selectedMarket;
  syncHeaderMarketChip();

  await sleepMs(300);

  await Promise.allSettled([refreshActivity(), refreshBotStatus()]);
  startCockpitHeartbeat();
  initHintPortals();
  bindSniperPanelControls();

  await Promise.allSettled([
    refreshBalanceCheck(),
    refreshSentiment(),
    refreshStorageHealth(),
  ]);

  connectSystemStatsSocket();
  connectBrainStatsSocket();
  /* Nieuws feed laden zonder zware grafieken te initieren */
  await Promise.allSettled([refreshNewsInsights()]);

  /* Server kan markten net na engine-boot vullen; één retry voorkomt lege cockpit bij snelle page-load. */
  if (!lastActiveMarketsRows.length) {
    await sleepMs(4000);
    await refreshMarkets();
  }

  // Start uitsluitend de data-fetches en chart-renders van de actieve tab.
  // Dit voorkomt dat er op de achtergrond 4-voudige grafiek-initialisaties in de DOM plaatsvinden.
  switchTab("terminal");
  void updatePredictionChart(selectedMarket || "BTC-EUR");
}

document.addEventListener("DOMContentLoaded", () => {
  updateChartIntervalUi(chartIntervalSelection);
  const foot = document.getElementById("chartTimeFootnote");
  if (foot) foot.textContent = `Timeframes: AI | 1m | 5m | 15m | 60m | active=${chartIntervalSelection.toUpperCase()}`;
  (async () => {
    try {
      await runSequencedStartup();
    } catch (err) {
      console.error("[startup] sequenced loader failed:", err);
    }
  })();
});

/**
 * Staggered intervals — Geïsoleerd per actieve tab om achtergrond-renders te stoppen.
 */
function scheduleStaggered(fn, periodMs, initialDelayMs = 0) {
  setTimeout(() => {
    try {
      fn();
    } catch (_) {}
    setInterval(() => {
      try {
        fn();
      } catch (_) {}
    }, Math.max(1000, Number(periodMs) || 1000));
  }, Math.max(0, Number(initialDelayMs) || 0));
}

scheduleStaggered(() => {
  if (activeTab !== "terminal") return;
  refreshHealthMode();
  refreshSentiment();
  refreshBotStatus();
  refreshStorageHealth();
}, 10000, 4000);

scheduleStaggered(() => {
  if (activeTab !== "terminal") return;
  const wsQuietMs = Date.now() - lastTradingWsActivityAtMs;
  if (wsQuietMs > 20000 || !lastTradingWsActivityAtMs) {
    void refreshActivity();
  }
}, 10000, 7000);

scheduleStaggered(() => {
  if (activeTab !== "terminal") return;
  refreshChart();
  const sel = document.getElementById("marketSelect")?.value || selectedMarket || selectedChartPair || "BTC-EUR";
  void updatePredictionChart(sel);
}, 30000, 8000);

scheduleStaggered(() => {
  if (activeTab !== "terminal" && activeTab !== "ledger") return;
  refreshTradesTable();
}, 8000, 5000);

scheduleStaggered(() => {
  if (activeTab !== "ledger") return;
  refreshHistoryTrades();
  refreshPerformanceAnalytics();
}, 20000, 9000);

scheduleStaggered(() => {
  if (activeTab !== "terminal") return;
  refreshNewsInsights();
}, 90000, 15000);

scheduleStaggered(() => {
  if (activeTab !== "aibrain") return;
  refreshBrainLab();
}, 15000, 6000);

/** Eerste /markets/active kan falen tijdens worker-boot; periodiek opnieuw proberen. */
scheduleStaggered(() => {
  if (activeTab !== "terminal") return;
  void refreshMarkets();
}, 60000, 20000);

scheduleStaggered(() => {
  if (activeTab !== "hardware") return;
  refreshSystemLogsSnapshot();
}, 6000, 3000);

// === KERN: HARDE TAB ISOLATIE & EXCLUSIEVE ENDPOINTS ===

window.switchTab = function switchTab(tabName) {
  activeTab = tabName;

  // 1. Verberg alle tabs en reset knoppen
  const allTabs = ["terminal", "aibrain", "ledger", "hardware", "logs"];
  allTabs.forEach(name => {
    const el = document.getElementById(`tab-${name}`);
    if (el) {
      el.classList.add("hidden");
      el.style.display = "none";
      el.setAttribute("aria-hidden", "true");
    }
    const btn = document.getElementById(`btn-${name}`);
    if (btn) btn.classList.remove("active");
  });

  // 2. Toon actieve tab
  const activeEl = document.getElementById(`tab-${tabName}`);
  if (activeEl) {
    activeEl.classList.remove("hidden");
    activeEl.style.display = "flex";
    activeEl.setAttribute("aria-hidden", "false");
  }
  const activeBtn = document.getElementById(`btn-${tabName}`);
  if (activeBtn) activeBtn.classList.add("active");

  // Header controls toggle
  const headerTrading = document.getElementById("headerTradingActions");
  const headerSystem = document.getElementById("headerSystemActions");
  if (headerTrading && headerSystem) {
    if (tabName === "hardware") {
      headerTrading.classList.add("hidden");
      headerSystem.classList.remove("hidden");
    } else {
      headerTrading.classList.remove("hidden");
      headerSystem.classList.add("hidden");
    }
  }

  // 3. Cleanup: Stop alle lopende processen van andere tabs
  clearAllCharts(); // voorkomt ghosting tussen tabs

  // Stop WebSockets op basis van tab (strikte data-scheiding)
  if (tabName !== "hardware" && tabName !== "logs") stopSystemLogsSocket();
  if (tabName !== "aibrain") stopBrainStatsSocket();

  // 4. Start exclusieve endpoints per view direct (polling wordt door scheduleStaggered afgehandeld)
  if (tabName === "terminal") {
    connectBitvavoPriceStream();
    connectTradingUpdatesSocket();
    refreshActivity();
    refreshNewsInsights();
    setTimeout(() => updateChart(selectedChartPair || selectedMarket), 100);
  }
  else if (tabName === "aibrain") {
    // Render direct de laatste gebufferde snapshot om wachttijd te voorkomen
    if (latestBrainStatsPayload) {
      const data = normalizeBrainWsPayload(latestBrainStatsPayload);
      if (data) enqueueCockpitWsRender(() => applyBrainDataPayload(data));
      latestBrainStatsPayload = null; // Consumeer
    }
    connectBrainStatsSocket();
    // refreshBrainLab is async; validator leest DOM ~1s na tab-klik — reasoning eerst, dan volledige lab-refresh
    void (async () => {
      try {
        const early = await fetch("/api/v1/brain/reasoning");
        if (activeTab !== "aibrain") return;
        const j = await brainApiSafeJson(early, null);
        if (j && j.generated_at) {
          window.__lastEngineTickIso = j.generated_at;
          updateLastScanLabel();
        }
      } catch (_) {}
      if (activeTab === "aibrain") await refreshBrainLab();
    })();
  }
  else if (tabName === "ledger") {
    refreshHistoryTrades();
    refreshPerformanceAnalytics();
  }
  else if (tabName === "hardware") {
    connectSystemStatsSocket(); // Nodig voor ring meters in de header
    connectSystemLogsSocket();
    void refreshSystemLogsSnapshot();
    // Layout kan net na display:flex nog geen hoogte hebben; snapshot opnieuw vullen.
    window.setTimeout(() => void refreshSystemLogsSnapshot(), 120);
    window.setTimeout(() => void refreshSystemLogsSnapshot(), 500);
  } else if (tabName === "logs") {
    connectSystemLogsSocket();
  }

  initHintPortals();
}
