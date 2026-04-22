/*
  Bestand: app/static/js/terminal.js
  Relatief pad: ./app/static/js/terminal.js
  Functie: Clientlogica voor immersive trading terminal met chart markers en live inzichten.
*/

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

/** Kleine delay-helper voor sequenced loader: voorkomt network-storm bij startup. */
function sleepMs(ms) {
  return new Promise((resolve) => setTimeout(resolve, Math.max(0, Number(ms) || 0)));
}
let lastBufferedPrice = null;
let priceAtLastSecondTick = null;
let cockpitHeartbeatTimer = null;
let wsRef = null;
let priceChart = null;
let priceSeries = null;
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
/** Laatste Redis/activity-payload (ms); WS kan open staan zonder berichten — dan blijft `/activity` nodig. */
let lastTradingWsActivityAtMs = 0;
let systemLogsReconnectTimer = null;
let systemStatsSocket = null;
let systemStatsReconnectTimer = null;
let brainTabTrainingLossChart = null;
let brainTabRewardChart = null;
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
let markerRenderState = { thresholdPx: 34, hideText: false, sizeMode: "normal" };

/** Donkere gridlijnen zodat witte assen en neon datasets contrast houden */
const CHART_GRID_DARK = "#222222";
const CHART_AXIS_WHITE = "#FFFFFF";

const HINT_PORTAL_ID = "brainHintPortal";
let chartInterval = "5m";
let headlessMode = false;

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
  const maxW = Math.min(320, vw - 16);
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
  el.className = `cockpit-mode mode-toggle mode-banner ${live ? "mode-live" : "mode-paper"}`;
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

function formatBrainReasoningHtml(raw) {
  let t = escapeHtmlText(String(raw || ""));
  t = t.replace(/(Besluit:\s*(?:HOLD|BUY|SELL)\.)/gi, '<span class="cockpit-besluit-callout">$1</span>');
  t = t.replace(/\n/g, "<br>");
  return t;
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
  let text = String(line || "");
  const fmt = new Intl.DateTimeFormat("nl-NL", {
    timeZone: "Europe/Amsterdam",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });

  // ISO UTC timestamps: 2026-04-20T15:42:31Z (or with milliseconds)
  text = text.replace(
    /\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\b/g,
    (raw) => {
      const d = new Date(raw);
      if (!Number.isFinite(d.getTime())) return raw;
      return fmt.format(d).replace(",", "");
    }
  );

  // Naive timestamps often emitted in UTC by backend loggers: 2026-04-20 15:42:31
  text = text.replace(/\b(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b/g, (raw) => {
    const asUtc = raw.replace(" ", "T") + "Z";
    const d = new Date(asUtc);
    if (!Number.isFinite(d.getTime())) return raw;
    return fmt.format(d).replace(",", "");
  });

  return text;
}

function shouldAutoScroll(consoleEl) {
  const threshold = 40;
  return consoleEl.scrollHeight - (consoleEl.scrollTop + consoleEl.clientHeight) < threshold;
}

function appendSystemLogLine(line) {
  if (systemLogsMuted) return;
  const consoleEl = document.getElementById("systemLogConsole");
  if (!consoleEl) return;
  const autoscroll = shouldAutoScroll(consoleEl);
  const div = document.createElement("div");
  div.className = `system-log-line ${classifyLogLine(line)}`;
  div.textContent = formatSystemLogLineForDisplay(line);
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
  if (!consoleEl) return;
  consoleEl.innerHTML = "";
  systemLogsBuffer = [];
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
    };
  }
  return d;
}

function flushHeaderPriceTick() {
  const lp = document.getElementById("livePrice");
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
  if (wsRef) {
    try {
      wsRef.close();
    } catch (_) {}
  }
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
}

async function refreshSelectedMarket() {
  try {
    const res = await fetch("/markets/selected");
    if (!res.ok) return;
    const data = await res.json();
    selectedMarket = data.selected_market || "BTC-EUR";
    syncHeaderMarketChip();
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
    rows.sort((a, b) => {
      const pa = a.is_pillar === true ? 0 : 1;
      const pb = b.is_pillar === true ? 0 : 1;
      if (pa !== pb) return pa - pb;
      return String(a.market || "").localeCompare(String(b.market || ""));
    });
    for (const m of rows) {
      const opt = document.createElement("option");
      opt.value = m.market;
      const star = m.is_pillar === true ? "* " : "";
      const q = Number(m.quality_score || 0);
      opt.textContent = `${star}${m.market} | Q:${q}/3 | Vol24h: ${m.volume_quote_24h}`;
      if (m.selection_reason) opt.title = String(m.selection_reason);
      if (m.market === selectedMarket) opt.selected = true;
      select.appendChild(opt);
    }
    void refreshActivity();
    renderScannerTickerBar(rows);
    syncHeaderMarketChip();
    if (wsRef) connectBitvavoPriceStream();
  } catch (err) {
    console.warn("[markets/active]", err);
    cockpitLedgerStatusParts.markets = "Markten: netwerkfout";
    paintCockpitLedgerStatus();
  }
}

async function switchEliteMarket(market, marketsSnapshot) {
  const m = String(market || "").toUpperCase();
  if (!m) return;
  selectedMarket = m;
  selectedChartPair = m;
  const sel = document.getElementById("marketSelect");
  if (sel) sel.value = m;

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
  await Promise.all([updateChart(m), refreshBalanceCheck(), refreshBrainLab(), refreshNewsInsights()]);
  await refreshActivity();
  const snap = marketsSnapshot && marketsSnapshot.length ? marketsSnapshot : lastActiveMarketsRows;
  if (snap && snap.length) renderScannerTickerBar(snap);
}

// Senior Fix: Expose de functie globaal als updateMarket voor universele DOM controls.
window.updateMarket = switchEliteMarket;

function renderElite8AiStatusBar(signals) {
  const root = document.getElementById("elite8AiStatusBar");
  if (!root) return;
  
  const list = Array.isArray(signals) ? signals : [];
  
  // Voorkom DOM thrashing door te diffen
  const sigHash = JSON.stringify(list);
  if (root.dataset.lastHash === sigHash) return;
  root.dataset.lastHash = sigHash;
  
  root.innerHTML = "";
  
  if (!list.length) {
    root.innerHTML = `<span class="elite8-ai-status-bar__empty">Elite-8 AI-status wordt geladen…</span>`;
    return;
  }
  for (const s of list) {
    const mk = String(s.market || "").toUpperCase();
    const base = String(s.base || (mk.includes("-") ? mk.split("-")[0] : mk) || "?").toUpperCase();
    const st = String(s.state || "neutral");
    const action = String(s.action || "").toUpperCase() || "…";
    const conf = Number(s.confidence || 0);
    const inPos = Boolean(s.in_position);
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = `elite8-ai-pill elite8-ai-pill--${st}${inPos ? " elite8-ai-pill--in-position" : ""}${
      mk === selectedMarket ? " is-active" : ""
    }`;
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

function renderScannerTickerBar(markets) {
  const root = document.getElementById("scannerTickerBar");
  if (!root) return;
  
  const rows = Array.isArray(markets) ? markets.slice(0, 8) : [];
  
  // Voorkom DOM thrashing door te diffen op markeringsdata
  const hash = JSON.stringify(rows);
  if (root.dataset.lastHash === hash) return;
  root.dataset.lastHash = hash;
  
  root.innerHTML = "";
  
  const held = window.__marketsInPosition instanceof Set ? window.__marketsInPosition : new Set();
  for (const m of rows) {
    const market = String(m.market || "-");
    const mku = market.toUpperCase();
    const inPos = held.has(mku);
    const pct = Number(m.price_change_pct_24h || 0);
    const el = document.createElement("button");
    el.type = "button";
    el.className = `scanner-ticker-badge ${pct >= 0 ? "is-up" : "is-down"} ${market === selectedMarket ? "is-active" : ""}${inPos ? " scanner-ticker-badge--in-position" : ""}`;
    el.style.setProperty("--coin-accent", baseAccentForMarket(mku));
    const star = m.is_pillar === true ? "* " : "";
    el.textContent = `${star}${market} ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%${inPos ? " ·POS" : ""}`;
    const baseTitle = m.selection_reason ? String(m.selection_reason) : "";
    el.title = (baseTitle ? baseTitle + " — " : "") + (inPos ? "IN POSITIE (paper wallet)" : "Geen paper-positie");
    el.addEventListener("click", async () => {
      await switchEliteMarket(market, markets);
    });
    root.appendChild(el);
  }
}

async function selectMarketFromDropdown() {
  const market = document.getElementById("marketSelect").value;
  await switchEliteMarket(market, lastActiveMarketsRows);
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
    window.botMetrics.sentiment = { score, confidence, barPct: pct };
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
  const ms = new Date(dateText).getTime();
  if (!Number.isFinite(ms) || ms <= 0) return null;
  return Math.floor(ms / 1000);
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
  const hostWidth = Math.max(320, Number(host.clientWidth || 0));
  const hostHeight = Math.max(260, Number(host.clientHeight || 0));

  if (!priceChart) {
    priceChart = LightweightCharts.createChart(host, {
      autoSize: true,
      layout: { background: { color: "#000000" }, textColor: "#888888" },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.2)" },
        horzLines: { color: "rgba(255, 255, 255, 0.2)" },
      },
      width: hostWidth,
      height: hostHeight,
      rightPriceScale: {
        borderColor: "#FFFFFF",
        textColor: "#FFFFFF",
        scaleMargins: { top: 0.08, bottom: 0.12 },
      },
      timeScale: {
        borderColor: "#FFFFFF",
        rightOffset: 20,
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time) => {
          try {
            const dt = new Date(Number(time) * 1000);
            const hh = String(dt.getHours()).padStart(2, "0");
            const mm = String(dt.getMinutes()).padStart(2, "0");
            return `${hh}:${mm}`;
          } catch (_err) {
            return "";
          }
        },
        textColor: "#888888",
      },
      localization: { locale: "nl-NL" },
    });
    priceChart.timeScale().subscribeVisibleLogicalRangeChange?.(() => {
      renderAdaptiveMarkers();
    });
    window.addEventListener("resize", () => {
      resizePriceChart(host);
    });
  }

  if (!priceChartResizeObserver && typeof ResizeObserver !== "undefined") {
    priceChartResizeObserver = new ResizeObserver(() => {
      resizePriceChart(host);
    });
    priceChartResizeObserver.observe(host);
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
}

function resizePriceChart(host) {
  if (!priceChart || !host) return;
  const w = Math.max(320, Number(host.clientWidth || host.offsetWidth || 0));
  const h = Math.max(500, Number(host.clientHeight || host.offsetHeight || 0));
  priceChart.applyOptions({ width: w, height: h });
}

function buildOrUpdateChart(labels, prices, markers, whaleDangerZone = null) {
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
  const lineData = labels
    .map((d, i) => ({ time: toEpochSeconds(d), value: Number(prices[i]) }))
    .filter((p) => p.time !== null && Number.isFinite(p.value))
    .sort((a, b) => Number(a.time) - Number(b.time));
  if (!lineData.length) {
    const skelEmpty = document.getElementById("priceChartSkeleton");
    if (skelEmpty) skelEmpty.classList.remove("is-visible");
    return;
  }
  rawChartLineData = lineData;
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
  } else {
    priceSeries.setData(lineData);
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
  priceChart.timeScale().fitContent();
  priceChart.applyOptions({
    timeScale: {
      rightOffset: 20,
      timeVisible: true,
      secondsVisible: false,
    },
  });
}

function setPanelCollapsed(panelEl, collapsed) {
  if (!panelEl) return;
  panelEl.classList.toggle("is-collapsed", Boolean(collapsed));
}

function toggleChartFullscreen() {
  document.body.classList.toggle("sniper-fullscreen");
  const host = document.getElementById("priceChart");
  if (host) resizePriceChart(host);
}

function bindSniperPanelControls() {
  const tickerBtn = document.getElementById("toggleTickerBtn");
  const ledgerBtn = document.getElementById("toggleLedgerBtn");
  const headlessBtn = document.getElementById("toggleHeadlessBtn");
  const fsBtn = document.getElementById("toggleChartFullscreenBtn");
  const tickerPanel = document.getElementById("intelligenceTickerPanel");
  const ledgerPanel = document.getElementById("liveLedgerPanel");
  tickerBtn?.addEventListener("click", () => {
    const next = !tickerPanel?.classList.contains("is-collapsed");
    setPanelCollapsed(tickerPanel, next);
    const host = document.getElementById("priceChart");
    if (host) resizePriceChart(host);
  });
  ledgerBtn?.addEventListener("click", () => {
    const next = !ledgerPanel?.classList.contains("is-collapsed");
    setPanelCollapsed(ledgerPanel, next);
    const host = document.getElementById("priceChart");
    if (host) resizePriceChart(host);
  });
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
  selectedChartPair = String(newPair || selectedMarket || "BTC-EUR").toUpperCase();
  const predictionEl = document.getElementById("prediction");
  const skel = document.getElementById("priceChartSkeleton");
  if (skel) skel.classList.add("is-visible");
  if (predictionEl) {
    predictionEl.textContent = `Laden van ${selectedChartPair} data...`;
  }
  try {
    // Senior Fix: Forceer Refresh - Grafiek instantie verwijderen en container volledig leegmaken
    const host = document.getElementById("priceChart");
    if (host) {
      if (priceChart && typeof priceChart.remove === 'function') {
        priceChart.remove();
      }
      priceChart = null;
      priceSeries = null;
      markerSeries = null;
      whaleDangerPriceLineHandle = null;
      priceSeriesIsCandle = false;
      host.innerHTML = "";
    }
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
      data.whale_danger_zone || null
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
    const score = Number(item.finbert_score ?? 0);
    const impactClass = score > 0.2 ? "impact-positive" : (score < -0.2 ? "impact-negative" : "impact-neutral");
    const neonClass = Math.abs(score) >= 0.65 ? (score >= 0 ? "impact-neon-positive" : "impact-neon-negative") : "";
    const finLabel = item.finbert_label || "neutral";
    const div = document.createElement("div");
    div.className = `news-item ${item.is_urgent ? "news-item-urgent" : ""}`;
    div.innerHTML =
      `<div class="news-headline-line"><strong class="news-headline">${item.headline || "-"}</strong> ${item.is_urgent ? '<span class="urgent-chip">BREAKING</span>' : ""}</div>` +
      `<div class="muted">${item.source || "Unknown source"} | ${formatNewsTimeAmsterdam(item.ts)}</div>` +
      `<div class="impact-row">` +
      `${sourceBadge(item.source_icon, item.source)}` +
      `<span class="impact-left tag ${coinBadgeClass(item.ticker_tag)}">${item.ticker_tag || "-"}</span>` +
      `<span class="impact-score ${impactClass} ${neonClass}">Impact ${score.toFixed(3)}</span>` +
      `</div>` +
      (withInsights
        ? `<div class="muted">FinBERT: <span class="${sentimentTagClass(finLabel)}">${finLabel}</span> (${Number(item.finbert_confidence || 0).toFixed(3)}) | Impacts ticker: ${Boolean(item.impacts_ticker)}</div>` +
          `<div class="muted">Social Volume: ${Number(item.social_volume || 0).toFixed(2)}</div>` +
          `<div class="muted">${item.explanation || "No AI explanation available."}</div>`
        : "");
    div.addEventListener("click", () => openNewsModal(item));
    root.appendChild(div);
  }
}

async function refreshNewsInsights() {
  const [newsRes, actRes] = await Promise.allSettled([
    fetch("/api/v1/news/ticker?elite_mix=1"),
    fetch("/activity"),
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
    source: i.source,
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
  renderNewsStream(window.botMetrics.domIds.terminalNewsStream, mapped);
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
  renderTickerTrack(window.botMetrics.domIds.tickerTrack, items);
}

function applyActivityResponse(data) {
  if (!data || typeof data !== "object") return;
  window.__lastEngineTickIso =
    data.last_engine_tick_utc ||
    (data.last_prediction && data.last_prediction.generated_at) ||
    null;
  updateLastScanLabel();
  const p = data.paper_portfolio || {};
  const alloc = data.allocation_snapshot || {};
  const heldMk = Array.isArray(alloc.markets_in_position)
    ? alloc.markets_in_position.map((x) => String(x || "").toUpperCase())
    : [];
  window.__marketsInPosition = new Set(heldMk);
  const allocRoot = document.getElementById("executiveAllocationSnapshot");
  if (allocRoot) {
    const sum = String(alloc.summary || "Allocatie: —");
    const lines = Array.isArray(alloc.lines) ? alloc.lines : [];
    const rows = lines.map((r) => {
      const c = String(r.coin || "?").toUpperCase();
      const w = Number(r.weight_pct || 0);
      const wTxt = Number.isFinite(w) ? w.toFixed(1) : "—";
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
  const lastOrder = data.last_order?.order || {};
    setClassText(".js-active-orders", `Actieve orders: ${lastOrder.signal || "-"} ${lastOrder.ticker || ""} ${lastOrder.amount_quote_eur || lastOrder.amount_quote || ""}`.trim());
  const fg = data.fear_greed || {};
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
  const cyc = data.last_order?.cycle_seq;
  if (cyc != null && cyc !== lastLedgerCycleSeq) {
    lastLedgerCycleSeq = cyc;
    const fs = String(data.last_order?.engine_risk?.final_signal || "").toUpperCase();
  }
  const rp = data.risk_profile;
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
    Array.isArray(data.elite_ai_signals) &&
      data.elite_ai_signals.some((s) => s && (s.state === "panic" || s.whale_danger === true))
  );
  if (Array.isArray(data.elite_ai_signals)) renderElite8AiStatusBar(data.elite_ai_signals);
  void refreshWhaleRadar();
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
    const res = await fetch("/activity");
    const text = await res.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch (_parseErr) {
      executiveSnapshotIfStillLoading("Kon activity niet lezen (ongeldige JSON).");
      return;
    }
    if (!res.ok) {
      executiveSnapshotIfStillLoading(`Activity niet beschikbaar (HTTP ${res.status}).`);
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
  const data = {
    last_engine_tick_utc: raw.last_engine_tick_utc || null,
    last_prediction: raw.last_prediction || null,
    paper_portfolio: raw.paper_portfolio || {},
    last_order: raw.last_order || {},
    fear_greed: raw.fear_greed || {},
    risk_profile: raw.risk_profile || null,
    elite_ai_signals: raw.elite_ai_signals,
    allocation_snapshot: raw.allocation_snapshot || {},
  };
  if (!data.last_engine_tick_utc && data.last_prediction && data.last_prediction.generated_at) {
    data.last_engine_tick_utc = data.last_prediction.generated_at;
  }
  applyActivityResponse(data);
  lastTradingWsActivityAtMs = Date.now();
  tradingUpdatesLive = true;
  const p = data.paper_portfolio || {};
  const mk = String(selectedMarket || "BTC-EUR").toUpperCase();
  const lpm = p.last_prices_by_market;
  const lp = lpm && typeof lpm === "object" ? lpm[mk] : null;
  if (lp != null && lp !== "") noteLivePriceFromWs(lp, mk);
  else if (p.last_price != null && p.last_price !== "") noteLivePriceFromWs(p.last_price, mk);
}

function connectTradingUpdatesSocket() {
  if (tradingUpdatesSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(tradingUpdatesSocket.readyState)) {
    return;
  }
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}/ws/trading-updates`;
  const ws = new WebSocket(wsUrl);
  tradingUpdatesSocket = ws;
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
    window.setTimeout(connectTradingUpdatesSocket, 5000);
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
    try {
      systemStatsSocket.close();
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

function upsertChart(current, canvasId, config) {
  const el = document.getElementById(canvasId);
  if (!el) return current;

  // Voorkom CPU-verspilling: render niet als het canvas (of de tab) verborgen is (display: none)
  if (el.offsetParent === null) {
    return current;
  }

  const normalizedConfig = {
    ...config,
    options: highContrastChartOptions(config.options || {}),
  };
  if (!current) {
    const ctx = el.getContext("2d");
    return new Chart(ctx, normalizedConfig);
  }
  current.data = normalizedConfig.data;
  current.options = normalizedConfig.options;
  current.update();
  return current;
}

const BRAIN_NEON_LOSS_POLICY = "#00f8ff";
const BRAIN_NEON_LOSS_VALUE = "#ff3131";
const BRAIN_NEON_REWARD = "#39ff14";
const BRAIN_NEON_REWARD_MA = "#fff176";
const BRAIN_MIN_LR = 1.0e-5;
const BRAIN_MIN_EPS_PCT = 5.0;
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
    calibrationEl.textContent = calibrating
      ? `Bezig met kalibreren... (${mk})`
      : `Strategy actief voor ${mk} (Global steps: ${steps}).`;
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

  let n = Math.max(policyLoss.length, valueLoss.length, 1);
  const labels = Array.from({ length: n }, (_, i) => String(i + 1));
  const pl = [];
  const vl = [];
  for (let i = 0; i < n; i += 1) {
    pl.push(i < policyLoss.length ? policyLoss[i] : null);
    vl.push(i < valueLoss.length ? valueLoss[i] : null);
  }
  if (!policyLoss.length && !valueLoss.length) {
    pl[0] = 0;
    vl[0] = 0;
  }
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
          pointRadius: 0,
          spanGaps: true,
        },
        {
          label: "Value loss",
          data: vl,
          borderColor: BRAIN_NEON_LOSS_VALUE,
          backgroundColor: "rgba(255, 49, 49, 0.08)",
          borderWidth: 3,
          pointRadius: 0,
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
  upsertChart("brainTabTrainingLossChart", lossCfg);

  const featCfg = featureWeightsGroupedBarConfig(mergedFw, rawObs, fi);
  upsertChart("brainTabFeatureChart", featCfg);

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
  upsertChart("brainTabRewardChart", rewardCfg);

  const entropySeries = Array.isArray(monitor.policy_entropy) ? monitor.policy_entropy : [];
  const lastEntropy = entropySeries.length ? entropySeries[entropySeries.length - 1] : null;
  const lrEl = document.getElementById("brainTabStatLR");
  const entEl = document.getElementById("brainTabStatEntropy");
  const exEl = document.getElementById("brainTabStatExplore");
  const discEl = document.getElementById("brainTabStatDiscount");
  const batchEl = document.getElementById("brainTabStatBatch");
  const stEl = document.getElementById("brainTabStatSteps");
  if (lrEl) lrEl.textContent = Math.max(BRAIN_MIN_LR, Number(stats.learning_rate || 0)).toExponential(2);
  if (entEl) {
    entEl.textContent =
      lastEntropy !== null && Number.isFinite(Number(lastEntropy)) ? Number(lastEntropy).toFixed(4) : "—";
  }
  if (exEl) exEl.textContent = `${Math.max(BRAIN_MIN_EPS_PCT, Number(stats.exploration_rate_pct || 0)).toFixed(2)}%`;
  if (discEl) discEl.textContent = Number(stats.discount_factor || 0.99).toFixed(3);
  if (batchEl) batchEl.textContent = String(Number(stats.batch_size || 128).toFixed(0));
  if (stEl) stEl.textContent = Number(stats.global_step_count || 0).toLocaleString();
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
    try {
      brainStatsSocket.close();
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
    brainStatsReconnectTimer = setTimeout(connectBrainStatsSocket, 5000);
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
  try {
    const res = await fetch(
      `/api/v1/trades?limit=${LEDGER_ROUNDTRIP_FETCH_LIMIT}&view=roundtrip`
    );
    const data = await res.json();
    if (!res.ok) {
      cockpitLedgerStatusParts.ledger = `Ledger: API-fout (${res.status})`;
      paintCockpitLedgerStatus();
      return;
    }
    let rows = Array.isArray(data.trades) ? data.trades.slice() : [];
    const rowTime = (t) => {
      const s = String(t.close_time_utc || t.open_time_utc || t.exit_ts_utc || t.entry_ts_utc || t.ts || "").trim();
      const ms = s ? Date.parse(s) : NaN;
      return Number.isFinite(ms) ? ms : 0;
    };
    rows.sort((a, b) => rowTime(b) - rowTime(a));
    body.innerHTML = "";
    if (!rows.length) {
      cockpitLedgerStatusParts.ledger = "Ledger: 0 rondes (nog geen afgeronde trades)";
      paintCockpitLedgerStatus();
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.className = "cockpit-ledger-empty";
      td.textContent = "Nog geen trades in de geschiedenis.";
      tr.appendChild(td);
      body.appendChild(tr);
      return;
    }
    cockpitLedgerStatusParts.ledger = `Ledger: ${rows.length} ronde(s)`;
    paintCockpitLedgerStatus();
    const ledgerCoinLabel = (t) => {
      const raw = String(t.coin || (t.market || "").split("-")[0] || "?").trim().toUpperCase();
      return raw || "?";
    };

    for (const t of rows) {
      const tr = document.createElement("tr");
      const ts = String(t.open_time_utc || t.entry_ts_utc || t.ts || "").replace("T", " ").slice(0, 19);
      const pnl = Number(t.pnl_eur || 0);
      const entryPx = Number(t.entry_price || 0);
      const hasExit = t.exit_price !== null && t.exit_price !== undefined && String(t.exit_price) !== "";
      const exitPx = hasExit ? Number(t.exit_price || 0) : null;
      const pnlPct = Number(t.pnl_pct || 0);
      const ctx = escapeHtmlText(String(t.ledger_context || "—").slice(0, 200));
      const coin = escapeHtmlText(ledgerCoinLabel(t));
      const pairEsc = escapeHtmlText(String(t.market || t.pair || "-"));
      const chip = `<span class="ledger-asset-chip" data-coin="${coin}" title="${pairEsc}">${coin}</span>`;
      tr.innerHTML =
        `<td>${ts || "-"}</td>` +
        `<td class="cockpit-ledger-asset"><span class="ledger-asset-cell">${chip}<span class="ledger-asset-pair">${pairEsc}</span></span></td>` +
        `<td>${entryPx.toFixed(4)}</td>` +
        `<td>${hasExit ? exitPx.toFixed(4) : "ACTIVE"}</td>` +
        `<td class="${pnl >= 0 ? "positive" : "negative"}">${hasExit ? pnl.toFixed(2) : "ACTIVE"}</td>` +
        `<td class="${pnlPct >= 0 ? "positive" : "negative"}">${hasExit ? `${pnlPct.toFixed(2)}%` : "ACTIVE"}</td>` +
        `<td class="cockpit-ledger-context">${ctx}</td>`;
      body.appendChild(tr);
    }
  } catch (_err) {
    cockpitLedgerStatusParts.ledger = "Ledger: netwerkfout";
    paintCockpitLedgerStatus();
    body.innerHTML = "";
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7;
    td.className = "cockpit-ledger-empty";
    td.textContent = "Kon trades niet laden.";
    tr.appendChild(td);
    body.appendChild(tr);
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
  const destroyChart = (chart, canvasId) => {
    if (chart && typeof chart.destroy === 'function') {
      chart.destroy();
    }
    // DOM Cleanup: Verwijder het element fysiek om context-ghosting te voorkomen
    if (canvasId) {
      const oldCanvas = document.getElementById(canvasId);
      if (oldCanvas && oldCanvas.parentElement) {
        const wrapper = oldCanvas.parentElement;
        const newCanvas = document.createElement("canvas");
        newCanvas.id = canvasId;
        newCanvas.className = oldCanvas.className;
        oldCanvas.remove();
        wrapper.appendChild(newCanvas);
      }
    }
    return null;
  };

  equityCurveChart = destroyChart(equityCurveChart, "equityCurveChart");
  winLossChart = destroyChart(winLossChart, "winLossChart");
  sentimentOutcomeChart = destroyChart(sentimentOutcomeChart, "sentimentOutcomeChart");
  brainTabTrainingLossChart = destroyChart(brainTabTrainingLossChart, "brainTabTrainingLossChart");
  brainTabRewardChart = destroyChart(brainTabRewardChart, "brainTabRewardChart");
  brainTabFeatureChart = destroyChart(brainTabFeatureChart, "brainTabFeatureChart");
  brainBenchmarkChart = destroyChart(brainBenchmarkChart, "brainBenchmarkChart");
  brainCorrelationChart = destroyChart(brainCorrelationChart, "brainCorrelationChart");
  brainEpisodeChart = destroyChart(brainEpisodeChart, "brainEpisodeChart");
  brainEntropyChart = destroyChart(brainEntropyChart, "brainEntropyChart");
  brainLossChart = destroyChart(brainLossChart, "brainLossChart");
  brainNewsLagChart = destroyChart(brainNewsLagChart, "brainNewsLagChart");

  // Lightweight charts cleanup
  const priceHost = document.getElementById("priceChart");
  if (priceChart && typeof priceChart.remove === 'function') {
    priceChart.remove();
    priceChart = null;
    priceSeries = null;
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
    const data = await res.json();
    if (!res.ok) return;
    const consoleEl = document.getElementById("systemLogConsole");
    consoleEl.innerHTML = "";
    for (const line of data.lines || []) {
      appendSystemLogLine(line);
    }
  } catch (_err) {
    // ignore transient errors
  }
}

function connectSystemLogsSocket() {
  if (activeTab !== "hardware") return;
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
    appendSystemLogLine(incoming);
  };
  systemLogsSocket.onclose = () => {
    if (status) {
      status.textContent = "Reconnecting...";
      status.className = "status-disconnected genesis-mono-strong";
    }
    if (activeTab === "hardware") {
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
    try { systemLogsSocket.close(); } catch (_) {}
    systemLogsSocket = null;
  }
}

async function refreshBrainLab() {
  if (activeTab !== "aibrain") return;
  try {
    const marketParam = encodeURIComponent(selectedMarket || selectedChartPair || "BTC-EUR");
    const [reasoningRes, fiRes, monitorRes, stateRes, lagRes] = await Promise.all([
      fetch("/api/v1/brain/reasoning"),
      fetch(`/api/v1/brain/feature-importance?market=${marketParam}`),
      fetch("/api/v1/brain/training-monitor"),
      fetch("/api/v1/brain/state-overview"),
      fetch("/api/v1/brain/news-lag"),
    ]);
    const reasoningData = await reasoningRes.json();
    const fiData = await fiRes.json();
    const monitorData = await monitorRes.json();
    const stateData = await stateRes.json();
    const lagData = await lagRes.json();

    paintBrainTabCharts(monitorData, fiData);

    const net = monitorData.network_logs || {};
    const latestKl = (net.approx_kl || []).slice(-1)[0];
    const latestValueLoss = (net.value_loss || []).slice(-1)[0];
    const reasoningText =
      (reasoningData.reasoning ||
        "Nog geen RL-besluit beschikbaar. Start een paper cycle om reasoning te activeren.") +
      (latestKl !== undefined || latestValueLoss !== undefined
        ? `\nNetwork health: approx_kl=${Number(latestKl || 0).toFixed(6)}, value_loss=${Number(latestValueLoss || 0).toFixed(6)}.`
        : "");
    const rb = document.getElementById("brainReasoningBox");
    window.botMetrics.reasoningText = reasoningText;
    if (rb) rb.innerHTML = formatBrainReasoningHtml(reasoningText);

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
      upsertChart(
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
      upsertChart(
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
      upsertChart(
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
      upsertChart(
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
      upsertChart("brainLossChart", lossCfg);
    }

    const stats = monitorData.stats || {};
    const lrText = Number(stats.learning_rate || 0).toExponential(2);
    const stepText = Number(stats.global_step_count || 0).toLocaleString();
    const exText = `${Number(stats.exploration_rate_pct || 0).toFixed(2)}%`;
    const lrEl = document.getElementById("brainTabStatLR");
    if (lrEl) lrEl.textContent = lrText;
    const scEl = document.getElementById("brainTabStatSteps");
    if (scEl) scEl.textContent = stepText;
    const exEl = document.getElementById("brainTabStatExplore");
    if (exEl) exEl.textContent = exText;
    window.botMetrics.trainingStats = {
      learningRate: lrText,
      steps: stepText,
      exploration: exText,
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
    upsertChart(
      window.botMetrics.domIds.brainBenchmarkChart,
      benchCfg
    );
    if (cockpitMetricEl("terminalBenchmarkChart")) {
      terminalBenchmarkChart = upsertChart(
        terminalBenchmarkChart,
        window.botMetrics.domIds.terminalBenchmarkChart,
        benchCfg
      );
    }
    const benchRows =
      `<tr><td>RL Agent</td><td>${Number(benchmark.rl_pnl_pct || 0).toFixed(3)}%</td></tr>` +
      `<tr><td>Buy & Hold</td><td>${Number(benchmark.buy_hold_pnl_pct || 0).toFixed(3)}%</td></tr>` +
      `<tr><td>Alpha</td><td>${Number(benchmark.alpha_pct || 0).toFixed(3)}%</td></tr>`;
    const body = cockpitMetricEl("brainBenchmarkBody");
    if (body) body.innerHTML = benchRows;
    const tBenchBody = cockpitMetricEl("terminalBenchmarkBody");
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
    upsertChart("brainCorrelationChart", corrCfg);
    upsertChart("terminalCorrelationChart", corrCfg);
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
  if (!Number.isFinite(n) || n <= 0) return "—";
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
  const eurEl = document.getElementById("ledgerPerfPnlEur");
  const pctEl = document.getElementById("ledgerPerfPnlPct");
  const wrEl = document.getElementById("ledgerPerfWinRate");
  const clEl = document.getElementById("ledgerPerfClosed");
  const mwEl = document.getElementById("ledgerPerfMaxWin");
  const mlEl = document.getElementById("ledgerPerfMaxLoss");
  const hdEl = document.getElementById("ledgerPerfHold");
  if (!eurEl) return;
  const pnlEur = Number(ps.total_pnl_eur ?? data.wallet?.realized_pnl_eur ?? 0);
  const pnlPct = Number(ps.total_pnl_pct ?? data.wallet?.realized_pnl_pct ?? 0);
  const sign = pnlEur >= 0 ? "+" : "";
  eurEl.textContent = `${sign}€${Math.abs(pnlEur).toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  pctEl.textContent = `(${sign}${pnlPct.toFixed(2)}%)`;
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
    upsertChart("equityCurveChart", {
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

    const wl = data.analytics?.win_loss_ratio || {};
    upsertChart("winLossChart", {
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

    const svo = data.analytics?.sentiment_vs_outcome || [];
    upsertChart("sentimentOutcomeChart", {
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
  const rows = trades.slice(0, 5);
  if (!rows.length) {
    root.innerHTML = `<div class="trade-list-empty">Nog geen trades.</div>`;
    return;
  }
  let rendered = 0;
  for (const t of rows) {
    const action = String(t.action || t.type || "").toUpperCase();
    if (!["BUY", "SELL"].includes(action)) continue;
    const pnl = Number(t.pnl_eur || 0);
    const side = action === "BUY" ? "buy" : "sell";
    const row = document.createElement("div");
    row.className = `trade-list-row trade-list-row--${side}`;
    const tstr = String(t.ts || t.exit_ts_utc || "").slice(11, 19);
    row.innerHTML =
      `<span class="trade-list-dot" aria-hidden="true"></span>` +
      `<div class="trade-list-body">` +
      `<div class="trade-list-line1">${tstr} · ${t.market || "-"} · <span class="${action === "BUY" ? "positive" : "negative"}">${action}</span></div>` +
      `<div class="trade-list-line2">€ ${Number(t.entry_price || t.price || 0).toFixed(4)} · sent ${Number(t.sentiment_score || 0).toFixed(3)} · PnL <span class="${pnl >= 0 ? "positive" : "negative"}">${pnl.toFixed(2)}</span></div>` +
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
    const res = await fetch(`/api/v1/trades?limit=${SIDEBAR_TRADES_FETCH_LIMIT}&view=events`);
    const data = await res.json();
    if (!res.ok) return;
    const rows = (data.trades || []).slice(0, 5).map((t) => ({
      ts: t.exit_ts_utc || t.entry_ts_utc,
      market: t.market,
      action: (t.type || "SELL").toUpperCase(),
      entry_price: t.entry_price,
      sentiment_score: t.sentiment_score,
      pnl_eur: t.pnl_eur,
    }));
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
  try {
    const res = await fetch(`/paper/run?ticker=${encodeURIComponent(pair)}`, { method: "POST" });
    if (!res.ok) {
      console.warn("[paper/run] Server returned error", res.status);
      return;
    }
    const data = await res.json();
    document.getElementById("prediction").textContent = JSON.stringify(data, null, 2);
    await refreshSentiment();
    await refreshNewsInsights();
    await refreshActivity();
    await refreshChart();
    await refreshHistoryTrades();
    await refreshTradesTable();
    await refreshPerformanceAnalytics();
  } catch (err) {
    console.warn("[paper/run] Fetch failed:", err);
    const pEl = document.getElementById("prediction");
    if (pEl) pEl.textContent = "Fout bij uitvoeren paper trade (Network error).";
  }
}

document.getElementById("ledgerFooterToggle")?.addEventListener("click", () => {
  const footer = document.getElementById("liveLedgerFooter");
  const icon = document.getElementById("ledgerFooterIcon");
  if (!footer) return;
  footer.classList.toggle("is-collapsed");
  const isCol = footer.classList.contains("is-collapsed");
  if (icon) icon.textContent = isCol ? "▲ Uitklappen" : "▼ Inklappen";
});

document.getElementById("paperBtn").addEventListener("click", runPaperTrade);
document.getElementById("marketSelect").addEventListener("change", selectMarketFromDropdown);
document.getElementById("refreshMarketsBtn").addEventListener("click", refreshMarkets);
document.getElementById("pauseBtn").addEventListener("click", () => postBotAction("/bot/pause"));
document.getElementById("resumeBtn").addEventListener("click", () => postBotAction("/bot/resume"));
document.getElementById("panicBtn").addEventListener("click", () => postBotAction("/bot/panic"));
document.getElementById("newsModalClose").addEventListener("click", closeNewsModal);
document.getElementById("btn-terminal")?.addEventListener("click", () => switchTab("terminal"));
document.getElementById("btn-aibrain")?.addEventListener("click", () => switchTab("aibrain"));
document.getElementById("btn-ledger")?.addEventListener("click", () => switchTab("ledger"));
document.getElementById("btn-hardware")?.addEventListener("click", () => switchTab("hardware"));
document.getElementById("systemLogPauseBtn").addEventListener("click", toggleSystemLogPause);
document.getElementById("systemLogClearBtn").addEventListener("click", clearSystemLogConsole);
document.getElementById("systemLogMuteBtn").addEventListener("click", toggleSystemLogMute);
document.getElementById("newsModal").addEventListener("click", (event) => {
  if (event.target.id === "newsModal") closeNewsModal();
});

window.setChartInterval = async function setChartInterval(next) {
  const allowed = new Set(["1m", "5m", "15m", "60m"]);
  const val = String(next || "").toLowerCase();
  if (!allowed.has(val)) return;
  chartInterval = val;
  const foot = document.getElementById("chartTimeFootnote");
  if (foot) foot.textContent = `Timeframes: 1m | 5m | 15m | 60m | active=${chartInterval}`;
  await updateChart(selectedChartPair || selectedMarket);
};

/**
 * Sequenced startup loader — voorkomt 'Network Storm' / ERR_EMPTY_RESPONSE na FinBERT/PPO-load.
 * Fase 1: health + markten (+ geselecteerde markt) → korte pauze.
 * Fase 2: activity + bot status → korte pauze.
 * Fase 3: zware endpoints (trades/ledger + performance analytics).
 * Fase 4: chart + nieuws + brain + WebSockets.
 */
async function runSequencedStartup() {
  updateSystemPauseButton();
  updateSystemMuteButton();
  cockpitLedgerStatusParts.markets = "Markten: ophalen…";
  cockpitLedgerStatusParts.ledger = "Ledger: ophalen…";
  paintCockpitLedgerStatus();

  await Promise.allSettled([refreshHealthMode(), refreshSelectedMarket(), refreshMarkets()]);
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
}

(async () => {
  try {
    await runSequencedStartup();
  } catch (err) {
    console.error("[startup] sequenced loader failed:", err);
  }
})();

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
}, 30000, 10000);

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
let activePollers = {};

function switchTab(tabName) {
  activeTab = tabName;

  // 1. Verberg alle tabs en reset knoppen
  const allTabs = ["terminal", "aibrain", "ledger", "hardware"];
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
  Object.values(activePollers).forEach(clearInterval);
  activePollers = {};

  // Stop WebSockets op basis van tab (strikte data-scheiding)
  if (tabName !== "hardware") stopSystemLogsSocket();
  if (tabName !== "terminal") {
    if (wsRef) { try { wsRef.close(); } catch(e){} wsRef = null; }
    if (tradingUpdatesSocket) { try { tradingUpdatesSocket.close(); } catch(e){} tradingUpdatesSocket = null; }
  }

  // 4. Start exclusieve endpoints per view
  if (tabName === "terminal") {
    connectBitvavoPriceStream();
    connectTradingUpdatesSocket();
    fetchTerminalSnapshot();
    activePollers.terminal = setInterval(fetchTerminalSnapshot, 3000);
    setTimeout(() => updateChart(selectedChartPair || selectedMarket), 100);
  }
  else if (tabName === "aibrain") {
    // Render direct de laatste gebufferde snapshot om wachttijd te voorkomen
    if (latestBrainStatsPayload) {
      const data = normalizeBrainWsPayload(latestBrainStatsPayload);
      if (data) enqueueCockpitWsRender(() => applyBrainDataPayload(data));
      latestBrainStatsPayload = null; // Consumeer
    }
    fetchAILogic();
    activePollers.aibrain = setInterval(fetchAILogic, 3000);
  }
  else if (tabName === "ledger") {
    fetchLedger();
    activePollers.ledger = setInterval(fetchLedger, 5000);
  }
  else if (tabName === "hardware") {
    fetchHardwareStats();
    activePollers.hardware = setInterval(fetchHardwareStats, 2000);
    connectSystemLogsSocket();
    refreshSystemLogsSnapshot();
  }

  initHintPortals();
}

// Geïsoleerde Fetch Functies (Endpoint Correctie)
async function fetchTerminalSnapshot() {
  try {
    const res = await fetch('/api/v1/snapshot');
    if (!res.ok) return;
    const data = await res.json();
    if (data && data.tenant) applyActivityResponse(data.tenant);
  } catch (e) { console.error("[Terminal] Snapshot fetch failed", e); }
}

async function fetchAILogic() {
  try {
    const res = await fetch('/api/v1/ai_logic');
    if (!res.ok) return;
    const data = await res.json();
    const reasoningBox = document.getElementById('brainReasoningBox');
    if (reasoningBox) reasoningBox.innerHTML = formatBrainReasoningHtml(data.reasoning || "Geen reasoning beschikbaar.");
    const weightsBox = document.getElementById('brainWeightsBox');
    if (weightsBox) weightsBox.innerText = JSON.stringify(data.feature_weights || {}, null, 2);
  } catch (e) { console.error("[AI Brain] Logic fetch failed", e); }
}

async function fetchHardwareStats() {
  try {
    const res = await fetch('/api/v1/system_stats');
    if (!res.ok) return;
    const data = await res.json();
    applySystemStatsPayload(data);
  } catch (e) { console.error("[Hardware] Stats fetch failed", e); }
}

async function fetchLedger() {
  try {
    await refreshHistoryTrades();
    await refreshPerformanceAnalytics();
  } catch (e) { console.error("[Ledger] Fetch failed", e); }
}
