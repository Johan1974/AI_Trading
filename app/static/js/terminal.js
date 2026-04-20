/*
  Bestand: app/static/js/terminal.js
  Relatief pad: ./app/static/js/terminal.js
  Functie: Clientlogica voor immersive trading terminal met chart markers en live inzichten.
*/

let selectedMarket = "BTC-EUR";
let selectedChartPair = "BTC-EUR";
let wsRef = null;
let priceChart = null;
let priceSeries = null;
let markerSeries = null;
let priceSeriesIsCandle = false;
let priceChartResizeObserver = null;
let equityCurveChart = null;
let winLossChart = null;
let sentimentOutcomeChart = null;
let latestNewsItems = [];
let brainFeatureChart = null;
let brainRewardChart = null;
let brainEpisodeChart = null;
let brainEntropyChart = null;
let brainLossChart = null;
let brainNewsLagChart = null;
let brainBenchmarkChart = null;
let brainCorrelationChart = null;
let terminalBenchmarkChart = null;
let terminalCorrelationChart = null;
let terminalRewardChart = null;
let terminalFeatureChart = null;
let terminalEpisodeChart = null;
let terminalEntropyChart = null;
let terminalLossChart = null;
let terminalNewsLagChart = null;
let activeTab = "terminal";
let systemLogsSocket = null;
let systemLogsReconnectTimer = null;
let systemStatsSocket = null;
let systemStatsReconnectTimer = null;
const MAX_SYSTEM_LOG_LINES = 500;
let systemLogsPaused = false;
let systemLogsMuted = false;
let systemLogsBuffer = [];
let lastLedgerCycleSeq = null;

/** Donkere gridlijnen zodat witte assen en neon datasets contrast houden */
const CHART_GRID_DARK = "#222222";
const CHART_AXIS_WHITE = "#FFFFFF";

const HINT_PORTAL_ID = "brainHintPortal";

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
    document.getElementById("terminalTab"),
    document.getElementById("brainTab"),
    document.getElementById("ledgerTab"),
    document.getElementById("hardwareTab"),
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
  const el = document.getElementById("liveUpdatedAt");
  if (!el) return;
  const now = new Date();
  el.textContent = `Laatste update: ${now.toLocaleTimeString()}`;
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

function connectBitvavoPriceStream() {
  if (wsRef) {
    try { wsRef.close(); } catch (_) {}
  }
  const ws = new WebSocket("wss://ws.bitvavo.com/v2/");
  wsRef = ws;
  ws.onopen = () => {
    ws.send(JSON.stringify({
      action: "subscribe",
      channels: [{ name: "ticker24h", markets: [selectedMarket] }],
    }));
  };
  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload?.event === "ticker24h" && payload?.data?.length) {
        const row = payload.data[0];
        const price = row.last ?? row.close ?? "-";
        const lp = document.getElementById("livePrice");
        if (lp) {
          const n = Number(row.last ?? row.close);
          lp.textContent = Number.isFinite(n)
            ? `${String(selectedMarket).toUpperCase()}  ${n.toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 6 })}`
            : String(price);
        }
      }
    } catch (_) {}
  };
  ws.onclose = () => setTimeout(connectBitvavoPriceStream, 2000);
}

async function refreshHealthMode() {
  const res = await fetch("/health");
  const data = await res.json();
  renderModeBanner(data.mode);
}

function syncHeaderMarketChip() {
  const chip = document.getElementById("headerPairDisplay");
  if (chip) chip.textContent = String(selectedMarket || selectedChartPair || "BTC-EUR").toUpperCase();
}

async function refreshSelectedMarket() {
  const res = await fetch("/markets/selected");
  const data = await res.json();
  selectedMarket = data.selected_market || "BTC-EUR";
  syncHeaderMarketChip();
}

async function refreshMarkets() {
  const res = await fetch("/markets/active");
  const data = await res.json();
  const select = document.getElementById("marketSelect");
  select.innerHTML = "";
  for (const m of data.markets || []) {
    const opt = document.createElement("option");
    opt.value = m.market;
    opt.textContent = `${m.market} | Vol24h: ${m.volume_quote_24h}`;
    if (m.market === selectedMarket) opt.selected = true;
    select.appendChild(opt);
  }
  syncHeaderMarketChip();
}

async function selectMarketFromDropdown() {
  const market = document.getElementById("marketSelect").value;
  const res = await fetch(`/markets/select?market=${encodeURIComponent(market)}`, { method: "POST" });
  const data = await res.json();
  selectedMarket = data.selected_market || market;
  syncHeaderMarketChip();
  connectBitvavoPriceStream();
  await updateChart(selectedMarket);
  await refreshBalanceCheck();
}

async function refreshBalanceCheck() {
  const bc = document.getElementById("balanceCheck");
  if (!bc) return;
  try {
    const res = await fetch(`/vault/balance-check?market=${encodeURIComponent(selectedMarket)}`);
    const data = await res.json();
    if (!res.ok || data.available === false) {
      bc.innerHTML =
        `Vault status: <span class="status-disconnected">Disconnected</span> (${data.reason || "unknown"})`;
      setLiveUpdatedAt();
      return;
    }
    bc.innerHTML =
      `Vault status: <span class="status-connected">Connected</span> | ` +
      `(${selectedMarket}) Buy: ${data.sufficient_for_buy} | Sell: ${data.sufficient_for_sell} | CanTrade: ${data.can_trade}`;
    setLiveUpdatedAt();
  } catch (err) {
    bc.innerHTML =
      `Vault status: <span class="status-disconnected">Disconnected</span> (error: ${String(err)})`;
    setLiveUpdatedAt();
  }
}

async function refreshSentiment() {
  const valEl = document.getElementById("sentimentValue");
  const barEl = document.getElementById("sentimentBar");
  if (!valEl || !barEl) return;
  const res = await fetch("/sentiment/current");
  const data = await res.json();
  const score = Number(data.sentiment_score ?? 0);
  const conf = Number(data.sentiment_confidence ?? 0);
  valEl.textContent = score.toFixed(3);
  const confEl = document.getElementById("sentimentConfidence");
  if (confEl) confEl.textContent = conf.toFixed(3);
  const pct = Math.max(0, Math.min(100, ((score + 1) / 2) * 100));
  barEl.style.width = `${pct}%`;
  setLiveUpdatedAt();
}

async function refreshBotStatus() {
  const res = await fetch("/bot/status");
  const data = await res.json();
  const st = document.getElementById("botStatus");
  if (st) st.textContent = `Bot status: ${data.bot_status}`;
  setLiveUpdatedAt();
}

async function refreshStorageHealth() {
  if (!document.getElementById("storageUsagePct")) return;
  try {
    const res = await fetch("/api/v1/system/storage");
    const data = await res.json();
    if (!res.ok) return;
    const disk = data.disk || {};
    const stats = data.stats || {};
    const usagePct = Number(disk.usage_pct || 0);
    document.getElementById("storageUsagePct").textContent = `${usagePct.toFixed(2)}%`;
    const bar = document.getElementById("storageUsageBar");
    bar.style.width = `${Math.max(0, Math.min(100, usagePct))}%`;
    bar.classList.toggle("storage-critical", usagePct >= 85);
    bar.classList.toggle("storage-warning", usagePct >= 70 && usagePct < 85);
    document.getElementById("storageOptimizationStats").textContent =
      `Laatste opschoning bespaarde: ${bytesToMB(stats.saved_bytes)} MB`;
    document.getElementById("storageDataHealth").textContent =
      `Historie: ${Number(stats.history_days || 400)} dagen | Resolutie: ${stats.resolution || "Mixed (1s/1m)"}`;
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

function ensureChartAndSeries(host) {
  const hostWidth = Math.max(320, Number(host.clientWidth || 0));
  const hostHeight = Math.max(260, Number(host.clientHeight || 0));

  if (!priceChart) {
    priceChart = LightweightCharts.createChart(host, {
      layout: { background: { color: "#000000" }, textColor: "#FFFFFF" },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.2)" },
        horzLines: { color: "rgba(255, 255, 255, 0.2)" },
      },
      width: hostWidth,
      height: hostHeight,
      rightPriceScale: {
        borderColor: "#FFFFFF",
        scaleMargins: { top: 0.08, bottom: 0.12 },
      },
      timeScale: { borderColor: "#FFFFFF" },
      localization: { locale: "nl-NL" },
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
  priceChart.applyOptions({
    width: Math.max(320, Number(host.clientWidth || 0)),
    height: Math.max(260, Number(host.clientHeight || 0)),
  });
}

function buildOrUpdateChart(labels, prices, markers) {
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
  if (!lineData.length) return;
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

  const lwMarkers = markers
    .map((m) => ({
      time: toEpochSeconds(m.ts),
      position: String(m.signal).toUpperCase() === "SELL" ? "aboveBar" : "belowBar",
      color: String(m.signal).toUpperCase() === "SELL" ? "#ff3131" : "#39ff14",
      shape: String(m.signal).toUpperCase() === "SELL" ? "arrowDown" : "arrowUp",
      text: `${m.signal} ${Number(m.expected_return_pct || 0).toFixed(2)}%`,
    }))
    .filter((m) => m.time !== null)
    .sort((a, b) => Number(a.time) - Number(b.time));
  setSeriesMarkers(markerSeries, lwMarkers);
  priceChart.timeScale().fitContent();
}

async function refreshChart() {
  await updateChart(selectedChartPair || selectedMarket);
}

async function updateChart(newPair) {
  selectedChartPair = String(newPair || selectedMarket || "BTC-EUR").toUpperCase();
  const predictionEl = document.getElementById("prediction");
  if (predictionEl) {
    predictionEl.textContent = `Laden van ${selectedChartPair} data...`;
  }
  try {
    if (priceSeries && typeof priceSeries.setData === "function") {
      priceSeries.setData([]);
      setSeriesMarkers(markerSeries, []);
    }
    const res = await fetch(`/api/v1/history?pair=${encodeURIComponent(selectedChartPair)}&lookback_days=180`);
    const data = await res.json();
    if (!res.ok) {
      if (predictionEl) predictionEl.textContent = `Kon ${selectedChartPair} data niet laden.`;
      return;
    }
    if (predictionEl && String(predictionEl.textContent || "").startsWith("Laden van ")) {
      predictionEl.textContent = `Chart geladen voor ${data.tv_symbol || selectedChartPair}.`;
    }
    buildOrUpdateChart(data.labels || [], data.prices || [], data.markers || []);
  } catch (_err) {
    if (predictionEl) predictionEl.textContent = `Laden van ${selectedChartPair} data mislukt.`;
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
  row.addEventListener("click", () => openNewsModal(item));
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
  const res = await fetch("/api/v1/news/ticker");
  const data = await res.json();
  const items = Array.isArray(data) ? data : [];
  latestNewsItems = items;
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
  renderNewsStream("terminalNewsStream", mapped);
  renderIntelligenceTicker(mapped);
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

async function refreshActivity() {
  const res = await fetch("/activity");
  const data = await res.json();
  const p = data.paper_portfolio || {};
  const eq = document.getElementById("portfolioEquity");
  const cash = document.getElementById("portfolioCash");
  const rest = document.getElementById("portfolioStatsRest");
  if (eq) eq.textContent = p.equity != null ? String(p.equity) : "-";
  if (cash) cash.textContent = p.cash != null ? String(p.cash) : "-";
  const heq = document.getElementById("headerEquity");
  const hca = document.getElementById("headerCash");
  if (heq) heq.textContent = p.equity != null ? String(p.equity) : "-";
  if (hca) hca.textContent = p.cash != null ? String(p.cash) : "-";
  const hBal = document.getElementById("headerBalanceEuro");
  if (hBal && p.equity != null) {
    const e = Number(p.equity);
    hBal.textContent = Number.isFinite(e) ? `€${e.toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "€—";
  }
  if (rest) {
    rest.textContent =
      `Pos Qty: ${p.position_qty ?? "-"} | Realized PnL: ${p.realized_pnl_eur ?? p.realized_pnl ?? "-"}`;
  }
  const lastOrder = data.last_order?.order || {};
  const ao = document.getElementById("activeOrders");
  if (ao) {
    ao.textContent =
      `Actieve orders: ${lastOrder.signal || "-"} ${lastOrder.ticker || ""} ${lastOrder.amount_quote_eur || lastOrder.amount_quote || ""}`.trim();
  }
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
  const beq = document.getElementById("brainPortfolioEquity");
  const bcash = document.getElementById("brainPortfolioCash");
  const bmeta = document.getElementById("brainPortfolioMeta");
  if (beq) beq.textContent = p.equity != null ? String(p.equity) : "-";
  if (bcash) bcash.textContent = p.cash != null ? String(p.cash) : "-";
  if (bmeta) {
    bmeta.textContent =
      `Pos: ${p.position_qty ?? "-"} | PnL: ${p.realized_pnl_eur ?? p.realized_pnl ?? "-"} | Orders: ${lastOrder.signal || "-"} ${lastOrder.ticker || ""}`.trim();
  }
  const cyc = data.last_order?.cycle_seq;
  if (cyc != null && cyc !== lastLedgerCycleSeq) {
    lastLedgerCycleSeq = cyc;
    const fs = String(data.last_order?.engine_risk?.final_signal || "").toUpperCase();
    console.info("[LEDGER] Bot cycle", cyc, "final_signal=", fs, "— activity + trade API in sync");
  }
  const rp = data.risk_profile;
  if (rp) {
    const applyRiskRow = (baseId, maxId, slId) => {
      const baseEl = document.getElementById(baseId);
      const maxEl = document.getElementById(maxId);
      const slEl = document.getElementById(slId);
      if (baseEl) {
        const v = Number(rp.base_trade_eur);
        baseEl.textContent = Number.isFinite(v) ? `€${v.toFixed(0)}` : String(rp.base_trade_eur ?? "-");
      }
      if (maxEl) {
        const v = Number(rp.max_risk_pct);
        maxEl.textContent = Number.isFinite(v) ? `${v.toFixed(1)}%` : String(rp.max_risk_pct ?? "-");
      }
      if (slEl) {
        const v = Number(rp.stop_loss_pct);
        slEl.textContent = Number.isFinite(v) ? `${v.toFixed(1)}%` : String(rp.stop_loss_pct ?? "-");
      }
    };
    applyRiskRow("riskProfileBase", "riskProfileMax", "riskProfileSl");
    applyRiskRow("brainRiskProfileBase", "brainRiskProfileMax", "brainRiskProfileSl");
  }
}

function applySystemStatsPayload(data) {
  if (!data || data.topic !== "system_stats") return;
  const cpu = Math.max(0, Math.min(100, Number(data.cpu_pct) || 0));
  const ramPct = Math.max(0, Math.min(100, Number(data.ram_pct) || 0));
  const gpuSm = Math.max(0, Math.min(100, Number(data.gpu_util_pct) || 0));
  const gpuEffIn = data.gpu_util_effective;
  const gpuEff = Math.max(0, Math.min(100, Number(gpuEffIn)));
  const gpu = gpuEffIn !== undefined && gpuEffIn !== null && Number.isFinite(gpuEff) ? gpuEff : gpuSm;
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
  if (hCpu) hCpu.textContent = `CPU ${cpu.toFixed(0)}%`;
  if (hRam) hRam.textContent = `RAM ${ramPct.toFixed(0)}%`;
  if (hGpu) {
    hGpu.textContent = `GPU ${gpu.toFixed(0)}%`;
    hGpu.classList.toggle("cockpit-gpu-neon", Boolean(data.gpu_ok) && gpu > 0.5);
    hGpu.classList.toggle("cockpit-gpu-glow", Boolean(data.gpu_ok) && gpu > 0);
    hGpu.classList.toggle("header-gpu-sensor-pulse", Boolean(data.gpu_ok) && gpu <= 0);
  }
  if (hDisk) hDisk.textContent = `DISK ${disk.toFixed(0)}%`;
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
      const data = JSON.parse(event.data || "{}");
      applySystemStatsPayload(data);
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

async function refreshHistoryTrades() {
  const body = document.getElementById("cockpitLedgerBody");
  if (!body) return;
  try {
    const res = await fetch("/api/v1/trades?limit=200");
    const data = await res.json();
    if (!res.ok) return;
    const rows = Array.isArray(data.trades) ? data.trades : [];
    body.innerHTML = "";
    if (!rows.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.className = "cockpit-ledger-empty";
      td.textContent = "Nog geen trades in de geschiedenis.";
      tr.appendChild(td);
      body.appendChild(tr);
      return;
    }
    for (const t of rows) {
      const action = String(t.type || t.action || "").toUpperCase();
      const tr = document.createElement("tr");
      const ts = String(t.exit_ts_utc || t.entry_ts_utc || t.ts || "").replace("T", " ").slice(0, 19);
      const pnl = Number(t.pnl_eur || 0);
      const px = Number(t.exit_price || t.entry_price || t.price || 0);
      const qty = Number(t.qty ?? 0);
      tr.innerHTML =
        `<td>${ts || "-"}</td>` +
        `<td>${t.market || "-"}</td>` +
        `<td class="${action === "BUY" ? "positive" : "negative"}">${action || "-"}</td>` +
        `<td>${px.toFixed(4)}</td>` +
        `<td>${qty.toFixed(6)}</td>` +
        `<td>${Number(t.sentiment_score || 0).toFixed(3)}</td>` +
        `<td class="${pnl >= 0 ? "positive" : "negative"}">${pnl.toFixed(2)}</td>`;
      body.appendChild(tr);
    }
  } catch (_err) {
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

function setActiveTab(tabId) {
  activeTab = tabId;
  const terminal = document.getElementById("terminalTab");
  const brain = document.getElementById("brainTab");
  const ledger = document.getElementById("ledgerTab");
  const hardware = document.getElementById("hardwareTab");
  const tBtn = document.getElementById("tabTerminalBtn");
  const bBtn = document.getElementById("tabBrainBtn");
  const hBtn = document.getElementById("tabLedgerBtn");
  const sBtn = document.getElementById("tabHardwareBtn");
  if (!terminal || !brain || !ledger || !hardware || !tBtn || !bBtn || !hBtn || !sBtn) return;
  const headerTrading = document.getElementById("headerTradingActions");
  const headerSystem = document.getElementById("headerSystemActions");
  if (headerTrading && headerSystem) {
    if (tabId === "hardware") {
      headerTrading.classList.add("hidden");
      headerSystem.classList.remove("hidden");
    } else {
      headerTrading.classList.remove("hidden");
      headerSystem.classList.add("hidden");
    }
  }
  initHintPortals();
  terminal.classList.add("hidden");
  brain.classList.add("hidden");
  ledger.classList.add("hidden");
  hardware.classList.add("hidden");
  tBtn.classList.remove("active");
  bBtn.classList.remove("active");
  hBtn.classList.remove("active");
  sBtn.classList.remove("active");
  stopSystemLogsSocket();

  if (tabId === "brain") {
    brain.classList.remove("hidden");
    bBtn.classList.add("active");
    refreshNewsInsights();
    refreshActivity();
    refreshBrainLab();
    return;
  }
  if (tabId === "ledger") {
    ledger.classList.remove("hidden");
    hBtn.classList.add("active");
    refreshLedgerTab();
    return;
  }
  if (tabId === "hardware") {
    hardware.classList.remove("hidden");
    sBtn.classList.add("active");
    refreshSystemLogsSnapshot();
    connectSystemLogsSocket();
    return;
  }
  terminal.classList.remove("hidden");
  tBtn.classList.add("active");
  connectSystemStatsSocket();
  refreshBrainLab();
  refreshNewsInsights();
  refreshActivity();
  requestAnimationFrame(() => {
    const host = document.getElementById("priceChart");
    if (host) resizePriceChart(host);
  });
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
  if (activeTab !== "brain" && activeTab !== "terminal") return;
  try {
    const [reasoningRes, fiRes, monitorRes, stateRes, lagRes] = await Promise.all([
      fetch("/api/v1/brain/reasoning"),
      fetch("/api/v1/brain/feature-importance"),
      fetch("/api/v1/brain/training-monitor"),
      fetch("/api/v1/brain/state-overview"),
      fetch("/api/v1/brain/news-lag"),
    ]);
    const reasoningData = await reasoningRes.json();
    const fiData = await fiRes.json();
    const monitorData = await monitorRes.json();
    const stateData = await stateRes.json();
    const lagData = await lagRes.json();

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
    if (rb) rb.innerHTML = formatBrainReasoningHtml(reasoningText);
    const trb = document.getElementById("terminalReasoningBox");
    if (trb) trb.innerHTML = formatBrainReasoningHtml(reasoningText);

    const fw = fiData.feature_weights || {};
    const featureChartConfig = {
      type: "bar",
      data: {
        labels: Object.keys(fw),
        datasets: [{ label: "Feature Importance", data: Object.values(fw), backgroundColor: "#00f8ff" }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: {
              maxRotation: 45,
              minRotation: 45,
              color: CHART_AXIS_WHITE,
            },
          },
        },
      },
    };
    brainFeatureChart = upsertChart(brainFeatureChart, "brainFeatureChart", featureChartConfig);
    if (document.getElementById("terminalFeatureChart")) {
      terminalFeatureChart = upsertChart(terminalFeatureChart, "terminalFeatureChart", featureChartConfig);
    }

    const rewards = monitorData.reward || [];
    const episodeLen = monitorData.episode_length || [];
    const entropy = monitorData.policy_entropy || [];
    const loss = monitorData.loss || [];
    const lagRows = Array.isArray(lagData.items) ? lagData.items : [];
    const lineOpts = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } };
    const rewardCfg = {
      type: "line",
      data: {
        labels: rewards.map((_, idx) => idx + 1),
        datasets: [
          {
            label: "Cumulative Reward",
            data: rewards,
            borderColor: "#66ff66",
            backgroundColor: "rgba(102, 255, 102, 0.06)",
            borderWidth: 4,
            pointRadius: 0,
          },
        ],
      },
      options: lineOpts,
    };
    if (document.getElementById("brainRewardChart")) {
      brainRewardChart = upsertChart(brainRewardChart, "brainRewardChart", rewardCfg);
    }
    if (document.getElementById("terminalRewardChart")) {
      terminalRewardChart = upsertChart(terminalRewardChart, "terminalRewardChart", rewardCfg);
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
      brainNewsLagChart = upsertChart(brainNewsLagChart, "brainNewsLagChart", lagCfg);
    }
    if (document.getElementById("terminalNewsLagChart")) {
      terminalNewsLagChart = upsertChart(terminalNewsLagChart, "terminalNewsLagChart", lagCfg);
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
      brainEpisodeChart = upsertChart(brainEpisodeChart, "brainEpisodeChart", epCfg);
    }
    if (document.getElementById("terminalEpisodeChart")) {
      terminalEpisodeChart = upsertChart(terminalEpisodeChart, "terminalEpisodeChart", epCfg);
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
      brainEntropyChart = upsertChart(brainEntropyChart, "brainEntropyChart", entCfg);
    }
    if (document.getElementById("terminalEntropyChart")) {
      terminalEntropyChart = upsertChart(terminalEntropyChart, "terminalEntropyChart", entCfg);
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
      brainLossChart = upsertChart(brainLossChart, "brainLossChart", lossCfg);
    }
    if (document.getElementById("terminalLossChart")) {
      terminalLossChart = upsertChart(terminalLossChart, "terminalLossChart", lossCfg);
    }

    const stats = monitorData.stats || {};
    const lrText = Number(stats.learning_rate || 0).toExponential(2);
    const stepText = Number(stats.global_step_count || 0).toLocaleString();
    const exText = `${Number(stats.exploration_rate_pct || 0).toFixed(2)}%`;
    const lrEl = document.getElementById("brainLearningRate");
    if (lrEl) lrEl.textContent = lrText;
    const scEl = document.getElementById("brainStepCount");
    if (scEl) scEl.textContent = stepText;
    const exEl = document.getElementById("brainExplorationRate");
    if (exEl) exEl.textContent = exText;
    const tlr = document.getElementById("terminalLearningRate");
    if (tlr) tlr.textContent = lrText;
    const tsc = document.getElementById("terminalStepCount");
    if (tsc) tsc.textContent = stepText;
    const tex = document.getElementById("terminalExplorationRate");
    if (tex) tex.textContent = exText;

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
    brainBenchmarkChart = upsertChart(brainBenchmarkChart, "brainBenchmarkChart", benchCfg);
    if (document.getElementById("terminalBenchmarkChart")) {
      terminalBenchmarkChart = upsertChart(terminalBenchmarkChart, "terminalBenchmarkChart", benchCfg);
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
    brainCorrelationChart = upsertChart(brainCorrelationChart, "brainCorrelationChart", corrCfg);
    if (document.getElementById("terminalCorrelationChart")) {
      terminalCorrelationChart = upsertChart(terminalCorrelationChart, "terminalCorrelationChart", corrCfg);
    }
    const cv = document.getElementById("brainCorrValue");
    if (cv) cv.textContent = corrValue.toFixed(4);
    const nw = document.getElementById("brainNewsWeight");
    if (nw) nw.textContent = newsWeight.toFixed(4);
    const pw = document.getElementById("brainPriceWeight");
    if (pw) pw.textContent = priceWeight.toFixed(4);
    const tcv = document.getElementById("terminalCorrValue");
    if (tcv) tcv.textContent = corrValue.toFixed(4);
    const tnw = document.getElementById("terminalNewsWeight");
    if (tnw) tnw.textContent = newsWeight.toFixed(4);
    const tpw = document.getElementById("terminalPriceWeight");
    if (tpw) tpw.textContent = priceWeight.toFixed(4);

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
    const tfg = document.getElementById("terminalStateFearGreed");
    if (tfg) tfg.textContent = fg;
    const tbtc = document.getElementById("terminalStateBtc");
    if (tbtc) tbtc.textContent = btc;
    const twh = document.getElementById("terminalStateWhale");
    if (twh) twh.textContent = wh;
    const tmw = document.getElementById("terminalStateMacro");
    if (tmw) tmw.textContent = macro;
    const trsi = document.getElementById("terminalStateRsi");
    if (trsi) trsi.textContent = rsi;
    const tmacd = document.getElementById("terminalStateMacd");
    if (tmacd) tmacd.textContent = macdVal;
    const tsw = document.getElementById("terminalStateWeights");
    if (tsw) tsw.textContent = focusLine;
  } catch (_err) {
    const fail = "AI Brain data kon niet geladen worden.";
    const b = document.getElementById("brainReasoningBox");
    if (b) b.textContent = fail;
    const t = document.getElementById("terminalReasoningBox");
    if (t) t.textContent = fail;
  }
}

async function refreshPerformanceAnalytics() {
  if (!document.getElementById("equityCurveChart")) return;
  const res = await fetch("/api/v1/performance/analytics");
  const data = await res.json();
  if (!res.ok) return;

  const equity = data.equity_curve || [];
  const equityLabels = equity.map((r) => (r.ts || "").slice(11, 19));
  const equityValues = equity.map((r) => Number(r.equity || 0));
  equityCurveChart = upsertChart(equityCurveChart, "equityCurveChart", {
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
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } },
  });

  const wl = data.analytics?.win_loss_ratio || {};
  winLossChart = upsertChart(winLossChart, "winLossChart", {
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
    options: { responsive: true, maintainAspectRatio: false },
  });

  const svo = data.analytics?.sentiment_vs_outcome || [];
  sentimentOutcomeChart = upsertChart(sentimentOutcomeChart, "sentimentOutcomeChart", {
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
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } },
  });

  renderTradeHistory(data.recent_actions || []);
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
  await refreshHistoryTrades();
  try {
    const res = await fetch("/api/v1/trades?limit=5");
    const data = await res.json();
    if (!res.ok) return;
    const rows = (data.trades || []).map((t) => ({
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
  await fetch(path, { method: "POST" });
  await refreshBotStatus();
  await refreshStorageHealth();
  await refreshActivity();
}

async function runPaperTrade() {
  const pair = String(selectedMarket || document.getElementById("marketSelect")?.value || "BTC-EUR").toUpperCase();
  const res = await fetch(`/paper/run?ticker=${encodeURIComponent(pair)}`, { method: "POST" });
  const data = await res.json();
  document.getElementById("prediction").textContent = JSON.stringify(data, null, 2);
  await refreshSentiment();
  await refreshNewsInsights();
  await refreshActivity();
  await refreshChart();
  await refreshPerformanceAnalytics();
}

document.getElementById("paperBtn").addEventListener("click", runPaperTrade);
document.getElementById("marketSelect").addEventListener("change", selectMarketFromDropdown);
document.getElementById("refreshMarketsBtn").addEventListener("click", refreshMarkets);
document.getElementById("pauseBtn").addEventListener("click", () => postBotAction("/bot/pause"));
document.getElementById("resumeBtn").addEventListener("click", () => postBotAction("/bot/resume"));
document.getElementById("panicBtn").addEventListener("click", () => postBotAction("/bot/panic"));
document.getElementById("newsModalClose").addEventListener("click", closeNewsModal);
document.getElementById("tabTerminalBtn")?.addEventListener("click", () => setActiveTab("terminal"));
document.getElementById("tabBrainBtn")?.addEventListener("click", () => setActiveTab("brain"));
document.getElementById("tabLedgerBtn")?.addEventListener("click", () => setActiveTab("ledger"));
document.getElementById("tabHardwareBtn")?.addEventListener("click", () => setActiveTab("hardware"));
document.getElementById("systemLogPauseBtn").addEventListener("click", toggleSystemLogPause);
document.getElementById("systemLogClearBtn").addEventListener("click", clearSystemLogConsole);
document.getElementById("systemLogMuteBtn").addEventListener("click", toggleSystemLogMute);
document.getElementById("newsModal").addEventListener("click", (event) => {
  if (event.target.id === "newsModal") closeNewsModal();
});

(async () => {
  updateSystemPauseButton();
  updateSystemMuteButton();
  await refreshHealthMode();
  await refreshSelectedMarket();
  selectedChartPair = selectedMarket;
  await refreshMarkets();
  connectBitvavoPriceStream();
  await refreshBalanceCheck();
  await refreshSentiment();
  await refreshBotStatus();
  await refreshActivity();
  await refreshNewsInsights();
  await updateChart(selectedMarket);
  await refreshTradesTable();
  await refreshPerformanceAnalytics();
  initHintPortals();
  connectSystemStatsSocket();
  await refreshBrainLab();
})();

setInterval(() => {
  refreshHealthMode();
  refreshSentiment();
  refreshBotStatus();
  refreshActivity();
  refreshStorageHealth();
  refreshPerformanceAnalytics();
}, 10000);

setInterval(() => {
  refreshChart();
}, 15000);

setInterval(() => {
  refreshTradesTable();
}, 5000);

setInterval(() => {
  refreshNewsInsights();
}, 60000);

setInterval(() => {
  refreshBrainLab();
}, 7000);

setInterval(() => {
  if (activeTab !== "hardware") return;
  refreshSystemLogsSnapshot();
}, 5000);
