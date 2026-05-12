/**
 * Live cockpit feeds zonder terminal.js: /activity poll + /ws/trading-updates + Elite-8 bar + ticker.
 */
(function () {
  if (typeof window.switchTab === "function") return;

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

  function escapeHtmlText(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function fetchActivityNoCache() {
    const url =
      typeof window.buildActivityFetchUrl === "function" ? window.buildActivityFetchUrl() : "/activity";
    const init = window.activityFetchInit || { cache: "no-store", credentials: "same-origin" };
    return fetch(url, init);
  }

  function setClassText(selector, text) {
    document.querySelectorAll(selector).forEach((el) => {
      el.textContent = text;
    });
  }

  function baseAccentForMarket(market) {
    const raw = String(market || "").toUpperCase();
    const base = raw.includes("-") ? raw.split("-")[0] : raw;
    return ELITE8_COIN_ACCENTS[base] || "#00ff88";
  }

  function selectedMarketU() {
    const sel = document.getElementById("marketSelect")?.value;
    const ac = window.AppCore?.state?.selectedMarket;
    // DOM-waarde wint: voorkomt stale AppCore (server-selected market) vs. user dropdown.
    return String(sel || ac || "BTC-EUR").toUpperCase();
  }

  function activeTabSymbolU() {
    return normalizePairKey(selectedMarketU());
  }

  function inferIncomingSymbol(raw) {
    if (!raw || typeof raw !== "object") return "";
    const direct = raw.symbol || raw.market || raw.ticker;
    if (direct) return normalizePairKey(direct);
    const lp = raw.last_prediction && typeof raw.last_prediction === "object" ? raw.last_prediction : null;
    if (lp && lp.ticker) return normalizePairKey(lp.ticker);
    const lo = raw.last_order && typeof raw.last_order === "object" ? raw.last_order : null;
    const ord = lo && lo.order && typeof lo.order === "object" ? lo.order : null;
    if (ord && (ord.ticker || ord.market)) return normalizePairKey(ord.ticker || ord.market);
    const ap = raw.ai_action_probs && typeof raw.ai_action_probs === "object" ? raw.ai_action_probs : null;
    if (ap && (ap.market || ap.ticker || ap.symbol)) return normalizePairKey(ap.market || ap.ticker || ap.symbol);
    return "";
  }

  function payloadHasActiveSymbolData(raw, activeU) {
    if (!raw || typeof raw !== "object") return false;
    const multi = raw.rl_multi_decisions && typeof raw.rl_multi_decisions === "object" ? raw.rl_multi_decisions : null;
    if (multi) {
      if (multi[activeU] && typeof multi[activeU] === "object") return true;
      for (const k of Object.keys(multi)) {
        if (normalizePairKey(k) === activeU && multi[k] && typeof multi[k] === "object") return true;
      }
    }
    const hints = raw.worker_calc_hints_by_market && typeof raw.worker_calc_hints_by_market === "object" ? raw.worker_calc_hints_by_market : null;
    if (hints) {
      if (Array.isArray(hints[activeU]) && hints[activeU].length) return true;
      for (const k of Object.keys(hints)) {
        if (normalizePairKey(k) === activeU && Array.isArray(hints[k]) && hints[k].length) return true;
      }
    }
    const ap = raw.ai_action_probs && typeof raw.ai_action_probs === "object" ? raw.ai_action_probs : null;
    if (ap) {
      const apU = normalizePairKey(ap.market || ap.ticker || ap.symbol || "");
      if (apU && apU === activeU) return true;
    }
    return false;
  }

  function shouldAcceptIncomingPayload(raw) {
    const activeU = activeTabSymbolU();
    const incomingU = inferIncomingSymbol(raw);
    if (!incomingU || incomingU === activeU) return true;
    return payloadHasActiveSymbolData(raw, activeU);
  }

  function normalizePairKey(m) {
    return String(m || "")
      .toUpperCase()
      .replace("/", "-");
  }

  function syncElite8AssetToolbarSelection() {
    const root = document.getElementById("elite8AiStatusBar");
    const sm = normalizePairKey(selectedMarketU());
    if (root) {
      root.querySelectorAll("button.elite8-ai-pill[data-market]").forEach((btn) => {
        const mk = normalizePairKey(btn.getAttribute("data-market"));
        const on = mk === sm;
        btn.classList.toggle("active-asset", on);
        btn.setAttribute("aria-pressed", on ? "true" : "false");
      });
    }
    const scan = document.getElementById("scannerTickerBar");
    if (scan) {
      scan.querySelectorAll("button.scanner-ticker-badge[data-market]").forEach((btn) => {
        btn.classList.toggle("active-asset", normalizePairKey(btn.getAttribute("data-market")) === sm);
      });
    }
  }

  window.syncElite8AssetToolbarSelection = syncElite8AssetToolbarSelection;

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

  /**
   * UI-allocatie: (posQty * actuele prijs) / totale balans * 100, afgerond op 2 decimalen.
   * Valt terug op server ``allocation_snapshot.lines`` als equity/qty ontbreekt.
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
        for (let j = 0; j < am.length; j++) {
          const row = am[j];
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

  /** Eén tijdstempel + tekst: strip herhaalde ISO [LEVEL] prefixes (Python logger + dubbele injectie). */
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

  function decisionMatchesMarket(decision, market) {
    if (!decision || typeof decision !== "object") return false;
    const m = normalizePairKey(market);
    const dk = normalizePairKey(decision.market || decision.ticker || decision.symbol || "");
    return !!m && !!dk && dk === m;
  }

  function renderBvThoughtFeed(data) {
    const root = document.getElementById("bvAiThoughtFeed");
    if (!root) return;
    const activeMarket = selectedMarketU();
    const activeU = normalizePairKey(activeMarket);
    const chunks = [];
    const hintsByMarket =
      data && data.worker_calc_hints_by_market && typeof data.worker_calc_hints_by_market === "object"
        ? data.worker_calc_hints_by_market
        : null;
    let hintsRaw = [];
    if (hintsByMarket) {
      hintsRaw = hintsByMarket[activeMarket] || hintsByMarket[activeU] || [];
      if (!Array.isArray(hintsRaw)) {
        hintsRaw = [];
        for (const k of Object.keys(hintsByMarket)) {
          if (normalizePairKey(k) === activeU && Array.isArray(hintsByMarket[k])) {
            hintsRaw = hintsByMarket[k];
            break;
          }
        }
      }
    }
    if (!Array.isArray(hintsRaw) || !hintsRaw.length) {
      const gh = Array.isArray(data && data.worker_calc_hints) ? data.worker_calc_hints : [];
      hintsRaw = gh.filter((x) => {
        const t = String(x || "").toUpperCase();
        return t.includes(activeU) || t.includes(activeU.replace(/-/g, "/"));
      });
    }
    hintsRaw
      .map((x) => String(x || "").trim())
      .filter((x) => x.length > 4)
      .slice(-6)
      .forEach((h) => chunks.push(h));
    let rld = null;
    const multi = data && data.rl_multi_decisions && typeof data.rl_multi_decisions === "object" ? data.rl_multi_decisions : null;
    if (multi) {
      rld = multi[activeMarket] || multi[activeU] || null;
      if (!rld) {
        for (const k of Object.keys(multi)) {
          if (normalizePairKey(k) === activeU) {
            rld = multi[k];
            break;
          }
        }
      }
    }
    if (!rld) rld = data && data.rl_last_decision;
    if (rld && typeof rld === "object") {
      const dk = normalizePairKey(rld.market || rld.ticker || rld.symbol || "");
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
      const scope = normalizePairKey(data.market || data.ticker || "");
      if (!scope || scope === activeU) {
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
      const tk = normalizePairKey(lp.ticker || "");
      if (sig && tk && tk === activeU) chunks.push(`Signaal ${sig} · ${tk}`);
    }
    const show = chunks.slice(-10);
    const esc = (t) =>
      String(t || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
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
        const side = esc(bvAiTradeSideGlyph(parts.text));
        return `<div class="bv-ai-trade-row ${rowCls}${hlCls}"><span class="bv-ai-trade-ts">${esc(ts)}</span><span class="bv-ai-trade-txt">${esc(parts.text)}</span><span class="bv-ai-trade-side">${side}</span></div>`;
      })
      .join("");
    if (typeof window.__rlHeartbeatRefresh === "function") window.__rlHeartbeatRefresh();
  }

  function normalizeProbToPercent(x) {
    const n = Number(x);
    if (!Number.isFinite(n) || n < 0) return null;
    return n <= 1.0 + 1e-9 ? n * 100.0 : Math.min(100, n);
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
    const elB = document.getElementById("hpAiProbBuy");
    const elH = document.getElementById("hpAiProbHold");
    const elS = document.getElementById("hpAiProbSell");
    const barB = document.getElementById("hpAiProbBuyBar");
    const barH = document.getElementById("hpAiProbHoldBar");
    const barS = document.getElementById("hpAiProbSellBar");
    if (!elB || !elH || !elS) return;
    const setRow = (valEl, barEl, raw) => {
      const p = normalizeProbToPercent(raw);
      if (valEl) valEl.textContent = p == null ? "—" : `${p.toFixed(1)}%`;
      if (barEl) barEl.style.width = p == null ? "0%" : `${Math.min(100, Math.max(0, p)).toFixed(2)}%`;
    };
    const activeMarket = selectedMarketU();
    const activeU = normalizePairKey(activeMarket);

    let dMulti = null;
    const multi = data && data.rl_multi_decisions && typeof data.rl_multi_decisions === "object" ? data.rl_multi_decisions : null;
    if (multi) {
      dMulti = multi[activeMarket] || multi[activeU] || null;
      if (!dMulti) {
        for (const k of Object.keys(multi)) {
          if (normalizePairKey(k) === activeU) {
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
      if (hasMulti && (taggedOk || untagged)) {
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
      const apMarket = normalizePairKey(ap.market || ap.ticker || ap.symbol || data?.market || data?.ticker || "");
      if (apMarket && apMarket === activeU) {
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
      const dk = normalizePairKey(d.market || d.ticker || d.symbol || "");
      if (dk && dk !== activeU) return;
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
  }

  let lastLedgerCycleSeq = null;
  let tradingSocket = null;
  let activityPollTimer = null;
  let newsPollTimer = null;
  let lastActivityData = null;
  let elitePredictionsTimer = null;
  let lastEliteMkKey = "";
  let currentPredictionPollMarkets = [];

  function mergePredictionCacheIntoData(data) {
    if (!data || typeof data !== "object") return data;
    const mk = selectedMarketU();
    const cache = (window.__predictionCacheBySymbol && window.__predictionCacheBySymbol[mk]) || null;
    if (!cache || typeof cache !== "object") return data;
    const merged = { ...data };
    if (cache.ai_action_probs && typeof cache.ai_action_probs === "object") {
      merged.ai_action_probs = cache.ai_action_probs;
    }
    if (cache.rl_last_decision && typeof cache.rl_last_decision === "object") {
      merged.rl_last_decision = cache.rl_last_decision;
    }
    if (cache.rl_confidence != null) merged.rl_confidence = cache.rl_confidence;
    return merged;
  }

  function marketsForPredictionPoll(signals) {
    const set = new Set();
    const sel = selectedMarketU();
    if (sel) set.add(sel);
    if (Array.isArray(signals)) {
      for (const s of signals) {
        const m = String((s && s.market) || "").toUpperCase();
        if (m) set.add(m);
      }
    }
    return [...set];
  }

  let elitePredictionsFetchInFlight = false;
  async function fetchPredictionsForMarkets(markets) {
    if (elitePredictionsFetchInFlight) return;
    elitePredictionsFetchInFlight = true;
    window.__predictionCacheBySymbol = window.__predictionCacheBySymbol || {};
    const list = Array.isArray(markets) ? markets : [];
    try {
      for (let i = 0; i < list.length; i += 1) {
        const sym = String(list[i] || "").toUpperCase();
        if (!sym || !sym.includes("-")) continue;
        try {
          const res = await fetch(`/api/v1/predictions?symbol=${encodeURIComponent(sym)}`);
          if (!res.ok) continue;
          const payload = await res.json().catch(() => null);
          if (!payload || payload.error) continue;
          window.__predictionCacheBySymbol[sym] = payload;
          if (window.ModuleLogs && typeof window.ModuleLogs.append === "function") {
            window.ModuleLogs.append("info", `Inference complete for ${sym}`);
          }
        } catch (_) {
          /* ignore */
        }
        await new Promise((r) => setTimeout(r, 220));
      }
    } finally {
      elitePredictionsFetchInFlight = false;
    }
    if (typeof window.__repaintCockpitTerminalFromCache === "function") {
      window.__repaintCockpitTerminalFromCache();
    }
  }

  function scheduleElitePredictionsPoll(signals) {
    const markets = marketsForPredictionPoll(signals);
    const key = markets.slice().sort().join(",");
    if (key === lastEliteMkKey && elitePredictionsTimer) return;
    lastEliteMkKey = key;
    if (elitePredictionsTimer) {
      clearInterval(elitePredictionsTimer);
      elitePredictionsTimer = null;
    }
    currentPredictionPollMarkets = markets;
    if (!markets.length) return;
    elitePredictionsTimer = setInterval(() => {
      void fetchPredictionsForMarkets(currentPredictionPollMarkets);
    }, 12000);
    window.setTimeout(() => void fetchPredictionsForMarkets(currentPredictionPollMarkets), 800);
  }

  function repaintHpTerminalFromCachedActivity() {
    if (!lastActivityData || typeof lastActivityData !== "object") return;
    paintHpTerminalFromActivity(lastActivityData);
  }
  window.__repaintCockpitTerminalFromCache = repaintHpTerminalFromCachedActivity;

  const cockpitLedgerStatusParts = { markets: "Markten: …", ledger: "Ledger: …" };

  function paintCockpitLedgerStatus() {
    const el = document.getElementById("cockpitLedgerStatusText");
    if (!el) return;
    el.textContent = `${cockpitLedgerStatusParts.markets} · ${cockpitLedgerStatusParts.ledger}`;
  }

  function paintSentimentGauge(ls) {
    const arc = document.getElementById("hpGaugeValueArc");
    const lab = document.getElementById("hpGaugeLabel");
    const cf = document.getElementById("hpGaugeConfidence");
    const setBar = (pct) => {
      document.querySelectorAll(".js-sentiment-bar").forEach((el) => {
        el.style.width = `${Math.max(0, Math.min(100, pct))}%`;
      });
    };
    const neutralUi = () => {
      if (arc) arc.setAttribute("stroke-dashoffset", "50");
      if (lab) lab.textContent = "Neutral";
      if (cf) cf.textContent = "—";
      setClassText(".js-sentiment-value", "0.000");
      setBar(50);
    };
    if (!ls || typeof ls !== "object") {
      neutralUi();
      return;
    }
    const raw = ls.sentiment_score;
    const score = raw === null || raw === "" || raw === undefined ? NaN : Number(raw);
    if (!Number.isFinite(score)) {
      neutralUi();
      return;
    }
    const conf = Number(ls.sentiment_confidence ?? 0);
    const norm = Math.max(0, Math.min(1, (score + 1) / 2));
    const isNeutralZero =
      Math.abs(score) < 1e-9 && (!Number.isFinite(conf) || Math.abs(conf) < 1e-9);
    if (arc) arc.setAttribute("stroke-dashoffset", String(100 * (1 - norm)));
    if (lab) lab.textContent = isNeutralZero ? "Neutral" : score.toFixed(3);
    if (cf) cf.textContent = Number.isFinite(conf) && !isNeutralZero ? `conf ${conf.toFixed(3)}` : isNeutralZero ? "—" : "";
    setClassText(".js-sentiment-value", score.toFixed(3));
    setBar(norm * 100);
  }

  function eventsSuggestGpuDisconnected(events) {
    if (!Array.isArray(events)) return false;
    for (let i = 0; i < Math.min(events.length, 60); i++) {
      const ev = events[i];
      const blob = JSON.stringify(ev == null ? {} : ev).toUpperCase();
      if (blob.includes("GPU DISCONNECTED")) return true;
    }
    return false;
  }

  function paintGpuFailCards(data) {
    const ss = data.system_stats && typeof data.system_stats === "object" ? data.system_stats : null;
    const gpuOk = ss && (ss.gpu_ok === true || ss.gk === 1 || ss.gk === "1");
    const fromLogs = eventsSuggestGpuDisconnected(data.events);
    const fail = (Boolean(ss) && !gpuOk) || fromLogs;
    ["bvAiFeedCard", "hpSentimentGaugeCard"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.classList.toggle("gpu-fail", Boolean(fail));
    });
  }

  function paintHpDashTop(data) {
    const p = data.paper_portfolio || {};
    const fmtEur = (v) => {
      const n = Number(v);
      return Number.isFinite(n)
        ? `€${n.toLocaleString("nl-NL", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
        : "—";
    };
    const eqEl = document.getElementById("hpDashProfitEquity");
    const cashEl = document.getElementById("hpDashProfitCash");
    const pnlEl = document.getElementById("hpDashProfitPnl");
    if (eqEl) eqEl.textContent = p.equity != null ? fmtEur(p.equity) : "—";
    if (cashEl) cashEl.textContent = p.cash != null ? `Cash ${fmtEur(p.cash)}` : "Cash —";
    const rpnl = p.realized_pnl_eur ?? p.realized_pnl;
    if (pnlEl) pnlEl.textContent = rpnl != null && rpnl !== "" ? `Realized PnL ${fmtEur(rpnl)}` : "Realized PnL —";

    const st = String(data.bot_status || "—");
    const aiEl = document.getElementById("hpDashAiStatus");
    const sigEl = document.getElementById("hpDashAiSignal");
    if (aiEl) aiEl.textContent = st.toUpperCase();
    const lo = data.last_order?.order || {};
    const sig = String(lo.signal || "—").toUpperCase();
    const tk = String(lo.ticker || lo.market || "").toUpperCase();
    const activeU = activeTabSymbolU();
    if (sigEl) {
      const tkU = normalizePairKey(tk);
      sigEl.textContent = tkU && tkU !== activeU ? "—" : (tk ? `${sig} · ${tk}` : sig);
    }

    const alloc = data.allocation_snapshot || {};
    const sumEl = document.getElementById("hpDashAllocSummary");
    const chips = document.getElementById("hpDashAllocChips");
    if (sumEl) sumEl.textContent = String(alloc.summary || "Allocatie: —");
    if (chips) {
      const lines = allocationDisplayLines(data).slice(0, 10);
      chips.innerHTML = lines
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
      const mkSel = selectedMarketU();
      const allLines = allocationDisplayLines(data);
      const line =
        allLines.find((r) => String((r && r.market) || "").toUpperCase() === mkSel) || allLines[0];
      const w = line ? Number(line.weight_pct) : NaN;
      badgeEl.textContent = Number.isFinite(w) ? `${w.toFixed(2)}%` : "—";
    }
    const leg = document.getElementById("allocatie");
    if (leg && alloc.summary) leg.textContent = String(alloc.summary);
    paintAiPolicyProbsFromPayload(data);
  }

  function paintHwMiniMonitor(ss) {
    const root = document.getElementById("cockpitTerminalHwMini");
    const hud = document.getElementById("hpMiniHud");
    if (!root || !hud) return;
    const cpu = ss && (ss.cpu_pct != null ? ss.cpu_pct : ss.c);
    const gpuOk = ss && (ss.gpu_ok === true || ss.gk === 1 || ss.gk === "1");
    const temp = ss && (ss.gpu_temp_max != null ? ss.gpu_temp_max : ss.gt);
    const cPart = Number.isFinite(Number(cpu)) ? `${Number(cpu).toFixed(0)}%` : "—";
    let gPart = "—";
    if (gpuOk && Number.isFinite(Number(temp)) && Number(temp) > 0) gPart = `${Number(temp).toFixed(0)}°C`;
    else if (gpuOk) gPart = "OK";
    else gPart = "off";
    hud.textContent = `${cPart} · ${gPart}`;
    root.title = `Hardware: ${cPart} CPU-belasting · GPU ${gPart}. Bron: worker system_stats.`;
    root.classList.toggle("hp-hw-mini-monitor--alert", Boolean(ss) && !gpuOk);
  }

  function paintHpTerminalFromActivity(data) {
    if (!data || typeof data !== "object") return;
    const merged = mergePredictionCacheIntoData(data);
    paintHpDashTop(merged);
    const ls = merged.last_scores;
    paintSentimentGauge(ls && typeof ls === "object" ? ls : null);
    paintHwMiniMonitor(merged.system_stats && typeof merged.system_stats === "object" ? merged.system_stats : null);
    paintGpuFailCards(merged);
    renderBvThoughtFeed(merged);
  }

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
        pill.title = `${mk} — AI ${action} (${conf.toFixed(2)})`;
        pill.addEventListener("click", () => {
          const sel = document.getElementById("marketSelect");
          if (!sel || !mk) return;
          const matchOpt = Array.from(sel.options || []).find((o) => normalizePairKey(o.value) === normalizePairKey(mk));
          if (matchOpt) sel.value = matchOpt.value;
          else {
            const opt = document.createElement("option");
            opt.value = mk.includes("-") ? mk : `${mk}-EUR`;
            opt.textContent = opt.value;
            sel.appendChild(opt);
            sel.value = opt.value;
          }
          if (window.AppCore && window.AppCore.state) window.AppCore.state.selectedMarket = sel.value;
          syncElite8AssetToolbarSelection();
          if (window.ModuleTerminal && typeof window.ModuleTerminal.refresh === "function") {
            void window.ModuleTerminal.refresh();
          }
          if (typeof window.__reloadTerminalPredictionChart === "function") {
            window.__reloadTerminalPredictionChart();
          }
          sel.dispatchEvent(new Event("change", { bubbles: true }));
        });
        root.appendChild(pill);
      }
      }
    }
    syncElite8AssetToolbarSelection();
  }

  /** Zelfde logica als portal stats: prijs per geselecteerde markt, met fallback naar `active_markets` in /activity. */
  function paintCockpitPriceFromSnapshot(p, data) {
    if (!p || typeof p !== "object") p = {};
    const mk = selectedMarketU();
    const lpm = p.last_prices_by_market;
    let raw = lpm && typeof lpm === "object" && lpm[mk] != null && lpm[mk] !== "" ? lpm[mk] : null;
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
    const el = document.getElementById("btc-price");
    if (el) el.textContent = `€ ${n.toFixed(2)}`;
    if (typeof window.tickTerminalPredictionChartLivePrice === "function") {
      window.tickTerminalPredictionChartLivePrice(n);
    }
  }

  function applyActivityResponse(data) {
    if (!data || typeof data !== "object") return;
    lastActivityData = data;
    window.__lastActivityPayload = data;
    if (window.TerminalLiveTail && typeof window.TerminalLiveTail.bootstrapFromServerTail === "function") {
      window.TerminalLiveTail.bootstrapFromServerTail(data.cockpit_log_tail);
    }
    window.botMetrics = window.botMetrics || {};

    const p = data.paper_portfolio || {};
    const alloc = data.allocation_snapshot || {};
    const linesForUi = allocationDisplayLines(data);
    window.__marketsInPosition =
      typeof window.buildMarketsInPositionSet === "function"
        ? window.buildMarketsInPositionSet(data)
        : new Set(linesForUi.map((x) => String(x.market || "").toUpperCase()).filter(Boolean));

    const allocRoot = document.getElementById("executiveAllocationSnapshot");
    if (allocRoot) {
      const sum = String(alloc.summary || "Allocatie: —");
      const lines = allocationDisplayLines(data);
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
          : '<p class="executive-allocation-empty">Geen actieve posities.</p>';
      allocRoot.innerHTML = `<p class="executive-allocation-summary">${escapeHtmlText(sum)}</p>` + list;
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
    if (hBal && p.equity != null) hBal.textContent = fmtEur(p.equity);
    setClassText(
      ".js-portfolio-stats-rest",
      `Pos Qty: ${p.position_qty ?? "-"} | Realized PnL: ${p.realized_pnl_eur ?? p.realized_pnl ?? "-"}`
    );
    const lastOrder = data.last_order?.order || {};
    {
      const sig = (lastOrder.signal || "").toUpperCase();
      const posQty = Number(p.position_qty ?? 0);
      const sigDisplay = (sig === "SELL" && posQty <= 0) || sig === "HOLD" || !sig ? "—" : `${sig} ${lastOrder.ticker || ""} ${lastOrder.amount_quote_eur || lastOrder.amount_quote || ""}`.trim();
      setClassText(".js-active-orders", `Signaal: ${sigDisplay}`);
    }

    const fg = data.fear_greed || {};
    const fgMain = document.getElementById("terminalFearGreed");
    const fgSub = document.getElementById("terminalFearGreedSub");
    const raw = fg.fear_greed_value ?? fg.fear_greed_score;
    if (fgMain) {
      fgMain.textContent = raw != null && raw !== "" ? `${Number(raw).toFixed(0)} / 100` : "-";
    }
    if (fgSub) {
      const cls = String(fg.classification || fg.fear_greed_class || "").trim();
      fgSub.textContent = cls ? cls : "";
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

    window.__eliteCriticalStream = Boolean(
      Array.isArray(data.elite_ai_signals) &&
        data.elite_ai_signals.some((s) => s && (s.state === "panic" || s.whale_danger === true))
    );
    if (Array.isArray(data.elite_ai_signals)) renderElite8AiStatusBar(data.elite_ai_signals);
    scheduleElitePredictionsPoll(Array.isArray(data.elite_ai_signals) ? data.elite_ai_signals : []);

    const cyc = data.last_order?.cycle_seq;
    if (cyc != null && cyc !== lastLedgerCycleSeq) {
      lastLedgerCycleSeq = cyc;
    }
    if (Array.isArray(data.active_markets)) {
      cockpitLedgerStatusParts.markets =
        data.active_markets.length === 0 ? "Markten: leeg" : `Markten: ${data.active_markets.length} geladen`;
    }
    if ("trades" in data && Array.isArray(data.trades)) {
      cockpitLedgerStatusParts.ledger =
        data.trades.length === 0 ? "Ledger: 0 trades" : `Ledger: ${data.trades.length} trade(s)`;
      if (window.ModuleLedger && typeof window.ModuleLedger.renderTable === "function") {
        window.ModuleLedger.renderTable({ trades: data.trades, risk_profile: data.risk_profile });
      }
    }
    paintCockpitLedgerStatus();
    paintHpTerminalFromActivity(data);
    paintCockpitPriceFromSnapshot(p, data);
    if (typeof window.syncHeaderPositionLamp === "function") window.syncHeaderPositionLamp();
    if (typeof window.setChartTimeHint === "function") {
      const mkU = typeof selectedMarketU === "function" ? selectedMarketU() : "";
      window.setChartTimeHint({
        market: mkU || data.selected_market || "BTC-EUR",
        updated_at: data.last_engine_tick_utc || (data.last_prediction && data.last_prediction.generated_at) || null,
        predicted_at: (data.last_prediction && data.last_prediction.generated_at) || null,
      });
    }
  }

  function applyTradingRedisPayload(raw) {
    if (!raw || typeof raw !== "object") return;
    if (raw.type === "error" || raw.type === "ping") return;
    const activeUWs = activeTabSymbolU();
    const incWs = inferIncomingSymbol(raw);
    const acceptWs = shouldAcceptIncomingPayload(raw);
    if (typeof window.updatePayloadFilterDebugBadge === "function") {
      window.updatePayloadFilterDebugBadge({
        active: activeUWs,
        incoming: incWs || "",
        dropped: !acceptWs,
        src: "ws",
      });
    }
    if (!acceptWs) return;
    if (raw.type === "cockpit_log_line" && raw.line && window.TerminalLiveTail && typeof window.TerminalLiveTail.push === "function") {
      const ln = String(raw.line);
      const activeU = activeTabSymbolU();
      const hasPair = /[A-Z0-9]+-[A-Z0-9]+/.test(ln.toUpperCase());
      const lineOk = !hasPair || ln.toUpperCase().includes(activeU);
      if (typeof window.updatePayloadFilterDebugBadge === "function") {
        window.updatePayloadFilterDebugBadge({
          active: activeU,
          incoming: incWs || "",
          dropped: !lineOk,
          src: "ws-log",
        });
      }
      if (lineOk) {
        window.TerminalLiveTail.push(ln);
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
      rl_last_decision:
        raw.rl_last_decision && typeof raw.rl_last_decision === "object" ? raw.rl_last_decision : undefined,
      rl_multi_decisions:
        raw.rl_multi_decisions && typeof raw.rl_multi_decisions === "object" ? raw.rl_multi_decisions : undefined,
      ai_action_probs:
        raw.ai_action_probs && typeof raw.ai_action_probs === "object" ? raw.ai_action_probs : undefined,
      worker_calc_hints: Array.isArray(raw.worker_calc_hints) ? raw.worker_calc_hints : undefined,
      worker_calc_hints_by_market:
        raw.worker_calc_hints_by_market && typeof raw.worker_calc_hints_by_market === "object"
          ? raw.worker_calc_hints_by_market
          : undefined,
      active_markets: Array.isArray(raw.active_markets) ? raw.active_markets : undefined,
    };
    if (Array.isArray(raw.trades)) data.trades = raw.trades;
    if (raw.system_stats && typeof raw.system_stats === "object") data.system_stats = raw.system_stats;
    if (!data.last_engine_tick_utc && data.last_prediction && data.last_prediction.generated_at) {
      data.last_engine_tick_utc = data.last_prediction.generated_at;
    }
    applyActivityResponse(data);
  }

  const TRADING_WS_RECONNECT_MS = 5000;
  let tradingSocketReconnectTimer = null;

  function scheduleTradingSocketReconnect() {
    if (tradingSocketReconnectTimer != null) return;
    tradingSocketReconnectTimer = window.setTimeout(() => {
      tradingSocketReconnectTimer = null;
      connectTradingUpdatesSocket();
    }, TRADING_WS_RECONNECT_MS);
  }

  function connectTradingUpdatesSocket() {
    if (tradingSocket && tradingSocket.readyState === WebSocket.CONNECTING) return;
    if (tradingSocket && tradingSocket.readyState === WebSocket.OPEN) return;
    if (tradingSocketReconnectTimer != null) {
      window.clearTimeout(tradingSocketReconnectTimer);
      tradingSocketReconnectTimer = null;
    }
    try {
      if (tradingSocket) {
        try {
          tradingSocket.onclose = null;
          tradingSocket.onerror = null;
          tradingSocket.onmessage = null;
          if (tradingSocket.readyState === WebSocket.OPEN || tradingSocket.readyState === WebSocket.CONNECTING) {
            tradingSocket.close();
          }
        } catch (_) {}
      }
    } catch (_) {}
    tradingSocket = null;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    let ws;
    try {
      ws = new WebSocket(`${protocol}://${window.location.host}/ws/trading-updates`);
    } catch (_e) {
      scheduleTradingSocketReconnect();
      return;
    }
    tradingSocket = ws;
    ws.onopen = () => {
      if (tradingSocketReconnectTimer != null) {
        window.clearTimeout(tradingSocketReconnectTimer);
        tradingSocketReconnectTimer = null;
      }
    };
    ws.onmessage = (event) => {
      try {
        const raw = JSON.parse(String(event.data || "{}"));
        applyTradingRedisPayload(raw);
      } catch (_) {}
    };
    ws.onclose = () => {
      tradingSocket = null;
      scheduleTradingSocketReconnect();
    };
    ws.onerror = () => {
      try {
        ws.close();
      } catch (_) {}
    };
  }

  async function pullActivity() {
    try {
      const res = await fetchActivityNoCache();
      if (!res.ok) return;
      const data = await res.json();
      const activeH = activeTabSymbolU();
      const incH = inferIncomingSymbol(data);
      const acceptH = shouldAcceptIncomingPayload(data);
      if (typeof window.updatePayloadFilterDebugBadge === "function") {
        window.updatePayloadFilterDebugBadge({
          active: activeH,
          incoming: incH || "",
          dropped: !acceptH,
          src: "http",
        });
      }
      if (!acceptH) return;
      applyActivityResponse(data);
    } catch (_) {}
  }

  async function pullNewsIntel() {
    const root = document.getElementById("intelligenceTickerNews");
    if (!root) return;
    try {
      const [newsRes, actRes] = await Promise.all([
        fetch("/api/v1/news/ticker?elite_mix=1"),
        fetchActivityNoCache(),
      ]);
      const news = newsRes.ok ? await newsRes.json().catch(() => []) : [];
      const act = actRes.ok ? await actRes.json().catch(() => ({})) : {};
      const feed = Array.isArray(act.scanner_intel_feed) ? act.scanner_intel_feed.slice(-12) : [];
      const parts = [];
      for (const row of feed) {
        const h = String(row.headline || row.text || row.title || "").slice(0, 80);
        if (h) parts.push(escapeHtmlText(h));
      }
      const arr = Array.isArray(news) ? news : [];
      for (const item of arr.slice(0, 8)) {
        const h = String(item.text || item.title || "").slice(0, 80);
        if (h) parts.push(escapeHtmlText(h));
      }
      root.innerHTML = "";
      if (!parts.length) {
        root.innerHTML = `<span class="cockpit-ticker-empty">Geen intelligence-feed (worker/scanner).</span>`;
        return;
      }
      const wrap = document.createElement("span");
      wrap.className = "cockpit-ticker-marquee-track";
      wrap.innerHTML = parts.map((p) => `<span class="cockpit-ticker-item-main">${p}</span>`).join(" · ");
      root.appendChild(wrap);
    } catch (_) {
      root.innerHTML = `<span class="cockpit-ticker-empty">Intelligence-feed tijdelijk niet beschikbaar.</span>`;
    }
  }

  window.ModuleFeeds = {
    init() {
      if (typeof window.switchTab === "function") return;
      const sel = document.getElementById("marketSelect");
      if (sel && !sel.dataset.hpElite8Sync) {
        sel.dataset.hpElite8Sync = "1";
        sel.addEventListener("change", () => {
          const next = String(sel.value || "BTC-EUR").toUpperCase();
          if (window.AppCore && window.AppCore.state) window.AppCore.state.selectedMarket = next;
          window.selectedMarket = next;
          const pairEl = document.getElementById("headerPairDisplay");
          if (pairEl) pairEl.textContent = next;
          syncElite8AssetToolbarSelection();
          if (window.ModuleTerminal && typeof window.ModuleTerminal.refresh === "function") {
            void window.ModuleTerminal.refresh();
          }
          if (typeof window.__reloadTerminalPredictionChart === "function") {
            window.__reloadTerminalPredictionChart();
          }
          void pullActivity();
          void fetchPredictionsForMarkets([next]);
        });
      }
      connectTradingUpdatesSocket();
      void pullActivity();
      void pullNewsIntel();
      if (activityPollTimer) clearInterval(activityPollTimer);
      activityPollTimer = setInterval(() => void pullActivity(), 8000);
      if (newsPollTimer) clearInterval(newsPollTimer);
      newsPollTimer = setInterval(() => void pullNewsIntel(), 30000);
      requestAnimationFrame(() => syncElite8AssetToolbarSelection());
      document.addEventListener("ai-trading-stats", (ev) => {
        const d = ev && ev.detail;
        if (d && typeof d === "object") {
          paintAiPolicyProbsFromPayload(d);
          renderBvThoughtFeed(d);
          syncElite8AssetToolbarSelection();
        }
      });
    },
  };

  document.addEventListener("DOMContentLoaded", () => {
    if (typeof window.switchTab === "function") return;
    if (window.ModuleFeeds) window.ModuleFeeds.init();
    document.addEventListener("ai-trading-refresh-activity", () => void pullActivity());
  });
})();
