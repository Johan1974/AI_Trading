/**
 * Live cockpit log tail: max 100 regels, filters ALL / ERROR / SYSTEM, WS /ws/logs + Redis cockpit_log_line.
 * Geladen op de cockpit-index; monolithische terminal.js gebruikt dezelfde API indien aanwezig.
 */
(function () {
  const MAX = 100;
  let activeFilter = "all";
  /** @type {{ text: string, kind: string, repeat: number }[]} */
  let lines = [];
  let ws = null;
  let wsTimer = null;
  let userPinned = false;

  function consoleEl() {
    return document.getElementById("cockpitLogConsole");
  }

  /** Zelfde payload zonder leidende timestamp / bracket-level → dedupe-key. */
  function messageKey(line) {
    let s = String(line || "");
    s = s.replace(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\s+/i, "");
    s = s.replace(/^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+/, "");
    while (/^\[[^\]]+\]\s+/.test(s)) s = s.replace(/^\[[^\]]+\]\s+/, "");
    return s.trim();
  }

  window.__terminalLogMessageKey = messageKey;

  function classify(line) {
    const s = String(line || "");
    const u = s.toUpperCase();
    if (u.includes("EXCEPTION") || u.includes("TRACEBACK") || u.includes("CRITICAL") || /\bERROR\b/.test(s)) return "error";
    if (u.includes("WARNING") || (u.includes("WARN") && !u.includes("[WARN-SUPPRESSED]"))) return "warn";
    if (u.includes("[AI-ENGINE]") || u.includes("[RL-BRAIN]")) return "ai";
    if (u.includes("[NEWS]")) return "news";
    if (
      u.includes("[REDIS SUCCESS]") ||
      (u.includes("[MEM-TRACE]") && !u.includes("WARN") && !u.includes("ERROR")) ||
      (u.includes("[DATA-INTEGRITY]") && u.includes("[OK]")) ||
      u.includes("[HB][STATUS]")
    ) return "noise";
    if (/\bINFO\b/.test(s)) return "info";
    return "system";
  }

  function passesFilter(kind) {
    if (activeFilter === "all") return true;
    if (activeFilter === "error") return kind === "error" || kind === "warn";
    if (activeFilter === "signal") return kind === "ai" || kind === "news" || kind === "error" || kind === "warn";
    if (activeFilter === "quiet") return kind !== "noise";
    return true;
  }

  function splitTimestampRest(text) {
    const s = String(text);
    let m = s.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+(.*)$/i);
    if (m) return { ts: m[1], rest: m[2] };
    m = s.match(/^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(.*)$/);
    if (m) return { ts: m[1], rest: m[2] };
    return { ts: "", rest: s };
  }

  function appendRowEl(el, row) {
    if (!passesFilter(row.kind)) return;
    const div = document.createElement("div");
    div.className = `tail-row tail-row--${row.kind}`;
    const { ts, rest } = splitTimestampRest(row.text);
    if (ts) {
      const tsSpan = document.createElement("span");
      tsSpan.className = "tail-ts";
      tsSpan.textContent = `${ts} `;
      div.appendChild(tsSpan);
    }
    if (row.repeat > 1) {
      const badge = document.createElement("span");
      badge.className = "tail-dup-badge";
      badge.setAttribute("aria-label", `${row.repeat} keer herhaald`);
      badge.textContent = `[×${row.repeat}]`;
      div.appendChild(badge);
    }
    const msg = document.createElement("span");
    msg.className = "tail-msg";
    msg.textContent = ts ? rest : row.text;
    div.appendChild(msg);
    el.appendChild(div);
  }

  function flashLastRow(el) {
    const rows = el.querySelectorAll(".tail-row");
    const last = rows[rows.length - 1];
    if (!last) return;
    last.classList.remove("tail-row--flash");
    void last.offsetWidth;
    last.classList.add("tail-row--flash");
    window.setTimeout(() => {
      try {
        last.classList.remove("tail-row--flash");
      } catch (_) {}
    }, 480);
  }

  /**
   * @param {{ dedupeOnly?: boolean }} [opts]
   */
  function render(opts) {
    const el = consoleEl();
    if (!el) return;
    const dedupeOnly = Boolean(opts && opts.dedupeOnly);
    const prevScrollTop = el.scrollTop;
    const nearBottom = el.scrollHeight - (el.scrollTop + el.clientHeight) < 48;
    const stick = !userPinned || nearBottom;
    el.innerHTML = "";
    for (const row of lines) {
      appendRowEl(el, row);
    }
    if (dedupeOnly) {
      el.scrollTop = prevScrollTop;
    } else if (stick) {
      el.scrollTop = el.scrollHeight;
    }
    if (dedupeOnly) flashLastRow(el);
  }

  function push(rawLine) {
    const text = String(rawLine || "");
    const kind = classify(text);
    const key = messageKey(text);
    const last = lines[lines.length - 1];
    if (last && key.length > 0 && messageKey(last.text) === key) {
      last.repeat = (last.repeat || 1) + 1;
      if (kind === "error") last.kind = "error";
      render({ dedupeOnly: true });
      return;
    }
    lines.push({ text, kind, repeat: 1 });
    while (lines.length > MAX) lines.shift();
    render({ dedupeOnly: false });
  }

  function bindScrollGuard() {
    const el = consoleEl();
    if (!el || el.dataset.tailScroll === "1") return;
    el.dataset.tailScroll = "1";
    el.addEventListener("scroll", () => {
      const nearBottom = el.scrollHeight - (el.scrollTop + el.clientHeight) < 48;
      userPinned = !nearBottom;
    });
  }

  function bindFilterButtons() {
    document.querySelectorAll("[data-cockpit-log-filter]").forEach((btn) => {
      if (btn.dataset.tailBound === "1") return;
      btn.dataset.tailBound = "1";
      btn.addEventListener("click", () => {
        activeFilter = btn.getAttribute("data-cockpit-log-filter") || "all";
        document.querySelectorAll("[data-cockpit-log-filter]").forEach((b) => {
          b.classList.toggle("is-active", b === btn);
        });
        render();
      });
    });
    bindScrollGuard();
  }

  function logsTabActiveDom() {
    const tab = document.getElementById("tab-logs");
    if (!tab) return false;
    const hidden = tab.classList.contains("hidden") || tab.style.display === "none";
    return !hidden;
  }

  function connectWs() {
    const ac =
      window.AppCore && window.AppCore.state && window.AppCore.state.activeTab === "logs";
    if (!ac && !logsTabActiveDom()) return;
    if (ws && [WebSocket.OPEN, WebSocket.CONNECTING].includes(ws.readyState)) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const st = document.getElementById("cockpitLogWsStatus");
    ws = new WebSocket(`${protocol}://${window.location.host}/ws/logs`);
    ws.onopen = () => {
      if (st) {
        st.textContent = "Live";
        st.className = "cockpit-log-ws-status cockpit-log-ws-status--live genesis-mono-strong";
      }
    };
    ws.onmessage = (ev) => push(String(ev.data || ""));
    ws.onclose = () => {
      ws = null;
      if (st) {
        st.textContent = "…";
        st.className = "cockpit-log-ws-status cockpit-log-ws-status--off genesis-mono-strong";
      }
      if (
        (window.AppCore && window.AppCore.state && window.AppCore.state.activeTab === "logs") ||
        logsTabActiveDom()
      ) {
        wsTimer = window.setTimeout(connectWs, 2500);
      }
    };
    ws.onerror = () => {
      try {
        ws.close();
      } catch (_) {}
    };
  }

  function disconnectWs() {
    if (wsTimer) {
      clearTimeout(wsTimer);
      wsTimer = null;
    }
    if (ws) {
      try {
        ws.close();
      } catch (_) {}
      ws = null;
    }
    const st = document.getElementById("cockpitLogWsStatus");
    if (st) {
      st.textContent = "—";
      st.className = "cockpit-log-ws-status cockpit-log-ws-status--off genesis-mono-strong";
    }
  }

  /** Eenmalige catch-up vanuit /activity (max 100) als buffer nog leeg is. */
  function bootstrapFromServerTail(tail) {
    if (!Array.isArray(tail) || !tail.length || lines.length > 0) return;
    for (const ln of tail.slice(-MAX)) {
      const text = String(ln || "");
      const kind = classify(text);
      const key = messageKey(text);
      const lst = lines[lines.length - 1];
      if (lst && key.length > 0 && messageKey(lst.text) === key) {
        lst.repeat = (lst.repeat || 1) + 1;
        if (kind === "error") lst.kind = "error";
      } else {
        lines.push({ text, kind, repeat: 1 });
      }
    }
    while (lines.length > MAX) lines.shift();
    render({ dedupeOnly: false });
  }

  window.TerminalLiveTail = {
    push,
    bootstrapFromServerTail,
    bindFilterButtons,
    connectWs,
    disconnectWs,
    clear() {
      lines = [];
      userPinned = false;
      const el = consoleEl();
      if (el) el.innerHTML = "";
    },
    onLogsTabShown() {
      bindFilterButtons();
      userPinned = false;
      connectWs();
    },
    onLogsTabHidden() {
      disconnectWs();
    },
  };

  document.addEventListener("DOMContentLoaded", () => {
    window.TerminalLiveTail.bindFilterButtons();
  });
})();

/**
 * `/activity` zonder browser-HTTP-cache + query-bust (saldo/ledger na paper-reset).
 * Cockpit-modules laden na dit bestand — gebruik overal i.p.v. vaste `"/activity"`.
 */
window.buildActivityFetchUrl = function buildActivityFetchUrl() {
  const b =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  return `/activity?_=${encodeURIComponent(b)}`;
};
window.activityFetchInit = Object.freeze({ cache: "no-store", credentials: "same-origin" });

/** Debug: toon active tab vs. inferred incoming symbol (WS/activity filter). */
(function payloadFilterDebugBadgeSetup() {
  function normPair(s) {
    return String(s || "")
      .toUpperCase()
      .replace("/", "-");
  }
  function activeFromDom() {
    const sel = document.getElementById("marketSelect")?.value;
    const ac = window.AppCore && window.AppCore.state ? window.AppCore.state.selectedMarket : null;
    const win = window.selectedMarket;
    return normPair(sel || ac || win || "BTC-EUR");
  }
  window.updatePayloadFilterDebugBadge = function updatePayloadFilterDebugBadge(opts) {
    const el = document.getElementById("payloadFilterDebugBadge");
    if (!el) return;
    const o = opts && typeof opts === "object" ? opts : {};
    const active = o.active != null && o.active !== "" ? normPair(o.active) : activeFromDom();
    const incRaw = o.incoming;
    const incoming = incRaw == null || incRaw === "" ? "—" : normPair(incRaw);
    const dropped = Boolean(o.dropped);
    const src = o.src != null ? String(o.src) : "—";
    el.textContent = `active=${active} | incoming=${incoming} | dropped=${dropped ? "true" : "false"} | src=${src}`;
    el.setAttribute(
      "aria-label",
      `Payload filter: actief ${active}, binnenkomend ${incoming}, gefilterd ${dropped ? "ja" : "nee"}, bron ${src}`
    );
    el.classList.toggle("bv-payload-filter-debug--dropped", dropped);
  };
})();
