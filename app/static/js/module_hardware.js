/**
 * Hardware-tab: log-snapshot + live tail (terminal.js wordt niet geladen in productie).
 */
window.ModuleHardware = {
  logsSocket: null,
  statsSocket: null,
  logsReconnectTimer: null,
  statsReconnectTimer: null,
  pollTimer: null,
  /** Ruwe logregels (gefilterd pas bij render). */
  rawLines: [],
  crashRawLines: [],
  MAX_BUFFER: 3000,
  /** Max zichtbare regels na filter (performance). */
  MAX_RENDER: 1200,
  paused: false,
  muted: false,
  buffer: [],
  _redrawPending: false,
  _searchDebounce: null,

  _classify(line) {
    const t = String(line || "").toUpperCase();
    if (t.includes("[RL-BRAIN]")) return "log-rl-brain";
    if (
      t.includes("TRACEBACK") ||
      t.includes("EXCEPTION") ||
      t.includes("CRITICAL") ||
      /\bERROR\b/.test(String(line || ""))
    ) {
      return "log-error";
    }
    if (t.includes("WARNING") || t.includes("WARN")) return "log-warning";
    if (t.includes("SUCCESS")) return "log-success";
    return "log-info";
  },

  /** Soort regel voor filter-dropdown (niet hetzelfde als CSS-class). */
  _lineKind(line) {
    const t = String(line || "").toUpperCase();
    if (t.includes("[RL-BRAIN]")) return "rl";
    if (
      t.includes("TRACEBACK") ||
      t.includes("EXCEPTION") ||
      t.includes("CRITICAL") ||
      /\bERROR\b/.test(String(line || ""))
    ) {
      return "error";
    }
    if (t.includes("WARNING") || t.includes("WARN")) return "warn";
    return "other";
  },

  _filterMode() {
    return document.getElementById("systemLogFilterSelect")?.value || "all";
  },

  _searchNeedle() {
    return (document.getElementById("systemLogSearchInput")?.value || "").trim().toLowerCase();
  },

  _matchesFilter(line, mode, needle) {
    const k = this._lineKind(line);
    if (needle && !String(line || "").toLowerCase().includes(needle)) return false;
    if (mode === "all") return true;
    if (mode === "error") return k === "error";
    if (mode === "warn_plus") return k === "error" || k === "warn";
    if (mode === "rl") return k === "rl";
    return true;
  },

  _trimRawLines() {
    while (this.rawLines.length > this.MAX_BUFFER) this.rawLines.shift();
  },

  _trimCrashLines() {
    while (this.crashRawLines.length > this.MAX_BUFFER) this.crashRawLines.shift();
  },

  _scheduleRedraw() {
    if (this._redrawPending) return;
    this._redrawPending = true;
    requestAnimationFrame(() => {
      this._redrawPending = false;
      this._redrawConsole();
    });
  },

  _redrawConsole() {
    const consoleEl = document.getElementById("systemLogConsole");
    if (!consoleEl) return;
    const mode = this._filterMode();
    const needle = this._searchNeedle();
    const nearBottom =
      consoleEl.scrollHeight - (consoleEl.scrollTop + consoleEl.clientHeight) < 48;
    consoleEl.innerHTML = "";
    let shown = 0;
    for (let i = 0; i < this.rawLines.length && shown < this.MAX_RENDER; i++) {
      const line = this.rawLines[i];
      if (!this._matchesFilter(line, mode, needle)) continue;
      const div = document.createElement("div");
      div.className = `system-log-line ${this._classify(line)}`;
      div.textContent = this._formatLine(line);
      consoleEl.appendChild(div);
      shown++;
    }
    if (nearBottom || shown === 0) consoleEl.scrollTop = consoleEl.scrollHeight;
  },

  _redrawCrashConsole() {
    const consoleEl = document.getElementById("crashLogConsole");
    if (!consoleEl) return;
    const nearBottom =
      consoleEl.scrollHeight - (consoleEl.scrollTop + consoleEl.clientHeight) < 48;
    consoleEl.innerHTML = "";
    let shown = 0;
    for (let i = 0; i < this.crashRawLines.length && shown < this.MAX_RENDER; i++) {
      const line = this.crashRawLines[i];
      const div = document.createElement("div");
      div.className = `system-log-line ${this._classify(line)}`;
      div.textContent = this._formatLine(line);
      consoleEl.appendChild(div);
      shown++;
    }
    if (nearBottom || shown === 0) consoleEl.scrollTop = consoleEl.scrollHeight;
  },

  _formatLine(line) {
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
    text = text.replace(/\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\b/g, (raw) => {
      const d = new Date(raw);
      return Number.isFinite(d.getTime()) ? fmt.format(d).replace(",", "") : raw;
    });
    return text;
  },

  _recordLine(line) {
    if (this.muted) return;
    this.rawLines.push(line);
    this._trimRawLines();
    this._scheduleRedraw();
  },

  clearConsole() {
    this.rawLines = [];
    this.buffer = [];
    const el = document.getElementById("systemLogConsole");
    if (el) el.innerHTML = "";
  },

  async copyCrashLog() {
    const btn = document.getElementById("crashLogCopyBtn");
    const payload = (this.crashRawLines || []).join("\n").trim();
    if (!payload) {
      if (btn) btn.textContent = "Crash log leeg";
      setTimeout(() => {
        if (btn) btn.textContent = "Copy crash log";
      }, 1200);
      return;
    }
    try {
      await navigator.clipboard.writeText(payload);
      if (btn) btn.textContent = "Gekopieerd";
    } catch (_) {
      // Fallback voor oudere omgevingen
      const ta = document.createElement("textarea");
      ta.value = payload;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      if (btn) btn.textContent = "Gekopieerd";
    }
    setTimeout(() => {
      if (btn) btn.textContent = "Copy crash log";
    }, 1200);
  },

  togglePause() {
    this.paused = !this.paused;
    const btn = document.getElementById("systemLogPauseBtn");
    if (btn) {
      btn.textContent = this.paused ? "Resume" : "Pause";
      btn.classList.toggle("active", this.paused);
    }
    if (!this.paused && this.buffer.length) {
      for (const ln of this.buffer) {
        if (!this.muted) this.rawLines.push(ln);
      }
      this.buffer = [];
      this._trimRawLines();
      this._redrawConsole();
    }
  },

  toggleMute() {
    this.muted = !this.muted;
    const btn = document.getElementById("systemLogMuteBtn");
    if (btn) {
      btn.textContent = this.muted ? "Unmute" : "Mute";
      btn.classList.toggle("active", this.muted);
    }
  },

  async refreshSnapshot() {
    if (window.AppCore?.state?.activeTab !== "hardware") return;
    const toggle = document.getElementById("systemLogAutoRefresh");
    if (toggle && !toggle.checked) return;
    try {
      const res = await fetch("/api/v1/system/logs?limit=200");
      const data = res.ok ? await res.json().catch(() => ({})) : {};
      if (!res.ok) {
        this.rawLines = [`[Hardware] /api/v1/system/logs HTTP ${res.status}`];
        this._redrawConsole();
        return;
      }
      const lines = Array.isArray(data.lines) ? data.lines : [];
      this.rawLines = lines.length ? lines.slice(-this.MAX_BUFFER) : [];
      this._trimRawLines();
      if (!lines.length) this.rawLines.push("[Hardware] Geen logregels (worker/portal nog geen schrijf naar volume).");
      this._redrawConsole();
    } catch (e) {
      this.rawLines = [`[Hardware] Snapshot: ${String(e)}`];
      this._redrawConsole();
    }
  },

  async refreshCrashSnapshot() {
    if (window.AppCore?.state?.activeTab !== "hardware") return;
    const toggle = document.getElementById("systemLogAutoRefresh");
    if (toggle && !toggle.checked) return;
    try {
      const res = await fetch("/api/v1/system/crash-log?limit=200");
      const data = res.ok ? await res.json().catch(() => ({})) : {};
      if (!res.ok) {
        this.crashRawLines = [`[Hardware] /api/v1/system/crash-log HTTP ${res.status}`];
        this._redrawCrashConsole();
        return;
      }
      const lines = Array.isArray(data.lines) ? data.lines : [];
      this.crashRawLines = lines.length ? lines.slice(-this.MAX_BUFFER) : [];
      this._trimCrashLines();
      if (!lines.length) this.crashRawLines.push("[Hardware] Geen crash-log regels.");
      this._redrawCrashConsole();
    } catch (e) {
      this.crashRawLines = [`[Hardware] Crash snapshot: ${String(e)}`];
      this._redrawCrashConsole();
    }
  },

  connectLogsWs() {
    if (window.AppCore?.state?.activeTab !== "hardware") return;
    if (this.logsSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(this.logsSocket.readyState)) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const status = document.getElementById("systemLogStatus");
    this.logsSocket = new WebSocket(`${protocol}://${window.location.host}/ws/logs`);
    this.logsSocket.onopen = () => {
      if (status) {
        status.textContent = "Live";
        status.className = "status-connected genesis-mono-strong";
      }
    };
    this.logsSocket.onmessage = (ev) => {
      const incoming = String(ev.data || "");
      if (this.paused) {
        this.buffer.push(incoming);
        if (this.buffer.length > 1000) this.buffer = this.buffer.slice(-1000);
        return;
      }
      this._recordLine(incoming);
    };
    this.logsSocket.onclose = () => {
      this.logsSocket = null;
      if (status) {
        status.textContent = "Reconnecting...";
        status.className = "status-disconnected genesis-mono-strong";
      }
      if (window.AppCore?.state?.activeTab === "hardware") {
        this.logsReconnectTimer = setTimeout(() => this.connectLogsWs(), 2000);
      }
    };
    this.logsSocket.onerror = () => {
      try {
        this.logsSocket.close();
      } catch (_) {}
    };
  },

  _normalizeStatsPayload(raw) {
    let d = raw;
    if (typeof raw === "string") {
      try {
        d = JSON.parse(raw);
      } catch (_) {
        return null;
      }
    }
    return d && typeof d === "object" ? d : null;
  },

  _applySystemStatsPayload(data) {
    if (!data || data.__ws === "hb" || data.topic !== "system_stats") return;
    const cpu = Math.max(0, Math.min(100, Number(data.cpu_pct) || 0));
    const ramPct = Math.max(0, Math.min(100, Number(data.ram_pct) || 0));
    const gpuSm = Math.max(0, Math.min(100, Number(data.gpu_util_pct) || 0));
    const gpuEffIn = data.gpu_util_effective;
    const gpuEff = Math.max(0, Math.min(100, Number(gpuEffIn)));
    let gpu = gpuEffIn !== undefined && gpuEffIn !== null && Number.isFinite(gpuEff) ? gpuEff : gpuSm;
    if (data.gpu_ok && (!Number.isFinite(gpu) || gpu < 1)) gpu = 1;
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
  },

  connectStatsWs() {
    if (window.AppCore?.state?.activeTab !== "hardware") return;
    if (this.statsSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(this.statsSocket.readyState)) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    this.statsSocket = new WebSocket(`${protocol}://${window.location.host}/ws/system-stats`);
    this.statsSocket.onmessage = (ev) => {
      try {
        const data = this._normalizeStatsPayload(ev.data);
        if (!data) return;
        if (data.__ws === "hb") {
          try {
            this.statsSocket.send("hb_ack");
          } catch (_) {}
          return;
        }
        this._applySystemStatsPayload(data);
      } catch (_) {}
    };
    this.statsSocket.onclose = () => {
      this.statsSocket = null;
      if (window.AppCore?.state?.activeTab === "hardware") {
        this.statsReconnectTimer = setTimeout(() => this.connectStatsWs(), 4000);
      }
    };
    this.statsSocket.onerror = () => {
      try {
        this.statsSocket.close();
      } catch (_) {}
    };
  },

  _stopLogsWs() {
    if (this.logsReconnectTimer) {
      clearTimeout(this.logsReconnectTimer);
      this.logsReconnectTimer = null;
    }
    if (this.logsSocket) {
      try {
        this.logsSocket.close();
      } catch (_) {}
      this.logsSocket = null;
    }
  },

  _stopStatsWs() {
    if (this.statsReconnectTimer) {
      clearTimeout(this.statsReconnectTimer);
      this.statsReconnectTimer = null;
    }
    if (this.statsSocket) {
      try {
        this.statsSocket.close();
      } catch (_) {}
      this.statsSocket = null;
    }
  },

  onActivate() {
    if (typeof window.switchTab === "function") return;
    void this.refreshSnapshot();
    void this.refreshCrashSnapshot();
    this.connectLogsWs();
    this.connectStatsWs();
    if (this.pollTimer) clearInterval(this.pollTimer);
    this.pollTimer = setInterval(() => {
      void this.refreshSnapshot();
      void this.refreshCrashSnapshot();
    }, 6000);
    window.setTimeout(() => {
      void this.refreshSnapshot();
      void this.refreshCrashSnapshot();
    }, 120);
    window.setTimeout(() => {
      void this.refreshSnapshot();
      void this.refreshCrashSnapshot();
    }, 500);
  },

  stop() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    this._stopLogsWs();
    this._stopStatsWs();
    const status = document.getElementById("systemLogStatus");
    if (status) {
      status.textContent = "Disconnected";
      status.className = "status-disconnected genesis-mono-strong";
    }
  },

  bindControls() {
    if (typeof window.switchTab === "function") return;
    document.getElementById("systemLogPauseBtn")?.addEventListener("click", () => this.togglePause());
    document.getElementById("systemLogClearBtn")?.addEventListener("click", () => this.clearConsole());
    document.getElementById("crashLogCopyBtn")?.addEventListener("click", () => void this.copyCrashLog());
    document.getElementById("systemLogMuteBtn")?.addEventListener("click", () => this.toggleMute());
    document.getElementById("systemLogFilterSelect")?.addEventListener("change", () => this._redrawConsole());
    const searchEl = document.getElementById("systemLogSearchInput");
    if (searchEl) {
      searchEl.addEventListener("input", () => {
        if (this._searchDebounce) clearTimeout(this._searchDebounce);
        this._searchDebounce = setTimeout(() => {
          this._searchDebounce = null;
          this._redrawConsole();
        }, 200);
      });
    }
  },
};

document.addEventListener("DOMContentLoaded", () => {
  if (typeof window.switchTab === "function") return;
  window.ModuleHardware.bindControls();
});
