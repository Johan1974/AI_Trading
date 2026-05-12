/**
 * Hint-portal: delegated tooltip engine.
 * Bindt één listener per root-container (event delegation) zodat hints werken
 * voor bestaande én toekomstige .hint-wrap elementen — ook na WS/DOM-updates.
 */
(function () {
  const HINT_PORTAL_ID = "brainHintPortal";

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
    const text = bubbleEl && bubbleEl.textContent ? bubbleEl.textContent.trim() : "";
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
      "z-index:2147483646",
      `max-width:${maxW}px`,
      "background:rgba(0,0,0,0.88)",
      "color:#ffffff",
      "border:1px solid rgba(255,255,255,0.25)",
      "border-radius:6px",
      "padding:10px 12px",
      "font-size:14px",
      "line-height:1.4",
      "backdrop-filter:blur(8px)",
      "-webkit-backdrop-filter:blur(8px)",
      "box-shadow:0 12px 40px rgba(0,0,0,0.55)",
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

  // Event delegation: één listener per root die zowel huidige als toekomstige
  // .hint-wrap elementen afhandelt — geen per-element binding meer nodig.
  function attachDelegatedHints(root) {
    if (!root || root.__hintDelegated) return;
    root.__hintDelegated = true;

    root.addEventListener("mouseover", function (e) {
      const wrap = e.target.closest(".hint-wrap");
      if (!wrap || !root.contains(wrap)) return;
      const bubble = wrap.querySelector(".hint-bubble");
      if (bubble) positionBrainHintPortal(wrap, bubble);
    });

    root.addEventListener("mouseout", function (e) {
      const wrap = e.target.closest(".hint-wrap");
      if (!wrap || !root.contains(wrap)) return;
      if (!wrap.contains(e.relatedTarget)) hideBrainHintPortal();
    });

    root.addEventListener("focusin", function (e) {
      const wrap = e.target.closest(".hint-wrap");
      if (!wrap || !root.contains(wrap)) return;
      const bubble = wrap.querySelector(".hint-bubble");
      if (bubble) positionBrainHintPortal(wrap, bubble);
    });

    root.addEventListener("focusout", function (e) {
      const wrap = e.target.closest(".hint-wrap");
      if (!wrap || !root.contains(wrap)) return;
      if (!wrap.contains(e.relatedTarget)) hideBrainHintPortal();
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
      attachDelegatedHints(root);
    }

    if (!window.__genesisHintScrollBound) {
      window.__genesisHintScrollBound = true;
      window.addEventListener("scroll", hideBrainHintPortal, true);
    }
  }

  window.initHintPortals = initHintPortals;
  window.hideBrainHintPortal = hideBrainHintPortal;

  window.addEventListener("resize", hideBrainHintPortal);

  document.addEventListener("DOMContentLoaded", initHintPortals);
})();
