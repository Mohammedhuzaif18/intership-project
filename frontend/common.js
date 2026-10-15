// Shared by index.html (intake) and results.html (report)
const PulmoScan = (() => {
  const API_BASE = window.location.origin;
  const STORAGE_KEY = 'pulmoscan:last-analysis';

  // Colors follow the categorical palette slots used in the report figures
  const CLASS_COLORS = {
    NORMAL: 'var(--c-normal)',
    PNEUMONIA: 'var(--c-pneumonia)',
    COVID: 'var(--c-covid)',
    TB: 'var(--c-tb)',
  };
  const colorFor = (cls) => CLASS_COLORS[cls] || 'var(--c-other)';

  // Model outputs are never exactly certain (the API rounds tiny values to 0); avoid showing 100.0% / 0.0%
  const fmtPct = (p) => (p >= 0.9995 ? '>99.9%' : p < 0.0005 ? '<0.1%' : `${(p * 100).toFixed(1)}%`);

  let displayNames = {};
  const labelFor = (cls) => displayNames[cls] || cls;
  const setDisplayNames = (names) => { displayNames = { ...displayNames, ...(names || {}) }; };

  async function fetchHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(res.statusText);
    const health = await res.json();
    setDisplayNames(health.display_names);
    return health;
  }

  /** Keeps the header status pill current; calls onHealth with each successful /health reply. */
  function watchStatus(onHealth) {
    const status = document.getElementById('status');
    const text = document.getElementById('statusText');
    const set = (state, msg) => { status.dataset.state = state; text.textContent = msg; };
    const tick = async () => {
      try {
        const h = await fetchHealth();
        if (h.model_loaded) set('ready', `${h.model} ready · ${h.classes.length} classes`);
        else set('waiting', 'API online · no trained model yet');
        if (onHealth) onHealth(h);
      } catch (e) {
        set('offline', 'Model API offline');
      }
    };
    tick();
    setInterval(tick, 15000);
  }

  // The analysis travels from the intake page to the report page in sessionStorage:
  // it never leaves this browser tab and is cleared when the tab closes.
  function saveAnalysis(record) {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(record));
      return true;
    } catch (e) {
      // Quota exceeded: keep everything except the full-size preview image
      try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ ...record, image: { ...record.image, dataUrl: null } }));
        return true;
      } catch (e2) {
        return false;
      }
    }
  }

  function loadAnalysis() {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  return { API_BASE, colorFor, fmtPct, labelFor, setDisplayNames, fetchHealth, watchStatus, saveAnalysis, loadAnalysis };
})();
