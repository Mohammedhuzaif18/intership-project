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

  function createComparisonView(comparison) {
    const root = document.createElement('div');
    root.className = 'comparison-view';
    const summary = document.createElement('section');
    const change = comparison.prediction_change || {};
    summary.className = `comparison-summary comparison-${change.assessment || 'neutral'}`;
    const summaryTitle = document.createElement('h3');
    const titles = {
      possible_improvement_signal: 'Possible improvement signal (model only)',
      possible_new_or_worsening_signal: 'Possible new or worsening signal (model only)',
      no_predicted_category_change_normal: 'No change in the model’s Normal prediction',
      no_predicted_category_change_disease: 'No change in the model’s disease prediction',
      predicted_disease_category_changed: 'The model’s predicted disease category changed',
    };
    summaryTitle.textContent = titles[change.assessment] || 'Model prediction comparison';
    const summaryText = document.createElement('p');
    summaryText.textContent = change.summary || 'The model comparison summary is unavailable.';
    const disclaimer = document.createElement('p');
    disclaimer.className = 'comparison-summary-note';
    disclaimer.textContent = comparison.disclaimer;
    summary.append(summaryTitle, summaryText, disclaimer);
    const columns = document.createElement('div');
    columns.className = 'comparison-grid';

    [comparison.previous, comparison.current].forEach((record, index) => {
      const article = document.createElement('article');
      article.className = 'comparison-record';
      const heading = document.createElement('h3');
      const date = new Date(record.uploaded_at).toLocaleString();
      heading.textContent = `${index === 0 ? 'Previous' : 'Current'} · ${date}`;
      const result = document.createElement('p');
      result.className = 'comparison-prediction';
      result.textContent = `${record.predicted_label} · ${fmtPct(record.confidence)} confidence`;
      const scores = document.createElement('p');
      scores.className = 'comparison-scores';
      scores.textContent = Object.entries(record.class_probabilities || {})
        .map(([name, value]) => `${labelFor(name)} ${fmtPct(value)}`).join(' · ');
      article.append(heading, result, scores);

      const images = document.createElement('div');
      images.className = 'comparison-image-pair';
      [['original', 'X-ray'], ['gradcam', 'Grad-CAM']].forEach(([kind, label]) => {
        const figure = document.createElement('figure');
        const img = document.createElement('img');
        const token = sessionStorage.getItem('pulmoscan:auth-token');
        if (token) {
          fetch(`${API_BASE}${record.image_urls[kind]}`, { headers: { Authorization: `Bearer ${token}` } })
            .then((response) => {
              if (!response.ok) throw new Error(`Image request failed (${response.status}).`);
              return response.blob();
            })
            .then((blob) => { img.src = URL.createObjectURL(blob); })
            .catch(() => { img.alt = `${label} unavailable: sign in as the patient to view this image.`; });
        } else {
          img.alt = `${label} unavailable: sign in as the patient to view this image.`;
        }
        if (token) img.alt = `${label} for ${index === 0 ? 'previous' : 'current'} record`;
        const caption = document.createElement('figcaption');
        caption.textContent = label;
        figure.append(img, caption);
        images.append(figure);
      });
      article.append(images);
      columns.append(article);
    });

    const difference = document.createElement('figure');
    difference.className = 'difference-figure';
    const image = document.createElement('img');
    image.src = comparison.difference_map_base64;
    image.alt = 'OpenCV visual difference map between the selected X-rays';
    const caption = document.createElement('figcaption');
    caption.textContent = `OpenCV difference map · ${comparison.alignment.message}`;
    difference.append(image, caption);
    root.append(summary, columns, difference);
    return root;
  }

  return {
    API_BASE, colorFor, fmtPct, labelFor, setDisplayNames, fetchHealth, watchStatus,
    saveAnalysis, loadAnalysis, createComparisonView,
  };
})();
