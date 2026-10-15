document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);

  const status = $('status');
  const statusText = $('statusText');
  const dropZone = $('dropZone');
  const fileInput = $('fileInput');
  const stage = $('stage');
  const toolbar = $('toolbar');
  const replaceBtn = $('replaceBtn');
  const imgBase = $('imgBase');
  const imgHeat = $('imgHeat');
  const imgBase2 = $('imgBase2');
  const imgHeat2 = $('imgHeat2');
  const viewerFoot = $('viewerFoot');
  const opacityControl = $('opacityControl');
  const opacitySlider = $('opacitySlider');
  const opacityVal = $('opacityVal');
  const fileMeta = $('fileMeta');
  const samples = $('samples');
  const sampleButtons = $('sampleButtons');
  const modeButtons = toolbar.querySelectorAll('[data-mode]');

  const emptyState = $('emptyState');
  const loadingState = $('loadingState');
  const errorState = $('errorState');
  const errorMsg = $('errorMsg');
  const resultState = $('resultState');

  const API_BASE = window.location.origin;

  // Colors follow the categorical palette slots used in the report figures
  const CLASS_COLORS = {
    NORMAL: 'var(--c-normal)',
    PNEUMONIA: 'var(--c-pneumonia)',
    COVID: 'var(--c-covid)',
    TB: 'var(--c-tb)',
  };
  const colorFor = (cls) => CLASS_COLORS[cls] || 'var(--c-other)';

  // Each frame takes the aspect ratio of the image it shows, so any view fills the viewer
  [imgBase, imgBase2].forEach((img) => img.addEventListener('load', () => {
    if (img.naturalWidth && img.naturalHeight) {
      img.parentElement.style.setProperty('--aspect', String(img.naturalWidth / img.naturalHeight));
    }
  }));

  // Model outputs are never exactly certain; avoid showing 100.0% / 0.0%
  const fmtPct = (p) => (p >= 0.9995 ? '>99.9%' : p > 0 && p < 0.0005 ? '<0.1%' : `${(p * 100).toFixed(1)}%`);

  let displayNames = {};
  const labelFor = (cls) => displayNames[cls] || cls;

  let health = null;
  let knownClasses = '';
  let originalUrl = null;
  let modelInputUrl = null;
  let requestId = 0;

  // ---------- API status ----------

  function setStatus(state, text) {
    status.dataset.state = state;
    statusText.textContent = text;
  }

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error(res.statusText);
      health = await res.json();
      displayNames = health.display_names || {};
      if (health.model_loaded) {
        setStatus('ready', `${health.model} ready · ${health.classes.length} classes`);
      } else {
        setStatus('waiting', 'API online · no trained model yet');
      }
      renderSampleButtons(health.classes || []);
    } catch (e) {
      setStatus('offline', 'Model API offline');
    }
  }
  checkHealth();
  setInterval(checkHealth, 15000);

  function renderSampleButtons(classes) {
    if (classes.join() === knownClasses) return;
    knownClasses = classes.join();
    sampleButtons.innerHTML = '';
    classes.forEach((cls) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'chip';
      btn.style.setProperty('--class-color', colorFor(cls));
      btn.innerHTML = '<span class="class-dot"></span>';
      btn.append(labelFor(cls));
      btn.addEventListener('click', () => loadSample(cls, btn));
      sampleButtons.appendChild(btn);
    });
    samples.hidden = classes.length === 0;
  }

  async function loadSample(cls, btn) {
    btn.disabled = true;
    try {
      const res = await fetch(`${API_BASE}/samples/${encodeURIComponent(cls)}`);
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      const blob = await res.blob();
      const name = filenameFrom(res) || `${cls.toLowerCase()}_sample.jpg`;
      handleFile(new File([blob], name, { type: blob.type || 'image/jpeg' }), cls);
    } catch (e) {
      showError(`Could not load a ${labelFor(cls)} test image: ${e.message}`);
    } finally {
      btn.disabled = false;
    }
  }

  function filenameFrom(res) {
    const cd = res.headers.get('content-disposition') || '';
    const m = cd.match(/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i);
    return m ? decodeURIComponent(m[1]) : null;
  }

  // ---------- File input ----------

  const openPicker = () => fileInput.click();
  dropZone.addEventListener('click', openPicker);
  dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      openPicker();
    }
  });
  replaceBtn.addEventListener('click', openPicker);
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
    fileInput.value = '';
  });

  [dropZone, stage].forEach((target) => {
    ['dragenter', 'dragover'].forEach((type) => target.addEventListener(type, (e) => {
      e.preventDefault();
      target.classList.add('dragover');
    }));
    ['dragleave', 'drop'].forEach((type) => target.addEventListener(type, (e) => {
      e.preventDefault();
      target.classList.remove('dragover');
    }));
    target.addEventListener('drop', (e) => {
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    });
  });

  function handleFile(file, groundTruth = null) {
    if (!file.type.startsWith('image/')) {
      showError('Please choose a JPEG or PNG image.');
      return;
    }
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    originalUrl = URL.createObjectURL(file);
    modelInputUrl = null;

    imgHeat.removeAttribute('src');
    imgHeat2.removeAttribute('src');
    setMode(stage.dataset.mode === 'input' ? 'overlay' : stage.dataset.mode);

    dropZone.hidden = true;
    stage.hidden = false;
    toolbar.hidden = false;
    viewerFoot.hidden = false;
    opacityControl.hidden = true;
    setModeEnabled(false);
    fileMeta.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;

    analyze(file, groundTruth);
  }

  // ---------- Inference ----------

  async function analyze(file, groundTruth) {
    const id = ++requestId;
    showState('loading');
    stage.classList.add('busy');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: formData });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
      if (id === requestId) renderResult(data, groundTruth);
    } catch (e) {
      if (id === requestId) showError(e.message);
    } finally {
      if (id === requestId) stage.classList.remove('busy');
    }
  }

  function showState(name) {
    emptyState.hidden = name !== 'empty';
    loadingState.hidden = name !== 'loading';
    errorState.hidden = name !== 'error';
    resultState.hidden = name !== 'result';
  }

  function showError(message) {
    errorMsg.textContent = message;
    showState('error');
  }

  function renderResult(data, groundTruth) {
    displayNames = { ...displayNames, ...(data.display_names || {}) };
    const cls = data.predicted_class;
    const conf = Number(data.confidence) || 0;

    $('verdict').style.setProperty('--class-color', colorFor(cls));
    $('predictedClass').textContent = labelFor(cls);
    $('confidenceValue').textContent = fmtPct(conf);

    const note = $('verdictNote');
    note.hidden = conf >= 0.7;
    note.textContent = 'Low confidence: the model is unsure about this image.';

    const truth = $('truthNote');
    if (groundTruth) {
      const match = groundTruth === cls;
      truth.hidden = false;
      truth.className = `truth ${match ? 'match' : 'miss'}`;
      truth.innerHTML = match
        ? '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>'
        : '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>';
      truth.append(`Held-out test image · labelled ${labelFor(groundTruth)}`);
    } else {
      truth.hidden = true;
    }

    const probs = Object.entries(data.class_probabilities || {}).sort((a, b) => b[1] - a[1]);
    const list = $('probBars');
    list.innerHTML = '';
    probs.forEach(([name, p]) => {
      const li = document.createElement('li');
      li.className = `prob-row${name === cls ? ' top' : ''}`;
      li.style.setProperty('--class-color', colorFor(name));
      li.innerHTML = `
        <div class="prob-head">
          <span class="prob-name"><span class="class-dot"></span></span>
          <span class="prob-val"></span>
        </div>
        <div class="prob-track"><div class="prob-fill"></div></div>`;
      li.querySelector('.prob-name').append(labelFor(name));
      li.querySelector('.prob-val').textContent = fmtPct(p);
      list.appendChild(li);
      requestAnimationFrame(() => {
        li.querySelector('.prob-fill').style.width = `${(p * 100).toFixed(1)}%`;
      });
    });

    modelInputUrl = data.model_input_base64 || null;
    imgHeat.src = data.heatmap_base64;
    imgHeat2.src = data.heatmap_base64;
    opacityControl.hidden = false;
    setModeEnabled(true);
    setMode(stage.dataset.mode === 'original' ? 'overlay' : stage.dataset.mode);

    const details = [
      ['Model', data.model || '—'],
      ['Task', data.task === 'binary' ? 'Normal vs Pneumonia' : `${probs.length}-class`],
      ['Grad-CAM layer', data.target_layer || '—'],
      ['Preprocessing', 'Grayscale → 224×224 → CLAHE'],
      ['File', data.filename || '—'],
    ];
    const dl = $('techDetails');
    dl.innerHTML = '';
    details.forEach(([k, v]) => {
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = k;
      dd.textContent = v;
      dl.append(dt, dd);
    });

    showState('result');
  }

  // ---------- View controls ----------

  function setMode(mode) {
    stage.dataset.mode = mode;
    modeButtons.forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === mode)));
    // "Model input" swaps the base layer for the preprocessed image the network saw
    const base = mode === 'input' && modelInputUrl ? modelInputUrl : originalUrl;
    if (base && imgBase.src !== base) imgBase.src = base;
    if (originalUrl && imgBase2.src !== originalUrl) imgBase2.src = originalUrl;
  }

  function setModeEnabled(enabled) {
    modeButtons.forEach((b) => {
      b.disabled = !enabled && b.dataset.mode !== 'original';
    });
  }

  modeButtons.forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));

  function applyOpacity() {
    const v = Number(opacitySlider.value);
    opacityVal.textContent = `${v}%`;
    stage.style.setProperty('--heat-opacity', String(v / 100));
  }
  opacitySlider.addEventListener('input', applyOpacity);
  applyOpacity();

  showState('empty');
});
