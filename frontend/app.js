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

  const CLASS_META = {
    NORMAL: { label: 'Normal', color: 'var(--c-normal)' },
    PNEUMONIA: { label: 'Pneumonia', color: 'var(--c-pneumonia)' },
    COVID: { label: 'COVID-19', color: 'var(--c-covid)' },
    TB: { label: 'Tuberculosis', color: 'var(--c-tb)' },
  };
  const metaFor = (name) => CLASS_META[name] || { label: name, color: 'var(--c-other)' };

  const MODEL_LABELS = {
    Transfer_resnet50: 'ResNet50 (transfer)',
    Transfer_mobilenet_v2: 'MobileNetV2 (transfer)',
    Baseline_CNN: 'Baseline CNN',
  };
  const modelLabel = (name) => MODEL_LABELS[name] || String(name || '').replace(/_/g, ' ');

  let knownClasses = [];
  let modelName = null;
  let objectUrl = null;
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
      const data = await res.json();
      modelName = data.model_name || null;
      if (data.model_loaded) {
        const n = (data.classes || []).length;
        setStatus('ready', `${modelLabel(data.model_name) || 'Model'} ready · ${n} classes`);
      } else {
        setStatus('waiting', 'API online · no trained model yet');
      }
      renderSampleButtons(data.classes || []);
    } catch (e) {
      setStatus('offline', 'Model API offline');
    }
  }
  checkHealth();
  setInterval(checkHealth, 10000);

  function renderSampleButtons(classes) {
    if (classes.join() === knownClasses.join()) return;
    knownClasses = classes;
    sampleButtons.innerHTML = '';
    classes.forEach((cls) => {
      const meta = metaFor(cls);
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'chip';
      btn.style.setProperty('--class-color', meta.color);
      btn.innerHTML = '<span class="class-dot"></span>';
      btn.append(meta.label);
      btn.addEventListener('click', () => loadSample(cls, btn));
      sampleButtons.appendChild(btn);
    });
    samples.hidden = classes.length === 0;
  }

  async function loadSample(cls, btn) {
    btn.disabled = true;
    try {
      const res = await fetch(`${API_BASE}/samples/${encodeURIComponent(cls)}`);
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const blob = await res.blob();
      const name = filenameFrom(res) || `${cls.toLowerCase()}_sample.jpg`;
      handleFile(new File([blob], name, { type: blob.type || 'image/jpeg' }), cls);
    } catch (e) {
      showError(`Could not load a ${metaFor(cls).label} test image: ${e.message}`);
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
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);

    // Full-resolution upload as the base layer; the heatmap is stretched over it
    imgBase.src = objectUrl;
    imgBase2.src = objectUrl;
    imgHeat.removeAttribute('src');
    imgHeat2.removeAttribute('src');

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
    const cls = data.predicted_class;
    const meta = metaFor(cls);
    const conf = Number(data.confidence) || 0;

    const verdict = $('verdict');
    verdict.style.setProperty('--class-color', meta.color);
    $('predictedClass').textContent = meta.label;
    $('confidenceValue').textContent = `${(conf * 100).toFixed(1)}%`;

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
      truth.append(`Held-out test image · labelled ${metaFor(groundTruth).label}`);
    } else {
      truth.hidden = true;
    }

    const probs = Object.entries(data.class_probabilities || {}).sort((a, b) => b[1] - a[1]);
    const list = $('probBars');
    list.innerHTML = '';
    probs.forEach(([name, p]) => {
      const m = metaFor(name);
      const li = document.createElement('li');
      li.className = `prob-row${name === cls ? ' top' : ''}`;
      li.style.setProperty('--class-color', m.color);
      li.innerHTML = `
        <div class="prob-head">
          <span class="prob-name"><span class="class-dot"></span></span>
          <span class="prob-val"></span>
        </div>
        <div class="prob-track"><div class="prob-fill"></div></div>`;
      li.querySelector('.prob-name').append(m.label);
      li.querySelector('.prob-val').textContent = `${(p * 100).toFixed(1)}%`;
      list.appendChild(li);
      requestAnimationFrame(() => {
        li.querySelector('.prob-fill').style.width = `${(p * 100).toFixed(1)}%`;
      });
    });

    // Older API versions only return the pre-blended overlay
    const heat = data.heatmap_base64 || data.gradcam_overlay_base64;
    imgHeat.src = heat;
    imgHeat2.src = heat;
    if (!data.heatmap_base64) {
      imgBase.src = data.original_image_base64;
      imgBase2.src = data.original_image_base64;
    }
    opacityControl.hidden = !data.heatmap_base64;
    setModeEnabled(true);
    if (stage.dataset.mode === 'original') setMode('overlay');

    const details = [
      ['Model', modelLabel(modelName) || '—'],
      ['Grad-CAM layer', data.target_layer_used || '—'],
      ['Model input', '224 × 224 RGB'],
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
