document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);
  const { API_BASE, colorFor, labelFor } = PulmoScan;

  const form = $('intakeForm');
  const nameInput = $('patientName');
  const ageInput = $('patientAge');
  const dropZone = $('dropZone');
  const fileInput = $('fileInput');
  const analyzeBtn = $('analyzeBtn');
  const busy = $('busyOverlay');

  let selected = null; // { file, groundTruth, previewUrl }
  let knownClasses = '';

  // ---------- Model status, sample buttons, model facts ----------

  PulmoScan.watchStatus((health) => {
    renderSampleButtons(health.classes || []);
    $('factModel').textContent = health.model || '—';
    $('factClasses').textContent = (health.classes || []).map(labelFor).join(', ') || '—';
    const m = health.test_metrics;
    $('factAccuracy').textContent = m ? `${(m.accuracy * 100).toFixed(1)}% on ${m.n} test images` : '—';
  });

  function renderSampleButtons(classes) {
    if (classes.join() === knownClasses) return;
    knownClasses = classes.join();
    const row = $('sampleButtons');
    row.innerHTML = '';
    classes.forEach((cls) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'chip';
      btn.style.setProperty('--class-color', colorFor(cls));
      btn.innerHTML = '<span class="class-dot"></span>';
      btn.append(labelFor(cls));
      btn.addEventListener('click', () => loadSample(cls, btn));
      row.appendChild(btn);
    });
    $('samples').hidden = classes.length === 0;
  }

  async function loadSample(cls, btn) {
    btn.disabled = true;
    try {
      const res = await fetch(`${API_BASE}/samples/${encodeURIComponent(cls)}`);
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      const blob = await res.blob();
      const name = filenameFrom(res) || `${cls.toLowerCase()}_sample.jpg`;
      selectFile(new File([blob], name, { type: blob.type || 'image/jpeg' }), cls);
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

  // ---------- Image selection ----------

  const openPicker = () => fileInput.click();
  dropZone.addEventListener('click', openPicker);
  dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      openPicker();
    }
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) selectFile(e.target.files[0]);
    fileInput.value = '';
  });
  ['dragenter', 'dragover'].forEach((type) => dropZone.addEventListener(type, (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((type) => dropZone.addEventListener(type, (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
  }));
  dropZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) selectFile(file);
  });

  function selectFile(file, groundTruth = null) {
    if (!file.type.startsWith('image/')) {
      showError('Please choose a JPEG or PNG image.');
      return;
    }
    if (selected && selected.previewUrl) URL.revokeObjectURL(selected.previewUrl);
    selected = { file, groundTruth, previewUrl: URL.createObjectURL(file) };
    $('pickedImg').src = selected.previewUrl;
    $('pickedName').textContent = file.name;
    $('pickedSub').textContent = groundTruth
      ? `Held-out test image · labelled ${labelFor(groundTruth)} · ${(file.size / 1024).toFixed(0)} KB`
      : `${(file.size / 1024).toFixed(0)} KB`;
    $('dropPrompt').hidden = true;
    $('picked').hidden = false;
    dropZone.classList.add('has-image');
    $('imageError').hidden = true;
    hideError();
  }

  // ---------- Validation ----------

  function readPatient() {
    const name = nameInput.value.trim().replace(/\s+/g, ' ');
    const age = Number(ageInput.value);
    const nameOk = name.length >= 2 && name.length <= 60;
    const ageOk = ageInput.value !== '' && Number.isInteger(age) && age >= 0 && age <= 120;
    return { name, age, nameOk, ageOk };
  }

  function validate() {
    const p = readPatient();
    $('nameError').hidden = p.nameOk;
    $('ageError').hidden = p.ageOk;
    $('imageError').hidden = Boolean(selected);
    nameInput.setAttribute('aria-invalid', String(!p.nameOk));
    ageInput.setAttribute('aria-invalid', String(!p.ageOk));
    if (!p.nameOk) nameInput.focus();
    else if (!p.ageOk) ageInput.focus();
    else if (!selected) dropZone.focus();
    return p.nameOk && p.ageOk && selected ? p : null;
  }

  [nameInput, ageInput].forEach((el) => el.addEventListener('input', () => {
    const p = readPatient();
    if (el === nameInput && p.nameOk) $('nameError').hidden = true;
    if (el === ageInput && p.ageOk) $('ageError').hidden = true;
  }));

  // ---------- Analyze -> report page ----------

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();
    const patient = validate();
    if (!patient) return;

    busy.hidden = false;
    analyzeBtn.disabled = true;
    try {
      const body = new FormData();
      body.append('file', selected.file); // only the image goes to the server
      const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);

      const saved = PulmoScan.saveAnalysis({
        version: 1,
        reportId: makeReportId(),
        createdAt: new Date().toISOString(),
        patient: { name: patient.name, age: patient.age },
        image: { name: selected.file.name, sizeKB: Math.round(selected.file.size / 1024), dataUrl: await toDataUrl(selected.file) },
        groundTruth: selected.groundTruth,
        result: data,
      });
      if (!saved) throw new Error('The browser would not store the report (storage is full or disabled).');
      window.location.href = 'results.html';
    } catch (err) {
      busy.hidden = true;
      analyzeBtn.disabled = false;
      showError(err.message);
    }
  });

  function makeReportId() {
    const d = new Date();
    const ymd = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}`;
    const rand = Math.floor(Math.random() * 0x10000).toString(16).toUpperCase().padStart(4, '0');
    return `PS-${ymd}-${rand}`;
  }

  /** Downsized JPEG copy of the upload for the report page (object URLs do not survive navigation). */
  async function toDataUrl(file, maxSide = 1600) {
    const url = URL.createObjectURL(file);
    try {
      const img = await new Promise((resolve, reject) => {
        const i = new Image();
        i.onload = () => resolve(i);
        i.onerror = () => reject(new Error('Could not read the image in the browser.'));
        i.src = url;
      });
      const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(img.naturalWidth * scale);
      canvas.height = Math.round(img.naturalHeight * scale);
      canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', 0.9);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function showError(message) {
    $('errorMsg').textContent = message;
    $('errorState').hidden = false;
  }

  function hideError() {
    $('errorState').hidden = true;
  }

  // Coming back with the browser's Back button restores the page from cache with the overlay up
  window.addEventListener('pageshow', () => {
    busy.hidden = true;
    analyzeBtn.disabled = false;
  });
});
