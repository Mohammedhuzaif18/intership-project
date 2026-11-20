document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);
  const { API_BASE, colorFor, labelFor } = PulmoScan;

  const form = $('intakeForm');
  const nameInput = $('patientName');
  const ageInput = $('patientAge');
  const patientIdInput = $('patientId');
  const dropZone = $('dropZone');
  const fileInput = $('fileInput');
  const analyzeBtn = $('analyzeBtn');
  const busy = $('busyOverlay');

  let selected = null; // { file, groundTruth, previewUrl }
  let knownClasses = '';
  const authToken = () => sessionStorage.getItem('pulmoscan:auth-token');

  let patientReportsTimer = null;
  const patientReportObjectUrls = new Set();

  async function configurePatientSession() {
    const token = authToken();
    if (!token) return true;
    try {
      const response = await fetch(`${API_BASE}/auth/me`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const identity = await response.json().catch(() => ({}));
      if (!response.ok || identity.user?.role !== 'PATIENT') {
        throw new Error('Your patient session is no longer valid. Please sign in again.');
      }
      const user = identity.user;
      nameInput.value = user.full_name;
      nameInput.readOnly = true;
      patientIdInput.value = user.patient_id;
      patientIdInput.readOnly = true;
      $('historyPatientId').value = user.patient_id;
      $('historyPatientId').readOnly = true;
      $('accountName').textContent = `Signed in: ${user.full_name}`;
      $('accountLink').hidden = true;
      $('accountSession').hidden = false;
      $('patientReportsPanel').hidden = false;
      await refreshPatientReports();
      if (patientReportsTimer) clearInterval(patientReportsTimer);
      patientReportsTimer = setInterval(() => {
        refreshPatientReports().catch((error) => {
          $('patientReportsMessage').textContent = `Could not refresh reports: ${error.message}`;
        });
      }, 12000);
      return true;
    } catch (error) {
      sessionStorage.removeItem('pulmoscan:auth-token');
      sessionStorage.removeItem('pulmoscan:auth-user');
      window.location.replace('auth.html');
      return false;
    }
  }

  const patientSessionReady = configurePatientSession();
  $('signOutBtn').addEventListener('click', () => {
    if (patientReportsTimer) clearInterval(patientReportsTimer);
    sessionStorage.removeItem('pulmoscan:auth-token');
    sessionStorage.removeItem('pulmoscan:auth-user');
    window.location.replace('auth.html');
  });
  $('refreshPatientReports').addEventListener('click', () => {
    refreshPatientReports().catch((error) => {
      $('patientReportsMessage').textContent = `Could not load reports: ${error.message}`;
    });
  });

  async function refreshPatientReports() {
    const token = authToken();
    if (!token) return;
    const message = $('patientReportsMessage');
    const list = $('patientReportsList');
    message.textContent = 'Loading your X-ray reports and doctor recommendations…';
    const response = await fetch(`${API_BASE}/auth/patient/predictions`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(apiError(data, response.status));
    }

    patientReportObjectUrls.forEach((url) => URL.revokeObjectURL(url));
    patientReportObjectUrls.clear();
    list.replaceChildren();
    if (!data.predictions.length) {
      message.textContent = 'Your reports and completed doctor recommendations will appear here after analysis.';
      return;
    }

    const completedCount = data.predictions.filter(
      (report) => report.review_request?.status === 'COMPLETED',
    ).length;
    message.textContent = completedCount
      ? `${completedCount} doctor recommendation${completedCount === 1 ? '' : 's'} completed.`
      : 'Your reports are shown newest first. Doctor-review status updates automatically.';

    data.predictions.forEach((report) => renderPatientReport(report, token));
  }

  function renderPatientReport(report, token) {
    const result = report.result;
    const reviewRequest = report.review_request;
    const review = reviewRequest?.review;
    const card = document.createElement('article');
    card.className = 'patient-report-card';

    const header = document.createElement('div');
    header.className = 'patient-report-header';
    const title = document.createElement('h3');
    title.textContent = `${result.predicted_label} · ${PulmoScan.fmtPct(result.confidence)} model confidence`;
    const date = document.createElement('time');
    date.dateTime = report.created_at;
    date.textContent = new Date(report.created_at).toLocaleString();
    header.append(title, date);

    const probabilities = document.createElement('p');
    probabilities.className = 'muted';
    probabilities.textContent = Object.entries(result.class_probabilities || {})
      .map(([name, value]) => `${PulmoScan.labelFor(name)} ${PulmoScan.fmtPct(value)}`)
      .join(' · ');
    card.append(header, probabilities);

    if (!reviewRequest) {
      const state = document.createElement('p');
      state.className = 'review-status';
      state.textContent = 'AI result only · no doctor review requested';
      card.append(state);
      $('patientReportsList').append(card);
      return;
    }

    const state = document.createElement('p');
    state.className = `review-status${reviewRequest.status === 'COMPLETED' ? ' review-complete' : ''}`;
    state.textContent = reviewRequest.status === 'COMPLETED'
      ? 'Doctor recommendation is ready'
      : `Doctor review ${reviewRequest.status.replace('_', ' ').toLowerCase()}`;
    card.append(state);

    if (review) {
      const recommendation = document.createElement('section');
      recommendation.className = 'doctor-recommendation';
      const heading = document.createElement('h4');
      heading.textContent = `Doctor recommendation: ${review.decision === 'APPROVE' ? 'AI result approved' : 'AI result not approved'}`;
      const attribution = document.createElement('p');
      attribution.textContent = `Reviewed by ${review.doctor_name} on ${new Date(review.reviewed_at).toLocaleString()}.`;
      const observations = document.createElement('p');
      observations.textContent = `Recommendation and observations: ${review.observations}`;
      const comments = document.createElement('p');
      comments.textContent = `Additional comments: ${review.comments || 'None provided.'}`;
      recommendation.append(heading, attribution, observations, comments);
      card.append(recommendation);

      const images = document.createElement('div');
      images.className = 'history-images patient-report-images';
      [['original', 'Reviewed X-ray'], ['gradcam', 'Grad-CAM']].forEach(([kind, label]) => {
        const figure = document.createElement('figure');
        const image = document.createElement('img');
        image.alt = label;
        fetch(`${API_BASE}${report.image_urls[kind]}`, {
          headers: { Authorization: `Bearer ${token}` },
        })
          .then((imageResponse) => {
            if (!imageResponse.ok) throw new Error(`Image request failed (${imageResponse.status}).`);
            return imageResponse.blob();
          })
          .then((blob) => {
            const objectUrl = URL.createObjectURL(blob);
            if (image.isConnected) {
              patientReportObjectUrls.add(objectUrl);
              image.src = objectUrl;
            } else {
              URL.revokeObjectURL(objectUrl);
            }
          })
          .catch(() => { image.alt = `${label} could not be loaded.`; });
        const caption = document.createElement('figcaption');
        caption.textContent = label;
        figure.append(image, caption);
        images.append(figure);
      });
      card.append(images);
    } else {
      const pendingText = document.createElement('p');
      pendingText.textContent = 'The doctor has not submitted a recommendation yet. This page checks for updates automatically.';
      card.append(pendingText);
    }

    $('patientReportsList').append(card);
  }

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
    $('qualityResult').hidden = true;
    hideError();
  }

  // ---------- Validation ----------

  function readPatient() {
    const name = nameInput.value.trim().replace(/\s+/g, ' ');
    const age = Number(ageInput.value);
    const patientId = patientIdInput.value.trim();
    const nameOk = name.length >= 2 && name.length <= 120;
    const ageOk = ageInput.value !== '' && Number.isInteger(age) && age >= 0 && age <= 120;
    const patientIdOk = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(patientId);
    return { name, age, patientId, nameOk, ageOk, patientIdOk };
  }

  function validate() {
    const p = readPatient();
    $('nameError').hidden = p.nameOk;
    $('ageError').hidden = p.ageOk;
    $('patientIdError').hidden = p.patientIdOk;
    $('imageError').hidden = Boolean(selected);
    nameInput.setAttribute('aria-invalid', String(!p.nameOk));
    ageInput.setAttribute('aria-invalid', String(!p.ageOk));
    patientIdInput.setAttribute('aria-invalid', String(!p.patientIdOk));
    if (!p.nameOk) nameInput.focus();
    else if (!p.ageOk) ageInput.focus();
    else if (!p.patientIdOk) patientIdInput.focus();
    else if (!selected) dropZone.focus();
    return p.nameOk && p.ageOk && p.patientIdOk && selected ? p : null;
  }

  [nameInput, ageInput, patientIdInput].forEach((el) => el.addEventListener('input', () => {
    const p = readPatient();
    if (el === nameInput && p.nameOk) $('nameError').hidden = true;
    if (el === ageInput && p.ageOk) $('ageError').hidden = true;
    if (el === patientIdInput && p.patientIdOk) $('patientIdError').hidden = true;
  }));

  // ---------- Analyze -> report page ----------

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();
    if (!(await patientSessionReady)) return;
    const patient = validate();
    if (!patient) return;

    busy.hidden = false;
    analyzeBtn.disabled = true;
    try {
      const qualityBody = new FormData();
      qualityBody.append('file', selected.file);
      const qualityResponse = await fetch(`${API_BASE}/quality`, { method: 'POST', body: qualityBody });
      const qualityData = await qualityResponse.json().catch(() => ({}));
      if (!qualityResponse.ok) throw new Error(apiError(qualityData, qualityResponse.status));
      renderQuality(qualityData.quality);
      if (!qualityData.quality.passed) {
        busy.hidden = true;
        analyzeBtn.disabled = false;
        return;
      }

      const body = new FormData();
      body.append('file', selected.file);
      let endpoint = '/predict';
      const token = authToken();
      const headers = {};
      if (token) {
        const identityResponse = await fetch(`${API_BASE}/auth/me`, { headers: { Authorization: `Bearer ${token}` } });
        const identity = await identityResponse.json().catch(() => ({}));
        if (!identityResponse.ok || identity.user?.role !== 'PATIENT') {
          throw new Error('Sign in with a patient account to save records or request a doctor review.');
        }
        if (patient.patientId !== identity.user.patient_id) {
          throw new Error(`Use the Patient ID linked to your account: ${identity.user.patient_id}.`);
        }
        endpoint = '/auth/patient/predict';
        headers.Authorization = `Bearer ${token}`;
      } else {
        body.delete('patient_id');
      }
      const res = await fetch(`${API_BASE}${endpoint}`, { method: 'POST', body, headers });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (data.detail && data.detail.quality) renderQuality(data.detail.quality);
        throw new Error(apiError(data, res.status));
      }

      const saved = PulmoScan.saveAnalysis({
        version: 1,
        reportId: makeReportId(),
        createdAt: new Date().toISOString(),
        patient: { name: patient.name, age: patient.age, patientId: patient.patientId },
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

  function apiError(data, status) {
    const detail = data.detail;
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail.message === 'string') return detail.message;
    return `Request failed (${status})`;
  }

  function renderQuality(quality) {
    const panel = $('qualityResult');
    panel.replaceChildren();
    panel.className = `quality-result quality-${quality.status}`;
    const title = document.createElement('strong');
    title.textContent = quality.status === 'pass'
      ? 'Technical image checks passed'
      : quality.status === 'review' ? 'Please review image framing' : 'Image quality check failed';
    const list = document.createElement('ul');
    [...quality.warnings, ...quality.suggestions].forEach((message) => {
      const item = document.createElement('li');
      item.textContent = message;
      list.append(item);
    });
    const disclaimer = document.createElement('p');
    disclaimer.textContent = quality.disclaimer;
    panel.append(title, list, disclaimer);
    panel.hidden = false;
  }

  // ---------- Persistent patient history ----------

  let historyPatientId = null;
  let historyRecords = [];
  const historyForm = $('historyForm');
  historyForm.addEventListener('submit', (event) => {
    event.preventDefault();
    loadHistory($('historyPatientId').value.trim());
  });

  async function loadHistory(patientId) {
    const message = $('historyMessage');
    $('historyRecords').replaceChildren();
    $('historyCompareControls').hidden = true;
    $('historyComparison').hidden = true;
    historyPatientId = patientId;
    historyRecords = [];
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(patientId)) {
      message.textContent = 'Enter a valid Patient ID (letters, numbers, dots, underscores or hyphens).';
      return;
    }
    message.textContent = 'Loading patient history…';
    try {
      const token = authToken();
      if (!token) throw new Error('Sign in to your patient account to view protected history.');
      const response = await fetch(`${API_BASE}/patients/${encodeURIComponent(patientId)}/history`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiError(data, response.status));
      historyRecords = data.records;
      message.textContent = `${historyRecords.length} record${historyRecords.length === 1 ? '' : 's'} for Patient ID ${patientId}.`;
      historyRecords.forEach(renderHistoryRecord);
      if (historyRecords.length >= 2) {
        populateComparisonSelects();
        $('historyCompareControls').hidden = false;
      }
    } catch (error) {
      message.textContent = error.message;
    }
  }

  function renderHistoryRecord(record) {
    const card = document.createElement('article');
    card.className = 'history-record';
    const details = document.createElement('div');
    const heading = document.createElement('h3');
    heading.textContent = new Date(record.uploaded_at).toLocaleString();
    const prediction = document.createElement('p');
    prediction.textContent = `${record.predicted_label} · ${PulmoScan.fmtPct(record.confidence)} confidence`;
    const scores = document.createElement('p');
    scores.className = 'muted';
    scores.textContent = Object.entries(record.class_probabilities)
      .map(([name, value]) => `${PulmoScan.labelFor(name)} ${PulmoScan.fmtPct(value)}`).join(' · ');
    details.append(heading, prediction, scores);
    const images = document.createElement('div');
    images.className = 'history-images';
    [['original', 'X-ray'], ['gradcam', 'Grad-CAM']].forEach(([kind, label]) => {
      const figure = document.createElement('figure');
      const image = document.createElement('img');
      image.alt = `${label} for ${record.predicted_label}`;
      const token = authToken();
      if (token) {
        fetch(`${API_BASE}${record.image_urls[kind]}`, {
          headers: { Authorization: `Bearer ${token}` },
        })
          .then((response) => {
            if (!response.ok) throw new Error(`Image request failed (${response.status}).`);
            return response.blob();
          })
          .then((blob) => { image.src = URL.createObjectURL(blob); })
          .catch(() => { image.alt = `${label} could not be loaded. Please refresh your patient history.`; });
      } else {
        image.alt = `${label} is available after signing in.`;
      }
      const caption = document.createElement('figcaption');
      caption.textContent = label;
      figure.append(image, caption);
      images.append(figure);
    });
    card.append(details, images);
    $('historyRecords').append(card);
  }

  function populateComparisonSelects() {
    const first = $('compareRecordA');
    const second = $('compareRecordB');
    [first, second].forEach((select) => {
      select.replaceChildren();
      historyRecords.forEach((record) => {
        const option = document.createElement('option');
        option.value = record.record_id;
        option.textContent = `${new Date(record.uploaded_at).toLocaleString()} · ${record.predicted_label}`;
        select.append(option);
      });
    });
    first.value = historyRecords[0].record_id;
    second.value = historyRecords[historyRecords.length - 1].record_id;
  }

  $('compareHistoryBtn').addEventListener('click', async () => {
    const first = $('compareRecordA').value;
    const second = $('compareRecordB').value;
    if (first === second) {
      $('historyMessage').textContent = 'Choose two different records to compare.';
      return;
    }
    try {
      const params = new URLSearchParams({ record_a_id: first, record_b_id: second });
      const token = authToken();
      const response = await fetch(`${API_BASE}/patients/${encodeURIComponent(historyPatientId)}/compare?${params}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiError(data, response.status));
      const panel = $('historyComparison');
      panel.replaceChildren(PulmoScan.createComparisonView(data));
      panel.hidden = false;
      $('historyMessage').textContent = 'Comparison is scoped to the selected Patient ID.';
    } catch (error) {
      $('historyMessage').textContent = error.message;
    }
  });

  // Coming back with the browser's Back button restores the page from cache with the overlay up
  window.addEventListener('pageshow', () => {
    busy.hidden = true;
    analyzeBtn.disabled = false;
  });
});
