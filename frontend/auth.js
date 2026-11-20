document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);
  const API = window.location.origin;
  const TOKEN_KEY = 'pulmoscan:auth-token';
  const USER_KEY = 'pulmoscan:auth-user';
  let token = sessionStorage.getItem(TOKEN_KEY);
  let currentUser = null;
  let refreshTimer = null;

  const request = async (path, options = {}) => {
    const headers = new Headers(options.headers || {});
    if (token) headers.set('Authorization', `Bearer ${token}`);
    const response = await fetch(`${API}${path}`, { ...options, headers });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 401 && token) logout(false);
      throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status})`);
    }
    return body;
  };

  function showMessage(target, message, isError = false) {
    target.textContent = message;
    target.classList.toggle('error-text', isError);
  }

  function setSession(data) {
    token = data.access_token;
    currentUser = data.user;
    sessionStorage.setItem(TOKEN_KEY, token);
    sessionStorage.setItem(USER_KEY, JSON.stringify(currentUser));
    if (currentUser.role === 'PATIENT') {
      window.location.assign('index.html');
      return;
    }
    renderDashboard();
  }

  function logout(notify = true) {
    token = null;
    currentUser = null;
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(USER_KEY);
    if (refreshTimer) clearInterval(refreshTimer);
    $('authPanel').hidden = false;
    $('authIntro').hidden = false;
    $('doctorDashboard').hidden = true;
    $('logoutBtn').hidden = true;
    if (notify) showMessage($('authMessage'), 'You have signed out.');
  }

  async function renderDashboard() {
    if (!token) return;
    try {
      const { user } = await request('/auth/me');
      currentUser = user;
      sessionStorage.setItem(USER_KEY, JSON.stringify(user));
      if (user.role === 'PATIENT') {
        window.location.replace('index.html');
        return;
      }
      $('authPanel').hidden = true;
      $('authIntro').hidden = true;
      $('logoutBtn').hidden = false;
      $('doctorDashboard').hidden = false;
      await refreshDoctorQueue();
      if (refreshTimer) clearInterval(refreshTimer);
      refreshTimer = setInterval(() => refreshDoctorQueue().catch((e) => showMessage($('doctorMessage'), e.message, true)), 12000);
    } catch (error) {
      logout(false);
      showMessage($('authMessage'), error.message, true);
    }
  }

  document.querySelectorAll('.portal-tab').forEach((tab) => tab.addEventListener('click', () => {
    document.querySelectorAll('.portal-tab').forEach((button) => {
      button.classList.toggle('active', button === tab);
      button.setAttribute('aria-pressed', String(button === tab));
    });
    document.querySelectorAll('.portal-form').forEach((form) => { form.hidden = form.id !== tab.dataset.panel; });
  }));
  $('logoutBtn').addEventListener('click', () => logout());

  $('loginPanel').addEventListener('submit', async (event) => {
    event.preventDefault();
    showMessage($('authMessage'), 'Signing in…');
    try {
      const data = await request('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: $('loginEmail').value,
          password: $('loginPassword').value,
          role: $('loginRole').value,
        }),
      });
      setSession(data);
    } catch (error) {
      showMessage($('authMessage'), error.message, true);
    }
  });

  $('patientRegisterPanel').addEventListener('submit', async (event) => {
    event.preventDefault();
    showMessage($('authMessage'), 'Creating patient account…');
    try {
      const data = await request('/auth/patient/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: $('patientEmail').value,
          full_name: $('patientFullName').value,
          password: $('patientPassword').value,
        }),
      });
      setSession(data);
    } catch (error) {
      showMessage($('authMessage'), error.message, true);
    }
  });

  $('doctorRegisterPanel').addEventListener('submit', async (event) => {
    event.preventDefault();
    showMessage($('authMessage'), 'Verifying doctor registration…');
    try {
      const data = await request('/auth/doctor/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: $('doctorEmail').value,
          full_name: $('doctorFullName').value,
          password: $('doctorPassword').value,
          verification_code: $('doctorCode').value,
        }),
      });
      setSession(data);
    } catch (error) {
      showMessage($('authMessage'), error.message, true);
    }
  });

  async function loadAuthorizedImage(url, alt) {
    const response = await fetch(`${API}${url}`, { headers: { Authorization: `Bearer ${token}` } });
    if (!response.ok) throw new Error(`Could not load protected image (${response.status}).`);
    const image = document.createElement('img');
    image.src = URL.createObjectURL(await response.blob());
    image.alt = alt;
    image.className = 'portal-review-image';
    return image;
  }

  async function refreshDoctorQueue() {
    const data = await request('/auth/doctor/reviews');
    const list = $('doctorQueue');
    list.replaceChildren();
    if (!data.reviews.length) {
      list.textContent = 'No pending review requests.';
      return;
    }
    data.reviews.forEach((review) => {
      const card = document.createElement('article');
      card.className = 'portal-list-item';
      const heading = document.createElement('strong');
      heading.textContent = `${review.patient_name} · Patient ID ${review.patient_id}`;
      const detail = document.createElement('p');
      detail.textContent = `${review.predicted_label} · ${(review.confidence * 100).toFixed(1)}% AI confidence · requested ${new Date(review.requested_at).toLocaleString()}`;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'btn btn-primary';
      button.textContent = review.assigned_doctor_user_id ? 'Continue review' : 'Open / claim review';
      button.addEventListener('click', () => openDoctorReview(review.request_id).catch((e) => showMessage($('doctorMessage'), e.message, true)));
      card.append(heading, detail, button);
      list.append(card);
    });
  }

  async function openDoctorReview(requestId) {
    const data = await request(`/auth/doctor/reviews/${encodeURIComponent(requestId)}/claim`, { method: 'POST' });
    const panel = $('doctorReviewDetail');
    panel.replaceChildren();
    panel.hidden = false;
    const heading = document.createElement('h2');
    heading.textContent = `Review · ${data.patient_name} · Patient ID ${data.patient_id}`;
    const time = document.createElement('p');
    time.textContent = `Requested ${new Date(data.requested_at).toLocaleString()} · Status ${data.status}`;
    const prediction = document.createElement('p');
    prediction.textContent = `AI predicted ${data.result.predicted_label} · ${(data.result.confidence * 100).toFixed(1)}% confidence`;
    const probabilities = document.createElement('p');
    probabilities.textContent = Object.entries(data.result.class_probabilities)
      .map(([label, probability]) => `${data.result.display_names[label] || label}: ${(probability * 100).toFixed(1)}%`)
      .join(' · ');
    const quality = document.createElement('p');
    quality.textContent = `Image quality: ${data.result.quality.status} — ${(data.result.quality.warnings || []).join(' ') || 'No technical issues detected.'}`;
    const historyHeading = document.createElement('h3');
    historyHeading.textContent = 'Previous review history for this patient';
    const previousHistory = document.createElement('div');
    previousHistory.className = 'portal-list';
    if (data.previous_review_history.length) {
      data.previous_review_history.forEach((entry) => {
        const historyItem = document.createElement('p');
        const review = entry.doctor_review;
        historyItem.textContent = `${new Date(entry.created_at).toLocaleString()} · AI: ${entry.ai_prediction} (${(entry.confidence * 100).toFixed(1)}%) · ${review ? `Doctor ${review.decision} by ${review.doctor_name}: ${review.observations}` : `review ${entry.review_status || 'not requested'}`}`;
        previousHistory.append(historyItem);
      });
    } else {
      previousHistory.textContent = 'No previous AI report or doctor review is available.';
    }
    const images = document.createElement('div');
    images.className = 'portal-image-grid';
    images.append(
      await loadAuthorizedImage(data.image_urls.original, 'Patient chest X-ray'),
      await loadAuthorizedImage(data.image_urls.gradcam, 'Grad-CAM overlay'),
      await loadAuthorizedImage(data.image_urls.heatmap, 'Grad-CAM heatmap'),
    );
    panel.append(heading, time, prediction, probabilities, quality, images, historyHeading, previousHistory);
    const form = document.createElement('form');
    form.className = 'portal-form doctor-review-form';
    const decisionField = document.createElement('div');
    decisionField.className = 'field';
    const decisionLabel = document.createElement('label');
    decisionLabel.textContent = 'Decision';
    const decision = document.createElement('select');
    decision.innerHTML = '<option value="APPROVE">Approve AI result</option><option value="REJECT">Reject AI result</option>';
    decisionField.append(decisionLabel, decision);
    const observationsField = document.createElement('div');
    observationsField.className = 'field';
    const observationsLabel = document.createElement('label');
    observationsLabel.textContent = 'Clinical observations (required)';
    const observations = document.createElement('textarea');
    observations.required = true;
    observations.maxLength = 5000;
    observationsField.append(observationsLabel, observations);
    const commentsField = document.createElement('div');
    commentsField.className = 'field';
    const commentsLabel = document.createElement('label');
    commentsLabel.textContent = 'Additional comments';
    const comments = document.createElement('textarea');
    comments.maxLength = 5000;
    commentsField.append(commentsLabel, comments);
    const submit = document.createElement('button');
    submit.type = 'submit';
    submit.className = 'btn btn-primary';
    submit.textContent = 'Submit doctor review';
    form.append(decisionField, observationsField, commentsField, submit);
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      submit.disabled = true;
      try {
        await request(`/auth/doctor/reviews/${encodeURIComponent(requestId)}/submit`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ decision: decision.value, observations: observations.value, comments: comments.value }),
        });
        showMessage($('doctorMessage'), 'Review submitted. The patient will see the completed review in their report.');
        panel.hidden = true;
        await refreshDoctorQueue();
      } catch (error) {
        showMessage($('doctorMessage'), error.message, true);
        submit.disabled = false;
      }
    });
    panel.append(form);
    showMessage($('doctorMessage'), 'Review is assigned to you. The AI prediction remains unchanged.');
  }

  $('refreshDoctorQueue').addEventListener('click', () => refreshDoctorQueue().catch((e) => showMessage($('doctorMessage'), e.message, true)));

  if (token) renderDashboard();
});
