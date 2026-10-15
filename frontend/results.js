// Five short lines of general, educational information per class (WHO and standard medical references)
const DISEASE_INFO = {
  NORMAL: {
    title: 'About a normal result',
    lines: [
      'The model did not find the patterns it associates with pneumonia, COVID-19 or tuberculosis in this X-ray.',
      'A normal chest X-ray shows clear lung fields and sharp outlines of the heart and diaphragm, without patches of abnormal whiteness (opacity).',
      'A normal X-ray does not rule out disease: early infections and small lesions can look normal on an X-ray.',
      'Symptoms such as fever, persistent cough, chest pain or shortness of breath should still be checked by a doctor.',
      'This result comes from an educational AI model and is not a medical diagnosis.',
    ],
  },
  PNEUMONIA: {
    title: 'About pneumonia',
    lines: [
      'Pneumonia is an infection that inflames the air sacs (alveoli) of one or both lungs, which can fill with fluid or pus.',
      'It is most often caused by bacteria such as Streptococcus pneumoniae or by viruses such as influenza and RSV; fungi are a less common cause.',
      'Common symptoms are cough with phlegm, fever, chills, fast or difficult breathing and chest pain when breathing.',
      'On a chest X-ray it usually appears as white, cloudy areas (consolidation or infiltrates) in the lung fields.',
      'Bacterial pneumonia is treated with antibiotics and viral pneumonia with supportive care; pneumococcal and flu vaccines help prevent it.',
    ],
  },
  COVID: {
    title: 'About COVID-19',
    lines: [
      'COVID-19 is a respiratory illness caused by the SARS-CoV-2 coronavirus, first identified in 2019.',
      'It spreads mainly through droplets and aerosols released when an infected person breathes, coughs or talks.',
      'Common symptoms are fever, cough, tiredness, sore throat and loss of taste or smell; severe cases cause difficulty breathing.',
      'On chest X-rays COVID-19 pneumonia often shows hazy or patchy opacities, frequently in both lungs and in the outer, lower zones.',
      'Diagnosis is confirmed with a laboratory test (PCR or rapid antigen), not by X-ray alone; vaccination lowers the risk of severe illness.',
    ],
  },
  TB: {
    title: 'About tuberculosis (TB)',
    lines: [
      'Tuberculosis is an infectious disease caused by the bacterium Mycobacterium tuberculosis and most often affects the lungs.',
      'It spreads through the air when a person with active lung TB coughs, sneezes or speaks.',
      'Typical symptoms are a cough lasting more than two to three weeks (sometimes with blood), fever, night sweats and weight loss.',
      'On chest X-rays it often appears in the upper parts of the lungs as patchy opacities, nodules or cavities.',
      'TB is curable with a full course of combined antibiotics, usually at least six months; diagnosis is confirmed with sputum tests.',
    ],
  },
};

document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);
  const { colorFor, fmtPct, labelFor } = PulmoScan;

  PulmoScan.watchStatus();

  const record = PulmoScan.loadAnalysis();
  if (!record || !record.result) {
    $('noReport').hidden = false;
    return;
  }
  const { patient, image, result, groundTruth } = record;
  PulmoScan.setDisplayNames(result.display_names);
  $('report').hidden = false;

  // ---------- Header and patient ----------

  const ageText = `${patient.age} year${patient.age === 1 ? '' : 's'}`;
  const when = new Date(record.createdAt);
  const whenText = when.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
  $('reportTitle').textContent = `${patient.name}, ${ageText}`;
  $('reportId').textContent = `Report ${record.reportId}`;
  $('reportDate').textContent = whenText;
  $('reportModel').textContent = `${result.model} (${result.task === 'binary' ? 'Normal vs Pneumonia' : `${Object.keys(result.display_names || {}).length} classes`})`;
  document.title = `PulmoScan · Report ${record.reportId}`;

  $('patientName').textContent = patient.name;
  $('patientAge').textContent = ageText;
  $('imageName').textContent = `${image.name} · ${image.sizeKB} KB`;
  $('analysedAt').textContent = whenText;

  function fillDetails(rows) {
    rows.forEach(([k, v]) => {
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = k;
      dd.textContent = v;
      $('techDetails').append(dt, dd);
    });
  }

  // ---------- Not a chest X-ray: no disease result, probabilities or Grad-CAM ----------

  if (result.is_chest_xray === false) {
    $('verdict').closest('section').hidden = true;
    $('aboutDisease').hidden = true;
    $('viewerTitle').closest('section').hidden = true;
    $('notChest').hidden = false;
    $('notChestScore').textContent = `Similarity to chest X-rays: ${Number(result.chest_score).toFixed(2)} (at least ${Number(result.chest_threshold).toFixed(2)} is needed).`;
    fillDetails([
      ['Model', result.model || '—'],
      ['Chest X-ray check', `Not passed (similarity ${Number(result.chest_score).toFixed(2)}, threshold ${Number(result.chest_threshold).toFixed(2)})`],
      ['Data sent to server', 'Image only (name and age stay in this browser)'],
    ]);
    $('printBtn').addEventListener('click', () => window.print());
    return;
  }

  // ---------- Result ----------

  const cls = result.predicted_class;
  const conf = Number(result.confidence) || 0;
  $('verdict').style.setProperty('--class-color', colorFor(cls));
  $('predictedClass').textContent = labelFor(cls);
  $('confidenceValue').textContent = fmtPct(conf);

  // The TB class failed an external test (results/external_tbx11k/REPORT.md)
  $('tbWarning').hidden = cls !== 'TB';

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
  }

  const probs = Object.entries(result.class_probabilities || {}).sort((a, b) => b[1] - a[1]);
  const list = $('probBars');
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

  // ---------- About the condition ----------

  const info = DISEASE_INFO[cls];
  if (info) {
    $('aboutDisease').style.setProperty('--class-color', colorFor(cls));
    $('aboutTitle').textContent = info.title;
    info.lines.forEach((line) => {
      const li = document.createElement('li');
      li.textContent = line;
      $('aboutLines').appendChild(li);
    });
  } else {
    $('aboutDisease').hidden = true;
  }

  // ---------- Grad-CAM viewer ----------

  const stage = $('stage');
  const imgBase = $('imgBase');
  const imgBase2 = $('imgBase2');
  const modeButtons = document.querySelectorAll('#toolbar [data-mode]');
  // The full-size upload when it fit in storage, otherwise the 224x224 model input
  const originalUrl = image.dataUrl || result.model_input_base64;
  const modelInputUrl = result.model_input_base64;

  // Each frame takes the aspect ratio of the image it shows, so any view fills the viewer
  [imgBase, imgBase2].forEach((img) => img.addEventListener('load', () => {
    if (img.naturalWidth && img.naturalHeight) {
      img.parentElement.style.setProperty('--aspect', String(img.naturalWidth / img.naturalHeight));
    }
  }));

  $('imgHeat').src = result.heatmap_base64;
  $('imgHeat2').src = result.heatmap_base64;

  function setMode(mode) {
    stage.dataset.mode = mode;
    modeButtons.forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === mode)));
    const base = mode === 'input' ? modelInputUrl : originalUrl;
    if (imgBase.src !== base) imgBase.src = base;
    if (imgBase2.src !== originalUrl) imgBase2.src = originalUrl;
  }
  modeButtons.forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  setMode('overlay');

  const slider = $('opacitySlider');
  const applyOpacity = () => {
    $('opacityVal').textContent = `${slider.value}%`;
    stage.style.setProperty('--heat-opacity', String(Number(slider.value) / 100));
  };
  slider.addEventListener('input', applyOpacity);
  applyOpacity();

  // ---------- Technical details and actions ----------

  fillDetails([
    ['Model', result.model || '—'],
    ['Task', result.task === 'binary' ? 'Normal vs Pneumonia' : `${probs.length}-class`],
    ['Grad-CAM layer', result.target_layer || '—'],
    ['Preprocessing', 'Grayscale → 224×224 → CLAHE'],
    ['Chest X-ray check', result.chest_score == null ? 'Not run' : `Passed (similarity ${Number(result.chest_score).toFixed(2)})`],
    ['Data sent to server', 'Image only (name and age stay in this browser)'],
  ]);

  $('printBtn').addEventListener('click', () => window.print());
});
