(() => {
  const analyzeButton = document.getElementById('analyze-image-btn');
  const imageInput = document.getElementById('ecg-image-input');
  const statusEl = document.getElementById('analyze-status');
  const predictForm = document.getElementById('predict-form');
  const aiSkeleton = document.getElementById('ai-skeleton');
  const waveformPath = document.getElementById('ecg-waveform-path');
  const waveformCaption = document.getElementById('ecg-waveform-caption');
  const csrfMeta = document.querySelector('meta[name="csrf-token"]');
  const csrfToken = csrfMeta ? csrfMeta.getAttribute('content') : '';

  const fallbackWaveformPath = 'M0 70 L60 70 L78 48 L90 92 L104 30 L116 82 L132 70 L198 70 L214 58 L226 80 L240 22 L254 88 L274 70 L342 70 L358 55 L370 82 L386 34 L400 86 L418 70 L480 70';

  if (!analyzeButton || !imageInput || !statusEl) {
    return;
  }

  function applyPredictionBarStyles() {
    const bars = document.querySelectorAll('.probability-bar-fill[data-width]');
    bars.forEach((bar) => {
      const width = Number(bar.getAttribute('data-width'));
      const glowOpacity = Number(bar.getAttribute('data-glow-opacity'));
      const glowSize = Number(bar.getAttribute('data-glow-size'));
      const glowRgb = String(bar.getAttribute('data-glow-rgb') || '148, 163, 184');

      if (!Number.isFinite(width)) {
        return;
      }

      const clampedWidth = Math.max(0, Math.min(100, width));
      const normalized = clampedWidth / 100;
      const clampedOpacity = Number.isFinite(glowOpacity)
        ? Math.max(0, Math.min(0.9, glowOpacity))
        : Math.min(0.9, 0.12 + normalized * 0.65);
      const clampedSize = Number.isFinite(glowSize)
        ? Math.max(0, glowSize)
        : Math.round(5 + normalized * 18);

      bar.style.width = `${clampedWidth}%`;
      bar.style.setProperty('--bar-glow-opacity', String(clampedOpacity));
      bar.style.setProperty('--bar-glow-size', `${clampedSize}px`);
      bar.style.boxShadow = `0 0 ${clampedSize}px rgba(${glowRgb}, ${clampedOpacity}), 0 0 ${Math.max(0, Math.round(clampedSize * 0.45))}px rgba(255, 255, 255, ${Math.min(0.22, clampedOpacity * 0.35)}) inset`;
    });
  }

  function setStatus(message, isError = false) {
    statusEl.textContent = message;
    statusEl.classList.toggle('is-error', isError);
    statusEl.classList.toggle('is-success', !isError && Boolean(message));
  }

  function collectWaveformValues() {
    if (!predictForm) {
      return [];
    }

    const inputs = Array.from(predictForm.querySelectorAll('input[type="number"]'));
    const values = inputs
      .map((input) => ({
        name: input.name || '',
        value: Number(input.value),
      }))
      .filter((item) => Number.isFinite(item.value));

    if (!values.length) {
      return [];
    }

    const priorityPattern = /(rPeak|qPeak|sPeak|tPeak|pPeak|qrs_interval|pq_interval|qt_interval|st_interval|pre-RR|post-RR)/i;
    const prioritized = values.filter((item) => priorityPattern.test(item.name));
    const source = prioritized.length >= 8 ? prioritized : values;
    return source.slice(0, 64).map((item) => item.value);
  }

  function buildWaveformPath(values) {
    const width = 480;
    const height = 120;
    const padX = 8;
    const padY = 12;
    const usableWidth = width - padX * 2;
    const usableHeight = height - padY * 2;

    if (!Array.isArray(values) || values.length < 2) {
      return null;
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min;

    const normalize = (value) => {
      if (span <= 0) {
        return 0.5;
      }
      return (value - min) / span;
    };

    return values
      .map((value, index) => {
        const x = padX + (index * usableWidth) / (values.length - 1);
        const y = padY + (1 - normalize(value)) * usableHeight;
        return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(' ');
  }

  function renderFeatureWaveform() {
    if (!waveformPath) {
      return;
    }

    const values = collectWaveformValues();
    const dynamicPath = buildWaveformPath(values);

    if (!dynamicPath) {
      waveformPath.setAttribute('d', fallbackWaveformPath);
      if (waveformCaption) {
        waveformCaption.textContent = 'Enter or analyze features to generate waveform';
      }
      return;
    }

    waveformPath.setAttribute('d', dynamicPath);
    if (waveformCaption) {
      waveformCaption.textContent = `Waveform generated from ${values.length} feature values`;
    }
  }

  async function handleAnalyzeClick() {
    const file = imageInput.files && imageInput.files[0];
    if (!file) {
      setStatus('Please choose an image first.', true);
      return;
    }

    const formData = new FormData();
    formData.append('ecgImage', file);

    analyzeButton.disabled = true;
    if (aiSkeleton) {
      aiSkeleton.hidden = false;
    }
    setStatus('Analyzing image with AI...');

    try {
      const response = await fetch('/analyze-image', {
        method: 'POST',
        headers: {
          'CSRF-Token': csrfToken || '',
        },
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        setStatus(payload.error || 'Image analysis failed.', true);
        return;
      }

      const features = payload.features || {};
      let fillCount = 0;
      Object.entries(features).forEach(([key, value]) => {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
          return;
        }

        const previewInput = document.querySelector(`.clinical-grid input[data-feature="${key}"]`);
        if (previewInput) {
          previewInput.value = value;
        }

        const targetInput = predictForm
          ? predictForm.querySelector(`input[name="${key}"]`)
          : document.querySelector(`input[name="${key}"]`);

        if (targetInput) {
          targetInput.value = value;
        }

        fillCount += 1;
      });

      if (fillCount === 0) {
        setStatus('Analysis complete, but no valid numeric features were found.', true);
      } else {
        setStatus(`Analysis complete. ${fillCount} main form fields auto-filled.`);
      }

      renderFeatureWaveform();
    } catch (_error) {
      setStatus('Could not connect to analyzer. Please try again.', true);
    } finally {
      analyzeButton.disabled = false;
      if (aiSkeleton) {
        aiSkeleton.hidden = true;
      }
    }
  }

  if (predictForm && waveformPath) {
    predictForm.addEventListener('input', renderFeatureWaveform);
    renderFeatureWaveform();
  }

  applyPredictionBarStyles();
  window.addEventListener('load', applyPredictionBarStyles);
  window.addEventListener('pageshow', applyPredictionBarStyles);
  window.setTimeout(applyPredictionBarStyles, 80);

  analyzeButton.addEventListener('click', handleAnalyzeClick);
})();

(() => {
  const svg = document.getElementById('feature-interactive-chart');
  const shell = document.getElementById('feature-svg-shell');
  const tooltip = document.getElementById('feature-chart-tooltip');
  const filter = document.getElementById('feature-group-filter');
  const dataNode = document.getElementById('feature-pair-series-data');

  let rawSeries = [];
  if (dataNode) {
    try {
      const parsed = JSON.parse(dataNode.getAttribute('data-series') || '[]');
      rawSeries = Array.isArray(parsed) ? parsed : [];
    } catch (_error) {
      rawSeries = [];
    }
  }

  if (!svg || !shell || !rawSeries.length) {
    return;
  }

  function createSvgElement(tagName, attrs = {}) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tagName);
    Object.entries(attrs).forEach(([key, value]) => {
      el.setAttribute(key, String(value));
    });
    return el;
  }

  function valueToText(value) {
    if (value === null || value === undefined || !Number.isFinite(value)) {
      return 'N/A';
    }
    return Number(value).toFixed(4);
  }

  function showTooltip(event, item) {
    if (!tooltip) {
      return;
    }

    tooltip.innerHTML = `
      <strong>${item.metric}</strong><br>
      Lead 0: ${valueToText(item.lead0)}<br>
      Lead 1: ${valueToText(item.lead1)}
    `;

    const shellRect = shell.getBoundingClientRect();
    const nextLeft = event.clientX - shellRect.left + 12;
    const nextTop = event.clientY - shellRect.top + 12;
    tooltip.style.left = `${nextLeft}px`;
    tooltip.style.top = `${nextTop}px`;
    tooltip.hidden = false;
  }

  function hideTooltip() {
    if (!tooltip) {
      return;
    }
    tooltip.hidden = true;
  }

  function render() {
    const selectedGroup = filter?.value || 'all';
    const data = selectedGroup === 'all'
      ? rawSeries
      : rawSeries.filter((item) => item.category === selectedGroup);

    const validData = data.filter((item) => item.lead0 !== null || item.lead1 !== null);
    const width = Math.max(760, shell.clientWidth || 760);
    const rowHeight = 34;
    const labelWidth = 170;
    const padding = { top: 20, right: 26, bottom: 26, left: 10 };
    const height = Math.max(220, padding.top + validData.length * rowHeight + padding.bottom);

    svg.innerHTML = '';
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', String(height));

    if (!validData.length) {
      const emptyText = createSvgElement('text', {
        x: width / 2,
        y: height / 2,
        fill: '#a8b4d6',
        'text-anchor': 'middle',
        'font-size': '14',
      });
      emptyText.textContent = 'No data for selected feature group';
      svg.appendChild(emptyText);
      hideTooltip();
      return;
    }

    const allNumbers = validData
      .flatMap((item) => [item.lead0, item.lead1])
      .filter((value) => Number.isFinite(value));

    const maxAbs = Math.max(1, ...allNumbers.map((value) => Math.abs(value)));
    const domain = maxAbs * 1.1;
    const plotStart = labelWidth;
    const plotEnd = width - padding.right;
    const plotWidth = plotEnd - plotStart;
    const centerX = plotStart + plotWidth / 2;

    const scaleX = (value) => {
      const normalized = (value + domain) / (2 * domain);
      return plotStart + normalized * plotWidth;
    };

    const axisLine = createSvgElement('line', {
      x1: centerX,
      y1: padding.top - 8,
      x2: centerX,
      y2: height - padding.bottom + 8,
      stroke: 'rgba(168, 180, 214, 0.45)',
      'stroke-width': 1,
      'stroke-dasharray': '4 4',
    });
    svg.appendChild(axisLine);

    const tickValues = [-domain, 0, domain];
    tickValues.forEach((tick) => {
      const x = scaleX(tick);
      const tickText = createSvgElement('text', {
        x,
        y: 12,
        fill: '#a8b4d6',
        'text-anchor': 'middle',
        'font-size': 11,
      });
      tickText.textContent = tick === 0 ? '0' : tick.toFixed(1);
      svg.appendChild(tickText);
    });

    validData.forEach((item, index) => {
      const y = padding.top + index * rowHeight + rowHeight / 2;

      const rowHit = createSvgElement('rect', {
        x: 0,
        y: y - rowHeight / 2,
        width,
        height: rowHeight,
        fill: 'transparent',
      });
      rowHit.addEventListener('mousemove', (event) => showTooltip(event, item));
      rowHit.addEventListener('mouseleave', hideTooltip);
      svg.appendChild(rowHit);

      const metricLabel = createSvgElement('text', {
        x: 8,
        y: y + 4,
        fill: '#f4f7ff',
        'font-size': 13,
      });
      metricLabel.textContent = item.metric;
      svg.appendChild(metricLabel);

      const rowGuide = createSvgElement('line', {
        x1: plotStart,
        y1: y,
        x2: plotEnd,
        y2: y,
        stroke: 'rgba(157, 178, 255, 0.2)',
        'stroke-width': 1,
      });
      svg.appendChild(rowGuide);

      const x0 = Number.isFinite(item.lead0) ? scaleX(item.lead0) : null;
      const x1 = Number.isFinite(item.lead1) ? scaleX(item.lead1) : null;

      if (x0 !== null && x1 !== null) {
        svg.appendChild(createSvgElement('line', {
          x1: x0,
          y1: y,
          x2: x1,
          y2: y,
          stroke: 'rgba(168, 180, 214, 0.5)',
          'stroke-width': 2,
        }));
      }

      if (x0 !== null) {
        svg.appendChild(createSvgElement('circle', {
          cx: x0,
          cy: y,
          r: 5,
          fill: '#35e7ff',
        }));
      }

      if (x1 !== null) {
        svg.appendChild(createSvgElement('circle', {
          cx: x1,
          cy: y,
          r: 5,
          fill: '#8b7bff',
        }));
      }
    });
  }

  if (filter) {
    filter.addEventListener('change', render);
  }

  window.addEventListener('resize', render);
  render();
})();
