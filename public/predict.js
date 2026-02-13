(() => {
  const analyzeButton = document.getElementById('analyze-image-btn');
  const imageInput = document.getElementById('ecg-image-input');
  const statusEl = document.getElementById('analyze-status');
  const predictForm = document.getElementById('predict-form');

  if (!analyzeButton || !imageInput || !statusEl) {
    return;
  }

  function setStatus(message, isError = false) {
    statusEl.textContent = message;
    statusEl.classList.toggle('is-error', isError);
    statusEl.classList.toggle('is-success', !isError && Boolean(message));
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
    setStatus('Analyzing image with AI...');

    try {
      const response = await fetch('/analyze-image', {
        method: 'POST',
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
    } catch (_error) {
      setStatus('Could not connect to analyzer. Please try again.', true);
    } finally {
      analyzeButton.disabled = false;
    }
  }

  analyzeButton.addEventListener('click', handleAnalyzeClick);
})();
