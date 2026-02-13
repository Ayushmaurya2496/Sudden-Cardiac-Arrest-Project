const express = require('express');
const path = require('path');
const fs = require('fs');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_DIR = path.join(__dirname, 'python');
const MODEL_PATH = path.join(PYTHON_DIR, 'ecg_xgboost_model.pkl');
const META_PATH = path.join(PYTHON_DIR, 'model_meta.json');

const defaultPython = process.platform === 'win32'
  ? path.join(__dirname, '.venv', 'Scripts', 'python.exe')
  : path.join(__dirname, '.venv', 'bin', 'python');
const PYTHON_BIN = process.env.PYTHON_BIN || (fs.existsSync(defaultPython) ? defaultPython : 'python');

if (!fs.existsSync(META_PATH)) {
  throw new Error('Missing model metadata. Run python/train_model.py first.');
}

const meta = JSON.parse(fs.readFileSync(META_PATH, 'utf8'));
const featureNames = meta.feature_names;
const labelMap = meta.label_map;

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));
app.use(bodyParser.urlencoded({ extended: true }));

app.get('/', (req, res) => {
  res.render('index', { featureNames, prediction: null, errors: [], values: {} });
});

app.post('/predict', async (req, res) => {
  const values = { ...req.body };
  const { payload, errors } = buildPayload(req.body);

  if (errors.length) {
    return res.render('index', { featureNames, prediction: null, errors, values });
  }

  try {
    const prediction = await runPython(payload);
    res.render('index', { featureNames, prediction, errors: [], values });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    res.render('index', { featureNames, prediction: null, errors: [message], values });
  }
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});

function buildPayload(formBody) {
  const payload = { features: {} };
  const errors = [];

  featureNames.forEach((name) => {
    const rawValue = formBody[name];
    if (rawValue === undefined || rawValue === '') {
      errors.push(`${name} is required`);
      return;
    }

    const numericValue = Number(rawValue);
    if (Number.isNaN(numericValue)) {
      errors.push(`${name} must be numeric`);
    } else {
      payload.features[name] = numericValue;
    }
  });

  return { payload, errors };
}

function runPython(payload) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(PYTHON_DIR, 'predict.py');
    const child = spawn(PYTHON_BIN, [scriptPath], { cwd: __dirname });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('close', (code) => {
      if (stderr) {
        return reject(new Error(stderr.trim()));
      }

      try {
        const result = JSON.parse(stdout);
        if (result.error) {
          return reject(new Error(result.error));
        }

        const response = {
          ...result,
          readableLabel: labelMap[String(result.label_id)] || result.label,
        };
        return resolve(response);
      } catch (err) {
        return reject(err);
      }
    });

    child.on('error', (err) => {
      reject(err);
    });

    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}
