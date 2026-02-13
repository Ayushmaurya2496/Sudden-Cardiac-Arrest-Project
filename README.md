# ECG Holter Beat Prediction (Same Technology Stack)

This project replicates the same stack as your reference project:
- **Python + XGBoost** for model training and prediction
- **Node.js + Express + EJS** for the prediction web UI

## 1) Install dependencies

### Python (inside your existing `.venv`)
```bash
.venv\Scripts\python.exe -m pip install -r python/requirements.txt
```

### Node.js
```bash
npm install
```

## 2) Train model

```bash
.venv\Scripts\python.exe python/train_model.py
```

This creates:
- `python/ecg_xgboost_model.pkl`
- `python/model_meta.json`
- `python/training_report.txt`

`model_meta.json` includes feature names and class labels used by UI + inference.

## 3) Run web app

```bash
npm start
```

Open: `http://localhost:3000`

The form inputs are generated automatically from `model_meta.json` feature columns, so they always match model input dataframe columns.

## 4) Optional custom Python path

If needed, set env variable before start:

```bash
set PYTHON_BIN=C:\path\to\python.exe
npm start
```
