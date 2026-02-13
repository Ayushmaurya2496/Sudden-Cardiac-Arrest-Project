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

### MongoDB Atlas connection

Create `.env` in project root and add:

```bash
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.9g4nsay.mongodb.net/arrhythmia?retryWrites=true&w=majority&appName=Cluster0
SESSION_SECRET=<any-random-secret>
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

## 5) Login and roles

The app now includes a role-based login system with session authentication backed by MongoDB.

- `admin` can access dashboard + prediction
- `doctor` can access dashboard + prediction
- `patient` can access dashboard only

Demo accounts:

- Username: `admin` | Password: `admin123`
- Username: `doctor` | Password: `doctor123`
- Username: `patient` | Password: `patient123`

Security and registration:

- Passwords are stored using `bcrypt` hash (not plaintext).
- Legacy plaintext passwords are auto-upgraded to hashed passwords on startup/login.
- New users can register from `/register` as `doctor` or `patient`.

## 4) Optional custom Python path

If needed, set env variable before start:

```bash
set PYTHON_BIN=C:\path\to\python.exe
npm start
```
