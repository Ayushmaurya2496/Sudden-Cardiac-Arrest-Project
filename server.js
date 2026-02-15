require('dotenv').config();
const express = require('express');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const bodyParser = require('body-parser');
const session = require('express-session');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const multer = require('multer');
const cors = require('cors');
const helmet = require('helmet');
const csrf = require('@dr.pogodin/csurf');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;
const IS_PRODUCTION = process.env.NODE_ENV === 'production';
const TRUST_PROXY = process.env.TRUST_PROXY !== 'false' && IS_PRODUCTION;
const MONGO_URI = process.env.MONGO_URI;
const PYTHON_DIR = path.join(__dirname, 'python');
const MODEL_PATH = path.join(PYTHON_DIR, 'ecg_xgboost_model.pkl');
const META_PATH = path.join(PYTHON_DIR, 'model_meta.json');
const UPLOAD_DIR = path.join(__dirname, 'public', 'uploads');
const MAX_UPLOAD_BYTES = Number(process.env.MAX_UPLOAD_BYTES || 5 * 1024 * 1024);
const GEMINI_RETRIES = Number(process.env.GEMINI_RETRIES || 3);
const GEMINI_RETRY_BASE_MS = Number(process.env.GEMINI_RETRY_BASE_MS || 500);
const ENABLE_MOCK_VISION = process.env.ENABLE_MOCK_VISION !== 'false';
const MODEL_CANDIDATES = [
  process.env.GOOGLE_MODEL,
  'gemini-2.5-flash',
  'gemini-2.5-flash-latest',
  'gemini-2.5-flash-001',
  'gemini-1.5-flash',
  'gemini-1.5-flash-latest',
  'gemini-1.5-flash-8b',
  'gemini-1.5-pro',
  'gemini-1.0-pro-vision',
  'gemini-1.0-pro',
].filter(Boolean);
const TEXT_MODEL_CANDIDATES = [
  process.env.GOOGLE_TEXT_MODEL,
  'gemini-1.5-flash-latest',
  'gemini-1.5-pro',
  'gemini-2.0-flash-exp',
].filter(Boolean);

const defaultPython = process.platform === 'win32'
  ? path.join(__dirname, '.venv', 'Scripts', 'python.exe')
  : path.join(__dirname, '.venv', 'bin', 'python');
const PYTHON_BIN = process.env.PYTHON_BIN || (fs.existsSync(defaultPython) ? defaultPython : 'python');
const genAI = process.env.GOOGLE_API_KEY ? new GoogleGenerativeAI(process.env.GOOGLE_API_KEY) : null;

if (!process.env.GOOGLE_API_KEY) {
  console.warn('Warning: GOOGLE_API_KEY not set. Image analysis will use mock data when enabled.');
}

if (!fs.existsSync(META_PATH)) {
  throw new Error('Missing model metadata. Run python/train_model.py first.');
}

const meta = JSON.parse(fs.readFileSync(META_PATH, 'utf8'));
const featureNames = meta.feature_names;
const FEATURE_FIELDS = Array.isArray(featureNames) ? featureNames : [];
const labelMap = meta.label_map;
const CLASS_SUMMARIES = {
  F: 'Fusion beats blend ventricular and supraventricular impulses; usually benign but signal mixed conduction.',
  N: 'Normal sinus beat with expected atrioventricular conduction and morphology.',
  Q: 'Unclassifiable/artifact beat; the tracing is noisy or atypical and needs manual confirmation.',
  SVEB: 'Supraventricular ectopic beat arising above the ventricles, often narrow and premature.',
  VEB: 'Ventricular ectopic beat with wide QRS indicating ventricular origin; monitor for frequency or runs.',
};
const MODEL_DISPLAY_NAME = 'ECG XGBoost Classifier';
const classDescriptionCache = new Map();
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
    filename: (_req, file, cb) => {
      const ext = path.extname(file.originalname || '').toLowerCase();
      const safeBase = path.basename(file.originalname || 'image', ext).replace(/[^a-zA-Z0-9_-]/g, '_');
      cb(null, `${Date.now()}-${safeBase || 'image'}${ext || '.png'}`);
    },
  }),
  limits: {
    fileSize: MAX_UPLOAD_BYTES,
  },
  fileFilter: (_req, file, cb) => {
    if (!file.mimetype || !file.mimetype.startsWith('image/')) {
      return cb(new Error('Only image files are allowed.'));
    }
    return cb(null, true);
  },
});

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(cors());
app.use(helmet());

if (TRUST_PROXY) {
  app.set('trust proxy', 1);
}

app.use(session({
  secret: process.env.SESSION_SECRET || 'ecg-final-year-project-secret',
  resave: false,
  saveUninitialized: false,
  proxy: TRUST_PROXY,
  cookie: {
    maxAge: 1000 * 60 * 60 * 8,
    httpOnly: true,
    sameSite: 'lax',
    secure: IS_PRODUCTION,
  },
}));

app.use(csrf());

const seedUsers = [
  { username: 'admin', password: 'admin123', role: 'admin', fullName: 'System Admin' },
  { username: 'doctor', password: 'doctor123', role: 'doctor', fullName: 'Dr. Demo' },
  { username: 'patient', password: 'patient123', role: 'patient', fullName: 'Patient Demo' },
];

const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true, trim: true },
  password: { type: String, required: true },
  role: { type: String, enum: ['admin', 'doctor', 'patient'], required: true },
  fullName: { type: String, required: true },
}, { timestamps: true });

const User = mongoose.models.User || mongoose.model('User', userSchema);

const predictionHistorySchema = new mongoose.Schema({
  username: { type: String, required: true, index: true },
  userDisplayName: { type: String, required: true },
  role: { type: String, enum: ['admin', 'doctor', 'patient'], required: true },
  features: { type: mongoose.Schema.Types.Mixed, required: true },
  predictionLabel: { type: String, required: true },
  labelId: { type: Number, required: true },
  probabilities: { type: mongoose.Schema.Types.Mixed, default: null },
  description: { type: String, default: null },
}, { timestamps: true });

const PredictionHistory = mongoose.models.PredictionHistory
  || mongoose.model('PredictionHistory', predictionHistorySchema);

app.use((req, res, next) => {
  res.locals.currentUser = req.session.user || null;
  res.locals.csrfToken = req.csrfToken();
  next();
});

app.get('/', requireAuth, (req, res) => {
  res.redirect('/dashboard');
});

app.get('/login', (req, res) => {
  if (req.session.user) {
    return res.redirect('/dashboard');
  }

  return res.render('login', { error: null, username: '' });
});

app.get('/register', (req, res) => {
  if (req.session.user) {
    return res.redirect('/dashboard');
  }

  return res.render('register', { error: null, success: null, values: {} });
});

app.post('/login', async (req, res) => {
  const username = (req.body.username || '').trim();
  const password = req.body.password || '';

  let matchedUser;
  try {
    matchedUser = await User.findOne({ username });
  } catch (error) {
    return res.status(500).render('login', {
      error: 'Database error during login. Please try again.',
      username,
    });
  }

  const isPasswordValid = await verifyAndUpgradePassword(matchedUser, password);

  if (!matchedUser || !isPasswordValid) {
    return res.status(401).render('login', {
      error: 'Invalid username or password',
      username,
    });
  }

  req.session.user = {
    username: matchedUser.username,
    role: matchedUser.role,
    fullName: matchedUser.fullName,
  };

  return res.redirect('/dashboard');
});

app.post('/register', async (req, res) => {
  const fullName = (req.body.fullName || '').trim();
  const username = (req.body.username || '').trim();
  const password = req.body.password || '';
  const confirmPassword = req.body.confirmPassword || '';
  const role = (req.body.role || 'patient').trim();

  const values = { fullName, username, role };

  if (!fullName || !username || !password || !confirmPassword || !role) {
    return res.status(400).render('register', {
      error: 'All fields are required.',
      success: null,
      values,
    });
  }

  if (!['doctor', 'patient'].includes(role)) {
    return res.status(400).render('register', {
      error: 'Invalid role selected.',
      success: null,
      values,
    });
  }

  if (password.length < 8) {
    return res.status(400).render('register', {
      error: 'Password must be at least 8 characters long.',
      success: null,
      values,
    });
  }

  if (password !== confirmPassword) {
    return res.status(400).render('register', {
      error: 'Password and confirm password do not match.',
      success: null,
      values,
    });
  }

  try {
    const existingUser = await User.findOne({ username });
    if (existingUser) {
      return res.status(409).render('register', {
        error: 'Username already exists. Please choose another.',
        success: null,
        values,
      });
    }

    const hashedPassword = await bcrypt.hash(password, 12);
    await User.create({
      fullName,
      username,
      password: hashedPassword,
      role,
    });

    return res.render('register', {
      error: null,
      success: 'Registration successful. You can now login.',
      values: {},
    });
  } catch (error) {
    return res.status(500).render('register', {
      error: 'Could not register user. Please try again.',
      success: null,
      values,
    });
  }
});

app.post('/logout', requireAuth, (req, res) => {
  req.session.destroy(() => {
    res.redirect('/login');
  });
});

app.get('/dashboard', requireAuth, async (req, res) => {
  const roleActions = {
    admin: ['Monitor all users', 'Review prediction activity', 'Manage system settings'],
    doctor: ['Run ECG beat predictions', 'Review patient trends', 'Track uncertain predictions'],
    patient: ['View your profile', 'Check your prediction history', 'Consult your doctor'],
  };

  const activeRole = req.session.user?.role;
  const canPredict = true;

  const filter = activeRole === 'patient'
    ? { username: req.session.user.username }
    : {};

  let dashboardData = {
    totalScans: 0,
    highRiskCount: 0,
    moderateCount: 0,
    normalCount: 0,
    classCounts: {
      N: 0,
      SVEB: 0,
      VEB: 0,
      F: 0,
      Q: 0,
    },
    recentPredictions: [],
    highRiskAlerts: [],
    trendLabels: [],
    trendScores: [],
    modelStatus: {
      isActive: true,
      accuracy: 97,
      lastScanAt: null,
    },
  };

  function computeRiskScore(entry) {
    const probabilities = entry?.probabilities && typeof entry.probabilities === 'object'
      ? entry.probabilities
      : null;
    if (probabilities && probabilities.VEB !== undefined) {
      const veb = Number(probabilities.VEB);
      if (Number.isFinite(veb)) {
        return Math.max(0, Math.min(100, veb * 100));
      }
    }

    const label = String(entry?.predictionLabel || '').toUpperCase();
    const fallback = {
      VEB: 95,
      SVEB: 65,
      F: 75,
      Q: 45,
      N: 20,
    };
    return fallback[label] || 35;
  }

  function normalizePredictionLabel(value) {
    const label = String(value || '').trim().toUpperCase();
    if (label === 'N' || label.includes('NORMAL')) return 'N';
    if (label === 'SVEB' || label.includes('SUPRAVENT')) return 'SVEB';
    if (label === 'VEB' || label.includes('VENTRIC')) return 'VEB';
    if (label === 'F' || label.includes('FUSION')) return 'F';
    if (label === 'Q' || label.includes('UNKNOWN') || label.includes('UNCLASS')) return 'Q';
    return 'Q';
  }

  try {
    const [totalScans, allLabelEntries, recentEntries] = await Promise.all([
      PredictionHistory.countDocuments(filter),
      PredictionHistory.find(filter)
        .select({ predictionLabel: 1 })
        .lean(),
      PredictionHistory.find(filter)
        .sort({ createdAt: -1 })
        .limit(24)
        .lean(),
    ]);

    const classCounts = {
      N: 0,
      SVEB: 0,
      VEB: 0,
      F: 0,
      Q: 0,
    };

    allLabelEntries.forEach((entry) => {
      const normalized = normalizePredictionLabel(entry?.predictionLabel);
      classCounts[normalized] += 1;
    });

    const highRiskCount = classCounts.VEB;
    const moderateCount = classCounts.SVEB;
    const normalCount = classCounts.N;

    const recentPredictions = recentEntries.slice(0, 3);
    const highRiskAlerts = recentEntries
      .filter((entry) => normalizePredictionLabel(entry?.predictionLabel) === 'VEB')
      .slice(0, 4);

    const trendEntries = recentEntries.slice(0, 12).reverse();
    const trendLabels = trendEntries.map((entry) => new Date(entry.createdAt).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    }));
    const trendScores = trendEntries.map((entry) => Number(computeRiskScore(entry).toFixed(1)));

    dashboardData = {
      totalScans,
      highRiskCount,
      moderateCount,
      normalCount,
      classCounts,
      recentPredictions,
      highRiskAlerts,
      trendLabels,
      trendScores,
      modelStatus: {
        isActive: true,
        accuracy: 97,
        lastScanAt: recentEntries[0]?.createdAt || null,
      },
    };
  } catch (error) {
    console.error('Failed to build dashboard data:', error.message);
  }

  res.render('dashboard', {
    actions: roleActions[activeRole] || [],
    canPredict,
    dashboardData,
  });
});

app.get('/history', requireAuth, async (req, res) => {
  try {
    const entries = await PredictionHistory.find({ username: req.session.user.username })
      .sort({ createdAt: -1 })
      .lean();

    res.render('history', {
      entries,
    });
  } catch (error) {
    console.error('Failed to load history:', error.message);
    res.status(500).render('history', {
      entries: [],
      error: 'Could not load prediction history. Please try again later.',
    });
  }
});

app.post('/history/:entryId/delete', requireAuth, async (req, res) => {
  try {
    await PredictionHistory.deleteOne({
      _id: req.params.entryId,
      username: req.session.user.username,
    });
    res.redirect('/history');
  } catch (error) {
    console.error('Failed to delete history entry:', error.message);
    res.status(500).redirect('/history');
  }
});

app.get('/predict', requireAuth, (req, res) => {
  res.render('index', {
    featureNames,
    prediction: null,
    predictionDescription: null,
    errors: [],
    values: {},
    uploadError: null,
    uploadSuccess: null,
    uploadedImageUrl: null,
    submittedFeatures: null,
  });
});

app.get('/index', requireAuth, (req, res) => {
  res.redirect('/predict');
});

app.post('/upload-image', requireAuth, (req, res) => {
  upload.single('ecgImage')(req, res, (error) => {
    if (error) {
      const message = error instanceof multer.MulterError
        ? `Upload failed: ${error.message}`
        : error.message;

      return res.status(400).render('index', {
        featureNames,
        prediction: null,
        predictionDescription: null,
        errors: [],
        values: {},
        uploadError: message,
        uploadSuccess: null,
        uploadedImageUrl: null,
        submittedFeatures: null,
      });
    }

    if (!req.file) {
      return res.status(400).render('index', {
        featureNames,
        prediction: null,
        predictionDescription: null,
        errors: [],
        values: {},
        uploadError: 'Please select an image to upload.',
        uploadSuccess: null,
        uploadedImageUrl: null,
        submittedFeatures: null,
      });
    }

    return res.render('index', {
      featureNames,
      prediction: null,
      predictionDescription: null,
      errors: [],
      values: {},
      uploadError: null,
      uploadSuccess: `Image uploaded successfully: ${req.file.originalname}`,
      uploadedImageUrl: `/uploads/${req.file.filename}`,
      submittedFeatures: null,
    });
  });
});

app.post('/analyze-image', requireAuth, (req, res) => {
  upload.single('ecgImage')(req, res, async (error) => {
    if (error) {
      const message = error instanceof multer.MulterError
        ? `Upload failed: ${error.message}`
        : error.message;
      return res.status(400).json({ error: message });
    }

    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    try {
      const features = await extractFeaturesWithGoogleLLM(req.file);
      try {
        fs.unlinkSync(req.file.path);
      } catch (_cleanupError) {
      }
      return res.json({ features });
    } catch (analysisError) {
      try {
        if (req.file && fs.existsSync(req.file.path)) {
          fs.unlinkSync(req.file.path);
        }
      } catch (_cleanupError) {
      }
      const message = analysisError instanceof Error ? analysisError.message : 'Failed to analyze image';
      return res.status(500).json({ error: message });
    }
  });
});

app.post('/predict', requireAuth, async (req, res) => {
  const values = { ...req.body };
  const { payload, errors } = buildPayload(req.body);

  if (errors.length) {
    return res.render('index', {
      featureNames,
      prediction: null,
      predictionDescription: null,
      errors,
      values,
      uploadError: null,
      uploadSuccess: null,
      uploadedImageUrl: null,
      submittedFeatures: payload.features,
    });
  }

  try {
    const prediction = await runPython(payload);
    let predictionDescription = null;
    try {
      predictionDescription = await describePredictionWithLLM(prediction);
    } catch (_descError) {
      predictionDescription = getFallbackClassSummary(prediction.readableLabel || prediction.label);
    }
    req.session.latestPredictionReport = {
      generatedAt: new Date().toISOString(),
      modelName: MODEL_DISPLAY_NAME,
      user: {
        fullName: req.session.user.fullName,
        username: req.session.user.username,
        role: req.session.user.role,
      },
      features: payload.features,
      prediction,
      predictionDescription,
    };
    await recordPredictionHistory(req.session.user, payload.features, prediction, predictionDescription);

    return req.session.save(() => res.render('index', {
      featureNames,
      prediction,
      predictionDescription,
      errors: [],
      values,
      uploadError: null,
      uploadSuccess: null,
      uploadedImageUrl: null,
      submittedFeatures: payload.features,
    }));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    res.render('index', {
      featureNames,
      prediction: null,
      predictionDescription: null,
      errors: [message],
      values,
      uploadError: null,
      uploadSuccess: null,
      uploadedImageUrl: null,
      submittedFeatures: payload.features,
    });
  }
});

app.get(['/predict/report', '/predict/report/', '/predict-report', '/report'], requireAuth, async (req, res) => {
  try {
    return await downloadPredictionReport(req, res);
  } catch (error) {
    console.error('Failed to generate prediction report:', error.message);
    return res.status(500).send('Could not generate prediction report. Please try again.');
  }
});

app.get('/forbidden', requireAuth, (req, res) => {
  const defaultDashboardData = {
    totalScans: 0,
    highRiskCount: 0,
    moderateCount: 0,
    normalCount: 0,
    classCounts: {
      N: 0,
      SVEB: 0,
      VEB: 0,
      F: 0,
      Q: 0,
    },
    recentPredictions: [],
    highRiskAlerts: [],
    trendLabels: [],
    trendScores: [],
    modelStatus: {
      isActive: false,
      accuracy: 0,
      lastScanAt: null,
    },
  };

  res.status(403).render('dashboard', {
    actions: [],
    canPredict: false,
    dashboardData: defaultDashboardData,
    forbiddenMessage: 'You do not have permission to access that page.',
  });
});

app.use((error, req, res, next) => {
  if (error?.code !== 'EBADCSRFTOKEN') {
    return next(error);
  }

  const message = 'Invalid or expired security token. Please refresh and try again.';
  let safeCsrfToken = '';
  try {
    safeCsrfToken = req.csrfToken();
  } catch (_tokenError) {
    safeCsrfToken = '';
  }

  if (req.path === '/login') {
    return res.status(403).render('login', {
      error: message,
      username: (req.body.username || '').trim(),
      csrfToken: safeCsrfToken,
    });
  }

  if (req.path === '/register') {
    const role = (req.body.role || 'patient').trim();
    return res.status(403).render('register', {
      error: message,
      success: null,
      csrfToken: safeCsrfToken,
      values: {
        fullName: (req.body.fullName || '').trim(),
        username: (req.body.username || '').trim(),
        role,
      },
    });
  }

  const acceptsHtml = Boolean(req.accepts('html'));
  const acceptsJson = Boolean(req.accepts('json'));
  const prefersJson = acceptsJson && !acceptsHtml;
  if (req.path === '/analyze-image' || req.xhr || req.is('application/json') || prefersJson) {
    return res.status(403).json({ error: message });
  }

  if (req.session.user) {
    return res.status(403).redirect(req.get('referer') || '/dashboard');
  }

  return res.status(403).redirect('/login');
});

startServer();

function buildPayload(formBody) {
  const payload = { features: {} };
  const errors = [];

  FEATURE_FIELDS.forEach((name) => {
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

function requireAuth(req, res, next) {
  if (!req.session.user) {
    return res.redirect('/login');
  }

  return next();
}

function requireRole(allowedRoles) {
  return (req, res, next) => {
    const currentRole = req.session.user?.role;
    if (!currentRole || !allowedRoles.includes(currentRole)) {
      return res.redirect('/forbidden');
    }

    return next();
  };
}

async function startServer() {
  try {
    await connectDatabase();
    await ensureSeedUsers();
    await upgradeLegacyPlaintextPasswords();

    app.listen(PORT, () => {
      console.log(`Server listening on http://localhost:${PORT}`);
    });
  } catch (error) {
    console.error('Startup failed:', error.message);
    process.exit(1);
  }
}

async function connectDatabase() {
  if (!MONGO_URI) {
    throw new Error('MONGO_URI is missing. Add it in your .env file.');
  }

  await mongoose.connect(MONGO_URI);
  console.log('Connected to MongoDB Atlas');
}

async function ensureSeedUsers() {
  for (const seedUser of seedUsers) {
    const existingUser = await User.findOne({ username: seedUser.username });
    if (existingUser) {
      continue;
    }

    const hashedPassword = await bcrypt.hash(seedUser.password, 12);
    await User.create({
      ...seedUser,
      password: hashedPassword,
    });
  }

  console.log('Inserted default role users into MongoDB');
}

async function upgradeLegacyPlaintextPasswords() {
  const users = await User.find({});
  let upgradedCount = 0;

  for (const user of users) {
    if (isBcryptHash(user.password)) {
      continue;
    }

    user.password = await bcrypt.hash(user.password, 12);
    await user.save();
    upgradedCount += 1;
  }

  if (upgradedCount > 0) {
    console.log(`Upgraded ${upgradedCount} legacy plaintext password(s)`);
  }
}

async function verifyAndUpgradePassword(user, inputPassword) {
  if (!user) {
    return false;
  }

  if (isBcryptHash(user.password)) {
    return bcrypt.compare(inputPassword, user.password);
  }

  const isMatch = user.password === inputPassword;
  if (!isMatch) {
    return false;
  }

  user.password = await bcrypt.hash(inputPassword, 12);
  await user.save();
  return true;
}

function isBcryptHash(value) {
  return typeof value === 'string' && /^\$2[aby]\$\d{2}\$/.test(value);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function extractFirstJsonBlock(text) {
  const start = text.indexOf('{');
  if (start === -1) {
    return null;
  }

  let depth = 0;
  for (let index = start; index < text.length; index += 1) {
    if (text[index] === '{') {
      depth += 1;
    } else if (text[index] === '}') {
      depth -= 1;
    }

    if (depth === 0) {
      return text.substring(start, index + 1);
    }
  }

  return null;
}

function tolerantParseJson(raw) {
  if (!raw) {
    return null;
  }

  const rawText = String(raw);
  const fencedJsonMatch = rawText.match(/```\s*json\s*([\s\S]*?)```/i);
  if (fencedJsonMatch && fencedJsonMatch[1]) {
    const block = extractFirstJsonBlock(fencedJsonMatch[1].trim()) || fencedJsonMatch[1].trim();
    try {
      return JSON.parse(block);
    } catch (_error) {
    }
  }

  const fencedAnyMatch = rawText.match(/```\s*([\s\S]*?)```/i);
  if (fencedAnyMatch && fencedAnyMatch[1]) {
    const block = extractFirstJsonBlock(fencedAnyMatch[1].trim()) || fencedAnyMatch[1].trim();
    try {
      return JSON.parse(block);
    } catch (_error) {
    }
  }

  let cleaned = rawText.trim();
  cleaned = cleaned.replace(/```(json)?/gi, '').replace(/```/g, '').trim();
  cleaned = cleaned
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/\bNone\b/g, 'null')
    .replace(/\bNaN\b/g, 'null')
    .replace(/\bInfinity\b/g, 'null')
    .replace(/\bTrue\b/g, 'true')
    .replace(/\bFalse\b/g, 'false')
    .replace(/,\s*([}\]])/g, '$1');

  if (
    (cleaned.startsWith('"') && cleaned.endsWith('"'))
    || (cleaned.startsWith("'") && cleaned.endsWith("'"))
  ) {
    try {
      const unwrapped = JSON.parse(cleaned.replace(/^'/, '"').replace(/'$/, '"'));
      if (typeof unwrapped === 'string') {
        cleaned = unwrapped.trim();
      }
    } catch (_error) {
    }
  }

  const candidate = extractFirstJsonBlock(cleaned);
  if (candidate) {
    try {
      return JSON.parse(candidate);
    } catch (_error) {
      try {
        return JSON.parse(candidate.replace(/'/g, '"'));
      } catch (_errorAgain) {
        return null;
      }
    }
  }

  try {
    return JSON.parse(cleaned);
  } catch (_error) {
    return null;
  }
}

// Last-ditch attempt to salvage key/value pairs from malformed JSON blobs.
function recoverLooseKeyValuePairs(rawText) {
  if (typeof rawText !== 'string' || !rawText.trim()) {
    return null;
  }

  const result = {};
  const pairRegex = /"([^"\\]+)"\s*:\s*(?:"([^"\\]*)"|(-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)|(null)|(true)|(false))/gi;
  let match;

  while ((match = pairRegex.exec(rawText)) !== null) {
    const key = match[1];
    if (!key) {
      continue;
    }

    let value;
    if (match[2] !== undefined) {
      value = match[2];
    } else if (match[3] !== undefined) {
      value = Number(match[3]);
    } else if (match[4] !== undefined) {
      value = null;
    } else if (match[5] !== undefined) {
      value = true;
    } else if (match[6] !== undefined) {
      value = false;
    } else {
      continue;
    }

    result[key] = value;
  }

  return Object.keys(result).length ? result : null;
}

async function resolveGeminiResponse(result) {
  const maybeResponse = result?.response;
  if (!maybeResponse) {
    return null;
  }

  if (typeof maybeResponse?.then === 'function') {
    try {
      return await maybeResponse;
    } catch (_error) {
      return null;
    }
  }

  return maybeResponse;
}

async function extractGeminiRawText(result) {
  try {
    const response = await resolveGeminiResponse(result);
    if (!response) {
      return '';
    }

    const segments = [];
    const candidates = Array.isArray(response?.candidates) ? response.candidates : [];
    for (const candidate of candidates) {
      const parts = Array.isArray(candidate?.content?.parts) ? candidate.content.parts : [];
      for (const part of parts) {
        if (typeof part?.text === 'string' && part.text) {
          segments.push(part.text);
        }
      }
    }

    const merged = segments.join('').trim();
    if (merged) {
      return merged;
    }

    const legacyParts = response?.response?.candidates?.[0]?.content?.parts || [];
    if (Array.isArray(legacyParts) && legacyParts.length) {
      let fullText = '';
      for (const part of legacyParts) {
        if (typeof part?.text === 'string') {
          fullText += part.text;
        }
      }
      if (fullText.trim()) {
        return fullText.trim();
      }
    }

    try {
      const text = typeof response.text === 'function' ? response.text() : '';
      if (typeof text === 'string') {
        return text.trim();
      }
    } catch (_error) {
    }

    return '';
  } catch (_error) {
    return '';
  }
}

function inferRangeForFeature(name) {
  const normalized = String(name || '').toLowerCase();
  if (normalized.includes('interval') || normalized.includes('rr') || normalized.includes('qt') || normalized.includes('pq') || normalized.includes('st')) {
    return [0, 400];
  }
  if (normalized.includes('peak') || normalized.includes('morph')) {
    return [-5, 5];
  }
  return [0, 300];
}

function getFeatureRange(name) {
  const knownRanges = {
    age: [0, 120],
    sex: [0, 1],
    cp: [0, 3],
    trestbps: [50, 250],
    chol: [50, 600],
    fbs: [0, 1],
    restecg: [0, 2],
    thalach: [50, 250],
    exang: [0, 1],
    oldpeak: [0, 10],
    slope: [0, 2],
    ca: [0, 3],
    thal: [0, 3],
  };

  if (Object.prototype.hasOwnProperty.call(knownRanges, name)) {
    return knownRanges[name];
  }

  return inferRangeForFeature(name);
}

// Attempts to recover usable numbers from OCR noise (units, commas, etc.).
function parseNumericLikeValue(rawValue) {
  if (rawValue === null || rawValue === undefined) {
    return null;
  }

  if (typeof rawValue === 'number') {
    return Number.isFinite(rawValue) ? rawValue : null;
  }

  if (typeof rawValue === 'boolean') {
    return rawValue ? 1 : 0;
  }

  if (Array.isArray(rawValue)) {
    for (const candidate of rawValue) {
      const parsed = parseNumericLikeValue(candidate);
      if (parsed !== null) {
        return parsed;
      }
    }
    return null;
  }

  if (typeof rawValue === 'string') {
    const trimmed = rawValue.trim();
    if (!trimmed) {
      return null;
    }

    const lowered = trimmed.toLowerCase();
    if (['null', 'none', 'nan', 'n/a', 'na', 'unknown'].includes(lowered)) {
      return null;
    }

    const normalized = lowered.replace(/,/g, '');
    const fractionMatch = normalized.match(/(-?\d+(?:\.\d+)?)\s*\/\s*(-?\d+(?:\.\d+)?)/);
    if (fractionMatch) {
      const numerator = Number(fractionMatch[1]);
      return Number.isFinite(numerator) ? numerator : null;
    }

    const match = normalized.match(/-?\d+(?:\.\d+)?/);
    if (match) {
      const numeric = Number(match[0]);
      return Number.isFinite(numeric) ? numeric : null;
    }

    const fallback = Number(normalized);
    return Number.isFinite(fallback) ? fallback : null;
  }

  return null;
}

function normalizeFeatureKey(key) {
  return String(key || '')
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');
}

function lookupValueFromObject(source, key) {
  if (!source || typeof source !== 'object') {
    return undefined;
  }

  if (Object.prototype.hasOwnProperty.call(source, key)) {
    return source[key];
  }

  const targetSlug = normalizeFeatureKey(key);
  for (const [candidateKey, value] of Object.entries(source)) {
    if (normalizeFeatureKey(candidateKey) === targetSlug) {
      return value;
    }
  }

  return undefined;
}

function getFeatureValue(parsed, key) {
  if (parsed === null || parsed === undefined) {
    return null;
  }

  if (Array.isArray(parsed)) {
    for (const item of parsed) {
      const match = getFeatureValue(item, key);
      if (match !== null && match !== undefined) {
        return match;
      }
    }
    return null;
  }

  if (typeof parsed !== 'object') {
    return null;
  }

  const direct = lookupValueFromObject(parsed, key);
  if (direct !== undefined) {
    return direct;
  }

  const nestedKeys = ['features', 'data', 'values', 'payload', 'measurements'];
  for (const nestedKey of nestedKeys) {
    const nested = parsed[nestedKey];
    if (nested && typeof nested === 'object') {
      const match = getFeatureValue(nested, key);
      if (match !== null && match !== undefined) {
        return match;
      }
    }
  }

  return null;
}

function sanitizeExtractedFeatures(parsed) {
  const clean = {};
  for (const key of FEATURE_FIELDS) {
    const rawValue = getFeatureValue(parsed, key);
    const numericValue = parseNumericLikeValue(rawValue);
    if (numericValue === null) {
      clean[key] = null;
      continue;
    }

    const [min, max] = getFeatureRange(key);
    const tolerance = Math.max((max - min) * 0.25, 5);
    if (numericValue < (min - tolerance) || numericValue > (max + tolerance)) {
      clean[key] = null;
      continue;
    }

    clean[key] = Number(numericValue.toFixed(3));
  }
  return clean;
}

function seededRandom(baseSeed, label) {
  const hash = crypto.createHash('sha256').update(`${baseSeed}-${label}`).digest();
  return hash.readUInt32BE(0) / 0xffffffff;
}

function randomFloat(baseSeed, label, min, max, decimals = 2) {
  const random = seededRandom(baseSeed, label);
  const value = min + random * (max - min);
  return Number(value.toFixed(decimals));
}

function buildMockFeatures(filePath) {
  let stats;
  try {
    stats = fs.statSync(filePath);
  } catch (_error) {
    stats = { size: Date.now(), mtimeMs: Date.now() };
  }

  const baseSeed = `${filePath}-${stats.size}-${stats.mtimeMs}`;
  const mock = {};
  for (const key of FEATURE_FIELDS) {
    const [min, max] = getFeatureRange(key);
    mock[key] = randomFloat(baseSeed, key, min, max, 2);
  }
  return mock;
}

async function callGeminiWithRetries(callFn, retries = GEMINI_RETRIES, baseDelayMs = GEMINI_RETRY_BASE_MS) {
  let lastError;
  for (let attempt = 0; attempt < retries; attempt += 1) {
    try {
      return await callFn();
    } catch (error) {
      lastError = error;
      if (error?.status === 404) {
        throw error;
      }
      const delayMs = baseDelayMs * (2 ** attempt);
      await sleep(delayMs);
    }
  }
  throw lastError;
}

function buildFeatureTemplate() {
  const template = {};
  for (const key of FEATURE_FIELDS) {
    template[key] = null;
  }
  return JSON.stringify(template, null, 2);
}

function buildFeatureSpecification() {
  const list = FEATURE_FIELDS.map((name) => `- "${name}"`).join('\n');
  return `Required JSON keys (include every key even if value is null):\n${list}`;
}

function getFallbackClassSummary(readableLabel) {
  if (!readableLabel) {
    return 'No additional context available for this class.';
  }

  const normalized = String(readableLabel).toUpperCase();
  return CLASS_SUMMARIES[normalized] || `Limited context is available for ${readableLabel}. Consult the ECG trace for confirmation.`;
}

function buildProbabilitySummary(probabilities) {
  if (!probabilities || typeof probabilities !== 'object') {
    return '';
  }

  const ranked = Object.entries(probabilities)
    .filter((entry) => typeof entry[1] === 'number')
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([label, prob]) => `${label}: ${(prob * 100).toFixed(1)}%`);

  return ranked.length ? `Top probabilities: ${ranked.join(', ')}.` : '';
}

function buildPredictionReportText(reportContext) {
  const prediction = reportContext?.prediction || {};
  const features = reportContext?.features && typeof reportContext.features === 'object'
    ? reportContext.features
    : {};
  const probabilities = prediction.probabilities && typeof prediction.probabilities === 'object'
    ? prediction.probabilities
    : null;

  const confidence = probabilities
    ? Object.values(probabilities)
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value))
    : [];
  const topConfidence = confidence.length ? `${(Math.max(...confidence) * 100).toFixed(2)}%` : 'N/A';

  const probabilityLines = probabilities
    ? Object.entries(probabilities)
      .map(([label, value]) => {
        const numeric = Number(value);
        const asPercent = Number.isFinite(numeric) ? `${(numeric * 100).toFixed(2)}%` : 'N/A';
        return `- ${label}: ${asPercent}`;
      })
      .join('\n')
    : '- N/A';

  const featureLines = Object.entries(features)
    .map(([key, value]) => `- ${key}: ${value}`)
    .join('\n');

  const generatedAt = reportContext?.generatedAt
    ? new Date(reportContext.generatedAt).toLocaleString()
    : new Date().toLocaleString();

  const lines = [
    'AI ARRHYTHMIA PREDICTION REPORT',
    '===============================',
    '',
    `Generated At: ${generatedAt}`,
    `Patient/User: ${reportContext?.user?.fullName || 'Unknown'} (${reportContext?.user?.username || 'N/A'})`,
    `Role: ${reportContext?.user?.role || 'N/A'}`,
    `Model: ${reportContext?.modelName || MODEL_DISPLAY_NAME}`,
    '',
    `Predicted Class: ${prediction.readableLabel || prediction.label || 'Unknown'}`,
    `Label ID: ${prediction.label_id ?? 'N/A'}`,
    `Top Confidence: ${topConfidence}`,
    '',
    'Clinical Insight:',
    reportContext?.predictionDescription || 'No additional context available.',
    '',
    'Probability Breakdown:',
    probabilityLines,
    '',
    'Submitted Features:',
    featureLines || '- N/A',
  ];

  return `${lines.join('\n')}\n`;
}

function buildReportContextFromHistory(sessionUser, historyEntry) {
  if (!sessionUser || !historyEntry) {
    return null;
  }

  return {
    generatedAt: historyEntry.createdAt ? new Date(historyEntry.createdAt).toISOString() : new Date().toISOString(),
    modelName: MODEL_DISPLAY_NAME,
    user: {
      fullName: sessionUser.fullName,
      username: sessionUser.username,
      role: sessionUser.role,
    },
    features: historyEntry.features || {},
    prediction: {
      label_id: Number.isFinite(Number(historyEntry.labelId)) ? Number(historyEntry.labelId) : -1,
      label: historyEntry.predictionLabel || 'Unknown',
      readableLabel: historyEntry.predictionLabel || 'Unknown',
      probabilities: historyEntry.probabilities || null,
    },
    predictionDescription: historyEntry.description || null,
  };
}

async function downloadPredictionReport(req, res) {
  let reportContext = req.session.latestPredictionReport;

  if (!reportContext) {
    const latestEntry = await PredictionHistory.findOne({ username: req.session.user.username })
      .sort({ createdAt: -1 })
      .lean();
    reportContext = buildReportContextFromHistory(req.session.user, latestEntry);
  }

  if (!reportContext) {
    return res.status(400).send('No prediction report available. Run a prediction first.');
  }

  const reportText = buildPredictionReportText(reportContext);
  const now = new Date();
  const stamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}-${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`;

  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Content-Disposition', `attachment; filename="ecg-prediction-report-${stamp}.txt"`);
  return res.send(reportText);
}

async function describePredictionWithLLM(prediction) {
  if (!prediction) {
    return 'Prediction details are unavailable.';
  }

  const fallback = getFallbackClassSummary(prediction.readableLabel || prediction.label);
  const cacheKey = `${prediction.label_id}-${prediction.readableLabel || prediction.label}`;

  if (classDescriptionCache.has(cacheKey)) {
    return classDescriptionCache.get(cacheKey);
  }

  if (!process.env.GOOGLE_API_KEY || !genAI || TEXT_MODEL_CANDIDATES.length === 0) {
    return fallback;
  }

  let lastError = null;
  let hadNotFoundError = false;
  try {
    const probabilitySummary = buildProbabilitySummary(prediction.probabilities);
    const prompt = [
      'You are acting as a cardiac electrophysiology expert explaining ECG beat classifications.',
      'Write a concise, two-sentence clinical description of the predicted class below.',
      'Avoid mentioning that you are an AI, and do not restate instructions.',
      `Predicted label: ${prediction.readableLabel || prediction.label}`,
      `Label id: ${prediction.label_id}`,
      probabilitySummary,
      'Tone: confident, clinical, and easy to skim for a busy cardiologist.',
    ].filter(Boolean).join('\n');

    for (const modelName of TEXT_MODEL_CANDIDATES) {
      try {
        const textModel = genAI.getGenerativeModel({ model: modelName });
        const response = await callGeminiWithRetries(() => textModel.generateContent({
          contents: [{
            role: 'user',
            parts: [{ text: prompt }],
          }],
          generationConfig: {
            temperature: 0.4,
            topP: 0.8,
            topK: 32,
            maxOutputTokens: 256,
          },
        }));

        const rawText = await extractGeminiRawText(response);
        const cleaned = (rawText || '').replace(/\s+/g, ' ').trim();
        if (!cleaned) {
          throw new Error('Empty description');
        }

        const finalText = cleaned.length > 360 ? `${cleaned.slice(0, 357)}...` : cleaned;
        classDescriptionCache.set(cacheKey, finalText);
        return finalText;
      } catch (modelError) {
        lastError = modelError;
        if (modelError?.status === 404) {
          hadNotFoundError = true;
          continue;
        }
      }
    }

    if (hadNotFoundError && (!lastError || lastError?.status === 404)) {
      classDescriptionCache.set(cacheKey, fallback);
      return fallback;
    }

    if (lastError) {
      throw lastError;
    }
  } catch (error) {
    if (error?.status !== 404) {
      console.error('Prediction description failed:', error.message);
    }
    const fallbackText = fallback;
    classDescriptionCache.set(cacheKey, fallbackText);
    return fallbackText;
  }
}

async function recordPredictionHistory(sessionUser, features, prediction, predictionDescription) {
  if (!sessionUser || !sessionUser.username || !prediction) {
    return;
  }

  try {
    await PredictionHistory.create({
      username: sessionUser.username,
      userDisplayName: sessionUser.fullName || sessionUser.username,
      role: sessionUser.role || 'doctor',
      features,
      predictionLabel: prediction.readableLabel || prediction.label || 'Unknown',
      labelId: Number.isFinite(Number(prediction.label_id)) ? Number(prediction.label_id) : -1,
      probabilities: prediction.probabilities || null,
      description: predictionDescription || null,
    });
  } catch (error) {
    console.error('Failed to record prediction history:', error.message);
  }
}

async function extractFeaturesWithGoogleLLM(file) {
  if (!process.env.GOOGLE_API_KEY || !genAI) {
    if (ENABLE_MOCK_VISION) {
      return buildMockFeatures(file.path);
    }
    throw new Error('GOOGLE_API_KEY is missing on server');
  }

  const imageBase64 = fs.readFileSync(file.path).toString('base64');
  const mimeType = file.mimetype || 'image/png';
  const prompt = [
  'You are a STRICT medical OCR extraction engine.',
  'TASK:',
  'Extract ONLY numeric values that are CLEARLY visible in the medical report image.',
  'CRITICAL RULES:',
  'DO NOT estimate, infer, guess, or assume any values.',
  'If a value is not explicitly written, return null.',
  'If unsure, return null.',
  'Never generate medical values from knowledge.',
  'Output must be STRICT JSON only.',
  'Return STRICT JSON only.',
  'No markdown. No code fences. No leading text.',
  'Response must start with "{" and end with "}".',
  buildFeatureSpecification(),
  'JSON template (fill values or null, keep the same order):',
  buildFeatureTemplate(),
  'Example output (values are illustrative only):',
  '{\n  "0_pre-RR": 0,\n  "0_post-RR": null,\n  "...": null\n}',
].join('\n\n');


  let response = null;
  let lastError = null;

  for (const modelName of MODEL_CANDIDATES) {
    try {
      response = await callGeminiWithRetries(async () => {
        const visionModel = genAI.getGenerativeModel({ model: modelName });
        return visionModel.generateContent({
          contents: [{
            role: 'user',
            parts: [
              { text: prompt },
              {
                inlineData: {
                  data: imageBase64,
                  mimeType,
                },
              },
            ],
          }],
          generationConfig: {
            temperature: 0,
            topP: 0.1,
            topK: 1,
            maxOutputTokens: 2048,
            responseMimeType: 'application/json',
          },
        });
      });
      lastError = null;
      break;
    } catch (error) {
      lastError = error;
      if (error?.status === 404) {
        continue;
      }
    }
  }

  if (!response) {
    if (ENABLE_MOCK_VISION) {
      return buildMockFeatures(file.path);
    }
    throw new Error(lastError?.message || 'Gemini model failed for all candidates');
  }

  const rawText = await extractGeminiRawText(response);
  console.log('[Gemini OCR] raw response text:', rawText || '<empty>');
  let parsed = tolerantParseJson(rawText);
  console.log('[Gemini OCR] parsed response object:', parsed);
  if (!parsed) {
    parsed = recoverLooseKeyValuePairs(rawText);
    console.log('[Gemini OCR] recovered via loose parser:', parsed);
  }

  if (!parsed) {
    if (ENABLE_MOCK_VISION) {
      return buildMockFeatures(file.path);
    }
    throw new Error('Gemini returned invalid JSON');
  }

  const sanitized = sanitizeExtractedFeatures(parsed);
  console.log('[Gemini OCR] sanitized features:', sanitized);
  return sanitized;
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
