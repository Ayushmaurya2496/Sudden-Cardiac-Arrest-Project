from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV_PATH = ROOT / "Sudden Cardiac Death Holter Database.csv"
MODEL_PATH = HERE / "ecg_xgboost_model.pkl"
META_PATH = HERE / "model_meta.json"
REPORT_PATH = HERE / "training_report.txt"

FEATURE_DROP = ["record"]
TARGET = "type"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ECG arrhythmia classifier")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap for quick experiments (use all rows by default)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_dataset(limit: int | None) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    df = pd.read_csv(CSV_PATH)
    if limit:
        df = df.sample(n=limit, random_state=42)
    df = df.drop(columns=FEATURE_DROP)
    df = df.dropna()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET])
    X = df.drop(columns=[TARGET])
    return X, y, encoder


def build_model(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )


def main() -> None:
    args = parse_args()
    X, y, encoder = load_dataset(args.limit)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = build_model(args.random_state)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=list(encoder.classes_))

    print("Training complete. Classification report:\n")
    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "feature_names": list(X.columns),
        "label_map": {int(idx): label for idx, label in enumerate(encoder.classes_)},
        "sample_count": int(len(X)),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    REPORT_PATH.write_text(report)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metadata to {META_PATH}")
    print(f"Saved training report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
