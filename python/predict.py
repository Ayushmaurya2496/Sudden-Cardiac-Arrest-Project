from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "ecg_xgboost_model.pkl"
DEFAULT_META = HERE / "model_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the ECG model")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the trained model")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="Path to the metadata JSON")
    return parser.parse_args()


def load_payload() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw:
        raise ValueError("Missing JSON payload on stdin")
    return json.loads(raw)


def build_frame(feature_names: list[str], feature_values: Dict[str, Any]) -> pd.DataFrame:
    row = []
    for name in feature_names:
        value = feature_values.get(name, 0)
        try:
            row.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{name}' must be numeric. Received: {value}") from exc
    return pd.DataFrame([row], columns=feature_names)


def main() -> None:
    try:
        args = parse_args()
        payload = load_payload()
        meta = json.loads(args.meta.read_text())
        feature_names = meta["feature_names"]
        df = build_frame(feature_names, payload.get("features", {}))

        model = joblib.load(args.model)
        preds = model.predict(df)
        try:
            proba = model.predict_proba(df)[0].tolist()
        except Exception:
            proba = None

        label_index = int(preds[0])
        label_map = {int(k): v for k, v in meta["label_map"].items()}
        label = label_map.get(label_index, str(label_index))

        response: Dict[str, Any] = {
            "label_id": label_index,
            "label": label,
        }
        if proba is not None:
            response["probabilities"] = {
                label_map.get(idx, str(idx)): score for idx, score in enumerate(proba)
            }

        print(json.dumps(response))
    except Exception as exc:  # pragma: no cover - piped to caller
        error = {"error": str(exc)}
        print(json.dumps(error))
        sys.exit(1)


if __name__ == "__main__":
    main()
