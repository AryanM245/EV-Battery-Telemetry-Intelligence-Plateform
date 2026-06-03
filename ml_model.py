"""
ml_model.py
===========
ML pipeline trained on the real EVIoT Predictive Maintenance Dataset.

Models:
  1. SoH Regressor          — Random Forest  → predicts State-of-Health (%)
  2. RUL Regressor          — Gradient Boost → predicts Remaining Useful Life (days)
  3. Failure Classifier     — Gradient Boost → binary failure probability prediction
  4. Thermal Risk Classifier— Gradient Boost → LOW/MEDIUM/HIGH/CRITICAL

Outputs saved to models/
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    GradientBoostingClassifier
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    classification_report, accuracy_score
)

from db_manager import DatabaseManager

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42

# ── Feature sets ────────────────────────────────────────────────────────────────

SOH_FEATURES = [
    "Charge_Cycles", "Battery_Temperature", "Battery_Current",
    "Battery_Voltage", "SoC", "Motor_Temperature",
    "Ambient_Temperature", "Power_Consumption", "Distance_Traveled",
]

RUL_FEATURES = [
    "SoH", "SoC", "Charge_Cycles", "Component_Health_Score",
    "Battery_Temperature", "Motor_Temperature", "Motor_Vibration",
    "Brake_Pad_Wear", "Route_Roughness", "Load_Weight",
]

FAILURE_FEATURES = [
    "SoH", "SoC", "Battery_Temperature", "Motor_Temperature",
    "Motor_Vibration", "Brake_Pad_Wear", "Component_Health_Score",
    "RUL", "TTF", "Charge_Cycles",
]

THERMAL_FEATURES = [
    "Battery_Temperature", "Motor_Temperature", "Ambient_Temperature",
    "Battery_Current", "SoC", "Power_Consumption",
]

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ── Data Loading ─────────────────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    print("[*] Loading training data from database...")
    with DatabaseManager() as db:
        df = db.get_ml_training_data()
    print(f"   -> Loaded {len(df):,} rows x {df.shape[1]} columns.")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df[(df["SoH"] > 10) & (df["SoH"] <= 100)]
    return df.reset_index(drop=True)


# ── Model 1: SoH Regressor ───────────────────────────────────────────────────────

def train_soh_regressor(df: pd.DataFrame) -> dict:
    print("\n[ML] Training SoH Regressor (Random Forest)...")
    X = df[SOH_FEATURES].values
    y = df["SoH"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=120, max_depth=12,
            min_samples_leaf=5, n_jobs=-1, random_state=SEED
        ))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"   -> MAE: {mae:.4f}%  |  R2: {r2:.4f}")

    rf = pipe.named_steps["model"]
    importances = dict(sorted(
        zip(SOH_FEATURES, rf.feature_importances_.tolist()),
        key=lambda x: x[1], reverse=True
    ))
    print("   -> Top features:", list(importances.keys())[:4])

    path = os.path.join(MODELS_DIR, "soh_regressor.pkl")
    joblib.dump(pipe, path)
    print(f"   [OK] Saved -> {path}")
    return {"model": pipe, "mae": mae, "r2": r2, "importances": importances}


# ── Model 2: RUL Regressor ───────────────────────────────────────────────────────

def train_rul_regressor(df: pd.DataFrame) -> dict:
    print("\n[ML] Training RUL Regressor (Gradient Boosting)...")
    available = [f for f in RUL_FEATURES if f in df.columns]
    X = df[available].values
    y = df["RUL"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=SEED
        ))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"   -> MAE: {mae:.2f} days  |  R2: {r2:.4f}")

    path = os.path.join(MODELS_DIR, "rul_regressor.pkl")
    joblib.dump(pipe, path)
    print(f"   [OK] Saved -> {path}")
    return {"model": pipe, "mae": mae, "r2": r2, "features": available}


# ── Model 3: Failure Classifier ──────────────────────────────────────────────────

def train_failure_classifier(df: pd.DataFrame) -> dict:
    print("\n[ML] Training Failure Classifier (Gradient Boosting)...")
    available = [f for f in FAILURE_FEATURES if f in df.columns]
    X = df[available].values
    y = df["Failure_Probability"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingClassifier(
            n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=SEED
        ))
    ])
    pipe.fit(X_train, y_train)

    y_pred   = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   -> Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=["No Failure", "Failure"], zero_division=0))

    path = os.path.join(MODELS_DIR, "failure_classifier.pkl")
    joblib.dump(pipe, path)
    print(f"   [OK] Saved -> {path}")
    return {"model": pipe, "accuracy": accuracy, "features": available}


# ── Model 4: Thermal Risk Classifier ────────────────────────────────────────────

def train_thermal_classifier(df: pd.DataFrame) -> dict:
    print("\n[ML] Training Thermal Risk Classifier (Gradient Boosting)...")
    risk_map = {r: i for i, r in enumerate(RISK_ORDER)}
    df = df.copy()
    df["risk_encoded"] = df["thermal_risk"].map(risk_map)
    df = df.dropna(subset=["risk_encoded"])

    X = df[THERMAL_FEATURES].values
    y = df["risk_encoded"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingClassifier(
            n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=SEED
        ))
    ])
    pipe.fit(X_train, y_train)

    y_pred   = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   -> Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=RISK_ORDER, zero_division=0))

    path = os.path.join(MODELS_DIR, "thermal_classifier.pkl")
    joblib.dump(pipe, path)
    print(f"   [OK] Saved -> {path}")
    return {"model": pipe, "accuracy": accuracy, "labels": RISK_ORDER}


# ── Metadata ─────────────────────────────────────────────────────────────────────

def save_metadata(soh, rul, failure, thermal):
    metadata = {
        "soh_regressor":       {"features": SOH_FEATURES,     "mae": round(soh["mae"],4),     "r2": round(soh["r2"],4),         "importances": soh["importances"]},
        "rul_regressor":       {"features": rul["features"],   "mae": round(rul["mae"],4),     "r2": round(rul["r2"],4)},
        "failure_classifier":  {"features": failure["features"],"accuracy": round(failure["accuracy"],4)},
        "thermal_classifier":  {"features": THERMAL_FEATURES,  "accuracy": round(thermal["accuracy"],4), "labels": RISK_ORDER},
    }
    path = os.path.join(MODELS_DIR, "feature_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[OK] Metadata saved -> {path}")


# ── Prediction helpers (used by dashboard) ───────────────────────────────────────

def load_soh_model():      return joblib.load(os.path.join(MODELS_DIR, "soh_regressor.pkl"))
def load_rul_model():      return joblib.load(os.path.join(MODELS_DIR, "rul_regressor.pkl"))
def load_failure_model():  return joblib.load(os.path.join(MODELS_DIR, "failure_classifier.pkl"))
def load_thermal_model():  return joblib.load(os.path.join(MODELS_DIR, "thermal_classifier.pkl"))


def predict_soh(model, inputs: dict) -> float:
    X = np.array([[inputs[f] for f in SOH_FEATURES]])
    return float(np.clip(model.predict(X)[0], 0, 100))

def predict_rul(model, inputs: dict, features: list) -> float:
    X = np.array([[inputs.get(f, 0) for f in features]])
    return float(max(0, model.predict(X)[0]))

def predict_failure_proba(model, inputs: dict, features: list) -> float:
    X = np.array([[inputs.get(f, 0) for f in features]])
    return float(model.predict_proba(X)[0][1])

def predict_thermal_risk(model, inputs: dict) -> str:
    X = np.array([[inputs[f] for f in THERMAL_FEATURES]])
    return RISK_ORDER[int(model.predict(X)[0])]

def predict_thermal_proba(model, inputs: dict) -> dict:
    X = np.array([[inputs[f] for f in THERMAL_FEATURES]])
    p = model.predict_proba(X)[0]
    return {label: round(float(v), 4) for label, v in zip(RISK_ORDER, p)}


# ── Entry Point ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EVIoT Dataset - ML Training Pipeline")
    print("=" * 60)

    df = load_training_data()
    df = preprocess(df)
    print(f"\n[*] Clean dataset: {len(df):,} rows")

    soh     = train_soh_regressor(df)
    rul     = train_rul_regressor(df)
    failure = train_failure_classifier(df)
    thermal = train_thermal_classifier(df)
    save_metadata(soh, rul, failure, thermal)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  SoH       R2={soh['r2']:.4f}  MAE={soh['mae']:.4f}%")
    print(f"  RUL       R2={rul['r2']:.4f}  MAE={rul['mae']:.2f} days")
    print(f"  Failure   Acc={failure['accuracy']:.4f}")
    print(f"  Thermal   Acc={thermal['accuracy']:.4f}")
    print("=" * 60)
