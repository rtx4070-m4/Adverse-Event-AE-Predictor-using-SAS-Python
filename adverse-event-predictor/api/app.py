"""
Adverse Event Risk Predictor — MIMIC-III
api/app.py

FastAPI REST API for ICU adverse event risk prediction.

Endpoints:
  GET  /health           — health check
  GET  /model-info       — model metadata + performance
  POST /predict          — single patient prediction
  POST /predict/batch    — batch prediction (≤100 patients)
  POST /explain          — SHAP / feature-contribution explanation

Dependencies: fastapi, uvicorn, joblib, numpy, pandas, sklearn.
Install: pip install fastapi uvicorn pydantic
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# ── FastAPI imports (with helpful error if missing) ───────────────────────────
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

sys.path.insert(0, str(PROJECT_ROOT / "python"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ae.api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Adverse Event Risk Predictor",
    description=(
        "ICU patient adverse event risk prediction API.\n\n"
        "Trained on MIMIC-III data using Logistic Regression, Random Forest, "
        "Gradient Boosting, Extra Trees (+ XGBoost/LightGBM when available). "
        "Best model selected by 5-fold cross-validated ROC-AUC.\n\n"
        "**Disclaimer:** For research/decision-support only. Not for clinical use."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
_model:           Any = None
_metadata:        Dict = {}
_startup_time:    datetime = datetime.now(timezone.utc)
_prediction_count: int = 0

FEATURE_COLS = [
    "age", "gender", "length_of_stay", "heart_rate_mean",
    "creatinine_max", "wbc_mean", "drug_count",
    "polypharmacy_score", "lab_abnormality_score",
]


def _load_model_artifacts():
    """Load model pipeline and metadata on startup. Trains if not found."""
    global _model, _metadata

    model_path = MODELS_DIR / "ae_model.pkl"
    meta_path  = MODELS_DIR / "model_metadata.json"

    if not model_path.exists():
        logger.warning("Model not found — training on synthetic data (one-time setup)...")
        from train_model import main as train_main
        train_main(data_source="synthetic", cv_folds=3, n_patients=3000)
        from evaluate_model import run_full_evaluation
        run_full_evaluation(data_source="synthetic")

    _model = joblib.load(model_path)
    logger.info(f"Model loaded: {model_path.name}  ({model_path.stat().st_size//1024} KB)")

    if meta_path.exists():
        with open(meta_path) as f:
            _metadata = json.load(f)
        logger.info(f"Model: {_metadata.get('best_model_name')} | "
                    f"AUC={_metadata.get('test_metrics', {}).get('roc_auc', 0):.4f}")


@app.on_event("startup")
async def startup_event():
    _load_model_artifacts()
    logger.info("API ready.")


# ── Pydantic models ───────────────────────────────────────────────────────────
class PatientFeatures(BaseModel):
    """Input features for a single ICU patient."""
    age:                   float = Field(..., ge=18, le=110,   description="Age in years")
    gender:                float = Field(..., ge=0,  le=1,     description="1=Male, 0=Female")
    length_of_stay:        float = Field(..., gt=0,  le=365,   description="ICU LOS (days)")
    heart_rate_mean:       float = Field(..., ge=20, le=300,   description="Mean HR (bpm)")
    creatinine_max:        float = Field(..., ge=0,  le=30,    description="Max creatinine (mg/dL)")
    wbc_mean:              float = Field(..., ge=0,  le=200,   description="Mean WBC (K/µL)")
    drug_count:            float = Field(..., ge=0,  le=200,   description="Distinct drugs")
    polypharmacy_score:    float = Field(..., ge=0,  le=10,    description="Polypharmacy score")
    lab_abnormality_score: float = Field(..., ge=0,            description="Critical lab count")
    patient_id:            Optional[str] = Field(None, description="Optional patient ID")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 72, "gender": 1, "length_of_stay": 5.3,
                "heart_rate_mean": 98.5, "creatinine_max": 2.1,
                "wbc_mean": 14.2, "drug_count": 12,
                "polypharmacy_score": 6.0, "lab_abnormality_score": 3,
            }
        }
    }


class PredictionResponse(BaseModel):
    patient_id:         Optional[str]
    risk_score:         float
    risk_category:      str
    adverse_event_flag: int
    threshold_used:     float
    clinical_message:   str
    input_features:     Dict[str, Any]
    model_version:      str
    timestamp:          str


class BatchRequest(BaseModel):
    patients: List[PatientFeatures]


class BatchResponse(BaseModel):
    predictions:        List[PredictionResponse]
    n_patients:         int
    n_high_risk:        int
    processing_time_ms: float


class ExplainRequest(BaseModel):
    patient: PatientFeatures
    top_n:   int = Field(8, ge=1, le=15)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _patient_to_df(patient: PatientFeatures) -> pd.DataFrame:
    """Convert PatientFeatures → engineered feature DataFrame."""
    from feature_engineering import engineer_features
    raw = {col: [getattr(patient, col)] for col in FEATURE_COLS}
    df  = pd.DataFrame(raw)
    X, _ = engineer_features(df, feature_cols=FEATURE_COLS, add_interactions=True)
    return X


def _risk_category(score: float) -> str:
    if score < 0.20:   return "Low"
    elif score < 0.40: return "Moderate"
    elif score < 0.65: return "High"
    else:              return "Critical"


def _clinical_message(category: str) -> str:
    return {
        "Low":      "Low adverse event risk. Continue standard ICU monitoring protocols.",
        "Moderate": "Moderate risk. Increase monitoring frequency; review medication list.",
        "High":     "High risk. Consider clinical review, escalation, and family notification.",
        "Critical": "CRITICAL RISK. Immediate intensivist review strongly recommended.",
    }[category]


def _make_prediction(patient: PatientFeatures) -> PredictionResponse:
    global _prediction_count
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    _prediction_count += 1

    threshold  = _metadata.get("optimal_threshold", 0.5)
    version    = _metadata.get("version", "v1")
    model_name = _metadata.get("best_model_name", "unknown")

    X          = _patient_to_df(patient)
    risk_score = float(_model.predict_proba(X)[0, 1])
    category   = _risk_category(risk_score)

    return PredictionResponse(
        patient_id         = patient.patient_id,
        risk_score         = round(risk_score, 4),
        risk_category      = category,
        adverse_event_flag = int(risk_score >= threshold),
        threshold_used     = threshold,
        clinical_message   = _clinical_message(category),
        input_features     = {col: getattr(patient, col) for col in FEATURE_COLS},
        model_version      = f"{version}:{model_name}",
        timestamp          = datetime.now(timezone.utc).isoformat(),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """API health check."""
    return {
        "status":             "healthy",
        "model_loaded":       _model is not None,
        "uptime_seconds":     (datetime.now(timezone.utc) - _startup_time).total_seconds(),
        "predictions_served": _prediction_count,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Model metadata, performance metrics, and feature list."""
    if not _metadata:
        raise HTTPException(503, "Model metadata not available")
    return {
        "model_name":        _metadata.get("best_model_name"),
        "version":           _metadata.get("version"),
        "description":       _metadata.get("description"),
        "feature_names":     _metadata.get("feature_names"),
        "optimal_threshold": _metadata.get("optimal_threshold"),
        "performance": {
            "cv_roc_auc":     _metadata.get("cv_metrics", {}).get("cv_roc_auc_mean"),
            "cv_roc_auc_std": _metadata.get("cv_metrics", {}).get("cv_roc_auc_std"),
            "test_roc_auc":   _metadata.get("test_metrics", {}).get("roc_auc"),
            "avg_precision":  _metadata.get("test_metrics", {}).get("avg_precision"),
        },
        "risk_categories": {
            "Low":      "risk_score < 0.20",
            "Moderate": "0.20 ≤ risk_score < 0.40",
            "High":     "0.40 ≤ risk_score < 0.65",
            "Critical": "risk_score ≥ 0.65",
        },
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientFeatures):
    """
    Predict adverse event risk for a single ICU patient.
    Returns risk score (0–1), risk category, and clinical recommendation.
    """
    try:
        return _make_prediction(patient)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {e}")


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(batch: BatchRequest):
    """
    Predict adverse event risk for a cohort of ICU patients (max 100).
    """
    if len(batch.patients) == 0:
        raise HTTPException(400, "No patients provided")
    if len(batch.patients) > 100:
        raise HTTPException(400, "Maximum 100 patients per batch request")

    t0 = time.time()
    preds, errors = [], []

    for i, patient in enumerate(batch.patients):
        try:
            preds.append(_make_prediction(patient))
        except Exception as e:
            errors.append({"patient_idx": i, "error": str(e)})
            logger.warning(f"Patient {i} failed: {e}")

    elapsed = (time.time() - t0) * 1000
    n_high  = sum(1 for p in preds if p.adverse_event_flag == 1)

    return BatchResponse(
        predictions        = preds,
        n_patients         = len(preds),
        n_high_risk        = n_high,
        processing_time_ms = round(elapsed, 2),
    )


@app.post("/explain", tags=["Explainability"])
async def explain_prediction(req: ExplainRequest):
    """
    Generate feature-contribution explanation for a patient's risk score.
    Uses SHAP (if installed) or permutation importance proxy.
    """
    try:
        from evaluate_model import explain_patient
        from feature_engineering import engineer_features

        raw = {col: [getattr(req.patient, col)] for col in FEATURE_COLS}
        df  = pd.DataFrame(raw)
        X, feat_names = engineer_features(df, feature_cols=FEATURE_COLS, add_interactions=True)

        pred = _make_prediction(req.patient)

        # For single-patient explanation we need a small background
        explanation = explain_patient(_model, X, feat_names, top_n=req.top_n)

        # Derive interpretation sentence
        top = explanation["contributions"][0] if explanation["contributions"] else None
        interp = (
            f"This patient's predicted risk of {pred.risk_score:.1%} is primarily driven by "
            f"'{top['feature']}' which {top['direction']}."
            if top else "No dominant risk factor identified."
        )

        return {
            "prediction":  pred,
            "explanation": {
                "base_value":     explanation["base_value"],
                "contributions":  explanation["contributions"][:req.top_n],
                "interpretation": interp,
                "method":         explanation["contributions"][0].get("method", "unknown")
                                  if explanation["contributions"] else "unknown",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(500, f"Explanation failed: {e}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("Install uvicorn: pip install uvicorn")
