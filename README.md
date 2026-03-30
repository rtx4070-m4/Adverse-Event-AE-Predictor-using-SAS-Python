# ⚕ Adverse Event Risk Predictor — MIMIC-III

> **Clinical AI Pipeline** · ICU patient risk stratification using machine learning on MIMIC-III data

[![CI/CD](https://github.com/your-org/adverse-event-predictor/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-org/adverse-event-predictor/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This system predicts which ICU patients are at high risk of adverse medical events using machine learning trained on the [MIMIC-III](https://physionet.org/content/mimiciii/) clinical database.

**Adverse Event Definition** (composite outcome):
- In-hospital mortality
- ICU readmission within the same hospitalization
- Sepsis / septic shock / severe septicemia (ICD-9)
- ≥3 critical laboratory abnormalities

---

## Architecture

```
MIMIC-III PostgreSQL
      │
      ▼ SQL (data/raw/extract_features.sql)
Feature Extraction Views
      │
      ▼ SAS (sas/data_cleaning.sas + sas/baseline_model.sas)
Statistical Preprocessing + Baseline Logistic Regression
      │
      ▼ Python (python/)
Feature Engineering → ML Training → SHAP Explainability
      │
      ┌─────────────────┬──────────────────┐
      ▼                 ▼                  ▼
FastAPI REST API   Streamlit Dashboard   Model Artifacts
(api/app.py)       (dashboard/)          (models/)
```

---

## Project Structure

```
adverse-event-predictor/
│
├── data/
│   ├── raw/
│   │   └── extract_features.sql     # MIMIC-III SQL extraction
│   └── processed/
│       ├── ae_cleaned.csv           # SAS-cleaned output
│       ├── feature_dictionary.csv
│       └── reports/                 # Generated plots
│
├── sas/
│   ├── data_cleaning.sas            # Clinical data cleaning
│   └── baseline_model.sas          # Statistical baseline
│
├── python/
│   ├── data_loader.py               # Data ingestion + validation
│   ├── feature_engineering.py       # Feature transforms + pipeline
│   ├── train_model.py               # Multi-model training
│   └── evaluate_model.py            # Evaluation + SHAP
│
├── models/
│   ├── ae_model.pkl                 # Best model pipeline
│   ├── model_metadata.json          # Performance metadata
│   └── shap_explainer.pkl           # SHAP explainer
│
├── api/
│   └── app.py                       # FastAPI REST API
│
├── dashboard/
│   └── streamlit_app.py             # Streamlit clinical dashboard
│
├── tests/
│   └── test_pipeline.py             # Full test suite
│
├── .github/workflows/
│   └── ci-cd.yml                    # GitHub Actions CI/CD
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Features

| Feature | Type | Description |
|---|---|---|
| `age` | Continuous | Patient age in years (18–110) |
| `gender` | Binary | Sex: 1=Male, 0=Female |
| `length_of_stay` | Continuous | ICU length of stay (days) |
| `heart_rate_mean` | Continuous | Mean heart rate (bpm) |
| `creatinine_max` | Continuous | Maximum serum creatinine (mg/dL) |
| `wbc_mean` | Continuous | Mean white blood cell count (K/μL) |
| `drug_count` | Count | Total distinct drugs prescribed |
| `polypharmacy_score` | Continuous | Polypharmacy burden (0–10) |
| `lab_abnormality_score` | Count | Critical lab value count |

Plus 17 derived features (interactions, clinical flags, log transforms).

---

## Machine Learning Models

| Model | Description |
|---|---|
| Logistic Regression | L2-regularized, clinical interpretable baseline |
| Random Forest | 300 trees, balanced class weights |
| XGBoost | Gradient boosting, scale_pos_weight for imbalance |
| LightGBM | Fast GBDT, `is_unbalance=True` |

**Selection criterion:** Cross-validated ROC-AUC (5-fold stratified)

---

## Quick Start (Local)

### Prerequisites
- Python 3.10+
- (Optional) PostgreSQL with MIMIC-III loaded
- (Optional) SAS 9.4+

### 1. Install dependencies

```bash
git clone https://github.com/your-org/adverse-event-predictor
cd adverse-event-predictor

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your database credentials if using real MIMIC-III data
```

### 3. Train the model

```bash
# Using synthetic data (no MIMIC-III required):
python python/train_model.py --source synthetic --cv 5

# Using real MIMIC-III data:
# 1. Run data/raw/extract_features.sql in PostgreSQL first
# 2. Run sas/data_cleaning.sas to produce data/processed/ae_cleaned.csv
# 3. Then:
python python/train_model.py --source csv --cv 5
```

### 4. Evaluate and generate plots

```bash
python python/evaluate_model.py
```

### 5. Start the API

```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API docs at: http://localhost:8000/docs

### 6. Launch the dashboard

```bash
# In a new terminal:
streamlit run dashboard/streamlit_app.py
```

Dashboard at: http://localhost:8501

---

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 72,
    "gender": 1,
    "length_of_stay": 5.3,
    "heart_rate_mean": 98.5,
    "creatinine_max": 2.1,
    "wbc_mean": 14.2,
    "drug_count": 12,
    "polypharmacy_score": 6.0,
    "lab_abnormality_score": 3
  }'
```

**Response:**
```json
{
  "risk_score": 0.7432,
  "risk_category": "Critical",
  "adverse_event_flag": 1,
  "threshold_used": 0.42,
  "clinical_message": "CRITICAL RISK. Immediate clinical review strongly recommended.",
  "model_version": "v1:lightgbm",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### `POST /explain`

Returns SHAP feature contributions for interpretability.

### `GET /health`

Health check and API status.

### `GET /model-info`

Model metadata, version, and performance metrics.

---

## Docker Deployment

### Local Docker Compose

```bash
# Build and start all services
docker compose up --build -d

# Services:
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# MLflow:    http://localhost:5000
# Postgres:  localhost:5432
```

### Cloud Deployment

#### AWS ECS

```bash
# Build and push to ECR
aws ecr create-repository --repository-name ae-predictor
docker build -t ae-predictor .
docker tag ae-predictor:latest <account>.dkr.ecr.us-east-1.amazonaws.com/ae-predictor:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/ae-predictor:latest

# Deploy with ECS task definition (see deploy/ecs-task-definition.json)
```

#### GCP Cloud Run

```bash
# Build and deploy
gcloud run deploy ae-predictor \
  --source . \
  --region us-central1 \
  --platform managed \
  --port 8000 \
  --set-env-vars="DATA_SOURCE=csv,MODEL_VERSION=v1" \
  --allow-unauthenticated
```

#### Azure Container Apps

```bash
az containerapp create \
  --name ae-predictor \
  --resource-group myRG \
  --environment myEnv \
  --image ae-predictor:latest \
  --target-port 8000 \
  --ingress external
```

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v --cov=python --cov=api --cov-report=term-missing

# Fast smoke test
pytest tests/test_pipeline.py::TestDataLoader -v

# API tests (requires trained model)
pytest tests/test_pipeline.py::TestAPI -v
```

---

## SAS Integration

If you have SAS licensed:

```bash
# 1. Edit the %LET PROJECT_ROOT macro in both SAS files
# 2. Run data cleaning:
sas sas/data_cleaning.sas

# 3. Run baseline statistical modeling:
sas sas/baseline_model.sas

# Outputs:
# data/processed/ae_cleaned.csv          — cleaned feature matrix
# data/processed/sas_logit_coefficients.csv
# data/processed/sas_odds_ratios.csv
# data/processed/sas_calibration.csv
```

---

## MLOps Features

- **Model Versioning:** Timestamped model artifacts in `models/`
- **Performance Tracking:** `model_metadata.json` stores all metrics
- **MLflow Integration:** Optional experiment tracking at `localhost:5000`
- **CI/CD:** GitHub Actions pipeline — lint → test → train → build → deploy
- **Data Validation:** Pandera schema validation on every data load
- **Docker:** Multi-stage build, non-root user, health checks

---

## Clinical Disclaimer

> ⚠️ **This system is intended for research purposes and clinical decision support only.**
> It is NOT a substitute for clinical judgment. All predictions must be interpreted
> by qualified clinicians in the context of the full patient presentation.
> This tool has not been cleared by the FDA or any regulatory body for clinical use.

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Data Access

MIMIC-III data requires credentialed access via PhysioNet:
1. Complete CITI training
2. Apply at https://physionet.org/content/mimiciii/
3. Sign the data use agreement
