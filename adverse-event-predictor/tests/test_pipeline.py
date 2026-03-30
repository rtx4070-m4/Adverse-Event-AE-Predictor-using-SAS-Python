"""
Adverse Event Risk Predictor — Test Suite
tests/test_pipeline.py

Tests all pipeline components: data loading, feature engineering,
model training, evaluation, and API endpoints.
Run: pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "api"))


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestDataLoader:

    def test_synthetic_generates_records(self):
        from data_loader import generate_synthetic_mimic
        df = generate_synthetic_mimic(n=200, seed=0)
        assert len(df) == 200

    def test_synthetic_has_all_feature_cols(self):
        from data_loader import generate_synthetic_mimic, FEATURE_COLS, TARGET_COL
        df = generate_synthetic_mimic(n=200)
        for col in FEATURE_COLS + [TARGET_COL]:
            assert col in df.columns, f"Missing column: {col}"

    def test_age_within_clinical_range(self):
        from data_loader import generate_synthetic_mimic
        df = generate_synthetic_mimic(n=500)
        assert df["age"].min() >= 18
        assert df["age"].max() <= 110

    def test_gender_is_binary(self):
        from data_loader import generate_synthetic_mimic
        df = generate_synthetic_mimic(n=500)
        assert set(df["gender"].unique()).issubset({0.0, 1.0})

    def test_target_is_binary(self):
        from data_loader import generate_synthetic_mimic
        df = generate_synthetic_mimic(n=500)
        assert set(df["adverse_event"].unique()).issubset({0.0, 1.0})

    def test_no_missing_in_features(self):
        from data_loader import generate_synthetic_mimic, FEATURE_COLS
        df = generate_synthetic_mimic(n=500)
        assert df[FEATURE_COLS].isnull().sum().sum() == 0

    def test_positive_los(self):
        from data_loader import generate_synthetic_mimic
        df = generate_synthetic_mimic(n=500)
        assert (df["length_of_stay"] > 0).all()

    def test_adverse_event_rate_realistic(self):
        from data_loader import generate_synthetic_mimic
        df = generate_synthetic_mimic(n=2000)
        rate = df["adverse_event"].mean()
        assert 0.03 < rate < 0.40, f"Unrealistic AE rate: {rate:.2%}"

    def test_load_data_synthetic(self):
        from data_loader import load_data, FEATURE_COLS, TARGET_COL
        df = load_data(source="synthetic", n=300, validate=True)
        assert len(df) > 0
        for col in FEATURE_COLS + [TARGET_COL]:
            assert col in df.columns

    def test_load_data_csv_fallback(self, tmp_path):
        """CSV loader falls back to synthetic when file missing."""
        from data_loader import load_data
        df = load_data(source="csv", path=tmp_path / "nonexistent.csv", n=200)
        assert len(df) > 0

    def test_get_data_summary_keys(self):
        from data_loader import load_data, get_data_summary
        df = load_data(source="synthetic", n=300)
        s  = get_data_summary(df)
        for key in ["n_patients", "adverse_event_rate", "age_mean", "n_features"]:
            assert key in s

    def test_split_features_target_shapes(self):
        from data_loader import load_data, split_features_target, FEATURE_COLS
        df = load_data(source="synthetic", n=300)
        X, y = split_features_target(df, FEATURE_COLS)
        assert X.shape[0] == y.shape[0] == len(df)
        assert X.shape[1] == len(FEATURE_COLS)
        assert set(y.unique()).issubset({0, 1})


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestFeatureEngineering:

    @pytest.fixture(autouse=True)
    def load_df(self):
        from data_loader import load_data
        self.df = load_data(source="synthetic", n=300)

    def test_clinical_risk_scorer_adds_features(self):
        from feature_engineering import ClinicalRiskScorer
        from data_loader import FEATURE_COLS
        scorer = ClinicalRiskScorer()
        X_in  = self.df[FEATURE_COLS].astype(float)
        X_out = scorer.transform(X_in)
        assert X_out.shape[1] > X_in.shape[1]
        assert "tachycardia_flag"     in X_out.columns
        assert "renal_impairment_flag" in X_out.columns
        assert "log_creatinine"       in X_out.columns

    def test_tachycardia_flag_correct(self):
        from feature_engineering import ClinicalRiskScorer
        from data_loader import FEATURE_COLS
        scorer = ClinicalRiskScorer()
        row    = {col: [80.0] for col in FEATURE_COLS}
        # Tachycardic patient
        row["heart_rate_mean"] = [110.0]
        out = scorer.transform(pd.DataFrame(row))
        assert out["tachycardia_flag"].iloc[0] == 1.0
        # Normal HR
        row["heart_rate_mean"] = [70.0]
        out = scorer.transform(pd.DataFrame(row))
        assert out["tachycardia_flag"].iloc[0] == 0.0

    def test_renal_flag_correct(self):
        from feature_engineering import ClinicalRiskScorer
        from data_loader import FEATURE_COLS
        scorer = ClinicalRiskScorer()
        row = {col: [1.0] for col in FEATURE_COLS}
        row["creatinine_max"] = [2.0]
        out = scorer.transform(pd.DataFrame(row))
        assert out["renal_impairment_flag"].iloc[0] == 1.0

    def test_engineer_features_shape(self):
        from feature_engineering import engineer_features
        from data_loader import FEATURE_COLS
        X, names = engineer_features(self.df, feature_cols=FEATURE_COLS)
        assert X.shape[0] == len(self.df)
        assert X.shape[1] == len(names)
        assert len(names) > len(FEATURE_COLS)

    def test_no_nan_after_engineering(self):
        from feature_engineering import engineer_features
        from data_loader import FEATURE_COLS
        X, _ = engineer_features(self.df, feature_cols=FEATURE_COLS)
        assert not X.isnull().any().any()

    def test_outlier_clipper(self):
        from feature_engineering import OutlierClipper
        clipper = OutlierClipper(0.01, 0.99)
        X = np.random.randn(500, 4)
        X[0, 0] = 1e9    # extreme outlier
        clipper.fit(X)
        X_clipped = clipper.transform(X)
        assert X_clipped[0, 0] < 1e9

    def test_preprocessing_pipeline_no_nan(self):
        from feature_engineering import build_preprocessing_pipeline
        from data_loader import FEATURE_COLS
        prep = build_preprocessing_pipeline(use_clinical_features=False)
        X    = self.df[FEATURE_COLS].astype(float)
        Xt   = prep.fit_transform(X)
        assert not np.isnan(Xt).any()

    def test_feature_metadata_schema(self):
        from feature_engineering import get_feature_metadata
        meta = get_feature_metadata()
        assert "name" in meta.columns
        assert "category" in meta.columns
        assert len(meta) == 9


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestModelTraining:

    def test_model_configs_contain_required_models(self):
        from train_model import get_model_configs
        configs = get_model_configs()
        assert "logistic_regression" in configs
        assert "random_forest"       in configs

    def test_cross_validate_returns_auc(self):
        from train_model import cross_validate_pipeline, get_model_configs
        from feature_engineering import build_preprocessing_pipeline, engineer_features
        from data_loader import load_data, FEATURE_COLS, TARGET_COL
        from sklearn.pipeline import Pipeline

        df = load_data(source="synthetic", n=600)
        X, _ = engineer_features(df, feature_cols=FEATURE_COLS)
        y    = df[TARGET_COL].astype(int)
        prep = build_preprocessing_pipeline(use_clinical_features=False)

        cfg  = get_model_configs()["logistic_regression"]
        pipe = Pipeline([("prep", prep), ("clf", cfg["model"])])
        metrics = cross_validate_pipeline(pipe, X, y, cv=3, verbose=False)

        assert "cv_roc_auc_mean" in metrics
        assert 0.5 <= metrics["cv_roc_auc_mean"] <= 1.0
        assert "cv_precision_mean" in metrics

    def test_model_selection_picks_best(self):
        from train_model import select_best_model
        dummy = {
            "model_a": {"cv_metrics": {"cv_roc_auc_mean": 0.70}, "test_auc": 0.69,
                        "test_ap": 0.3, "roc_curve": {"fpr":[], "tpr":[]},
                        "pipeline": None, "optimal_threshold": 0.5,
                        "classification_report": {}, "description": "A",
                        "interpretable": True, "feature_names": []},
            "model_b": {"cv_metrics": {"cv_roc_auc_mean": 0.82}, "test_auc": 0.80,
                        "test_ap": 0.4, "roc_curve": {"fpr":[], "tpr":[]},
                        "pipeline": None, "optimal_threshold": 0.5,
                        "classification_report": {}, "description": "B",
                        "interpretable": False, "feature_names": []},
        }
        name, result = select_best_model(dummy)
        assert name == "model_b"

    def test_full_training_pipeline(self, tmp_path, monkeypatch):
        """Smoke-test the full training pipeline end-to-end."""
        monkeypatch.setattr("train_model.MODELS_DIR", tmp_path)
        from train_model import main
        result = main(
            data_source="synthetic",
            cv_folds=2,
            version="test_v0",
            n_patients=800,
        )
        assert "best_model"    in result
        assert "test_auc"      in result
        assert result["test_auc"] > 0.50
        assert (tmp_path / "ae_model.pkl").exists()
        assert (tmp_path / "model_metadata.json").exists()

    def test_saved_metadata_structure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("train_model.MODELS_DIR", tmp_path)
        from train_model import main
        main(data_source="synthetic", cv_folds=2, n_patients=800)

        with open(tmp_path / "model_metadata.json") as f:
            meta = json.load(f)

        for key in ["version", "best_model_name", "feature_names",
                    "optimal_threshold", "cv_metrics", "test_metrics"]:
            assert key in meta, f"Missing key: {key}"

        assert 0 < meta["optimal_threshold"] < 1
        assert meta["test_metrics"]["roc_auc"] > 0.5


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestEvaluation:

    def test_compute_metrics_keys(self):
        from evaluate_model import compute_metrics
        rng    = np.random.default_rng(0)
        y_true = rng.binomial(1, 0.15, 300)
        y_prob = rng.uniform(0, 1, 300)
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        for key in ["roc_auc", "avg_precision", "sensitivity",
                    "specificity", "ppv", "npv", "tp", "tn", "fp", "fn"]:
            assert key in m

    def test_compute_metrics_range(self):
        from evaluate_model import compute_metrics
        rng    = np.random.default_rng(1)
        y_true = rng.binomial(1, 0.2, 500)
        y_prob = rng.beta(2, 5, 500)
        m = compute_metrics(y_true, y_prob)
        assert 0 <= m["roc_auc"]     <= 1
        assert 0 <= m["sensitivity"] <= 1
        assert 0 <= m["specificity"] <= 1
        assert 0 <= m["ppv"]         <= 1
        assert 0 <= m["npv"]         <= 1

    def test_plots_saved(self, tmp_path, monkeypatch):
        from evaluate_model import compute_metrics, plot_roc_curve, plot_confusion_matrix
        monkeypatch.setattr("evaluate_model.REPORTS_DIR", tmp_path)
        rng    = np.random.default_rng(2)
        y_true = rng.binomial(1, 0.15, 400)
        y_prob = rng.beta(2, 5, 400)
        plot_roc_curve(y_true, y_prob, model_name="test_model")
        m = compute_metrics(y_true, y_prob)
        plot_confusion_matrix(m)
        assert (tmp_path / "roc_curve.png").exists()
        assert (tmp_path / "confusion_matrix.png").exists()


# ══════════════════════════════════════════════════════════════════════════════
# API TESTS  (requires fastapi + httpx)
# ══════════════════════════════════════════════════════════════════════════════
class TestAPI:

    @pytest.fixture(scope="class")
    def client(self):
        """Set up FastAPI test client, training model if needed."""
        try:
            from fastapi.testclient import TestClient
            import httpx  # noqa
        except ImportError:
            pytest.skip("fastapi / httpx not installed — skipping API tests")

        # Ensure model exists
        if not (MODELS_DIR / "ae_model.pkl").exists():
            from train_model import main
            main(data_source="synthetic", cv_folds=2, n_patients=1000)

        from app import app as fastapi_app
        return TestClient(fastapi_app)

    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_model_info_ok(self, client):
        r = client.get("/model-info")
        assert r.status_code == 200
        data = r.json()
        assert "model_name"  in data
        assert "performance" in data

    def test_predict_valid_patient(self, client):
        payload = {
            "age": 68, "gender": 1, "length_of_stay": 4.5,
            "heart_rate_mean": 92, "creatinine_max": 1.8,
            "wbc_mean": 12.5, "drug_count": 10,
            "polypharmacy_score": 5.0, "lab_abnormality_score": 2,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert 0 <= data["risk_score"] <= 1
        assert data["risk_category"] in {"Low", "Moderate", "High", "Critical"}
        assert data["adverse_event_flag"] in {0, 1}

    def test_predict_invalid_age(self, client):
        payload = {
            "age": 5,  # below minimum 18
            "gender": 1, "length_of_stay": 4.5, "heart_rate_mean": 92,
            "creatinine_max": 1.8, "wbc_mean": 12.5, "drug_count": 10,
            "polypharmacy_score": 5.0, "lab_abnormality_score": 2,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 422    # Pydantic validation error

    def test_batch_prediction(self, client):
        patients = [
            {"age": 55, "gender": 0, "length_of_stay": 2.0,
             "heart_rate_mean": 78, "creatinine_max": 0.9,
             "wbc_mean": 8.0, "drug_count": 4, "polypharmacy_score": 2.0,
             "lab_abnormality_score": 0},
            {"age": 82, "gender": 1, "length_of_stay": 12.0,
             "heart_rate_mean": 108, "creatinine_max": 3.5,
             "wbc_mean": 16.0, "drug_count": 18, "polypharmacy_score": 9.0,
             "lab_abnormality_score": 4},
        ]
        r = client.post("/predict/batch", json={"patients": patients})
        assert r.status_code == 200
        data = r.json()
        assert data["n_patients"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_too_large(self, client):
        patients = [{
            "age": 55, "gender": 0, "length_of_stay": 2.0,
            "heart_rate_mean": 78, "creatinine_max": 0.9,
            "wbc_mean": 8.0, "drug_count": 4, "polypharmacy_score": 2.0,
            "lab_abnormality_score": 0,
        }] * 101
        r = client.post("/predict/batch", json={"patients": patients})
        assert r.status_code == 400

    def test_risk_score_monotonicity(self, client):
        """Sicker patient should score higher than healthy patient."""
        healthy = {
            "age": 25, "gender": 0, "length_of_stay": 0.5,
            "heart_rate_mean": 68, "creatinine_max": 0.7,
            "wbc_mean": 6.5, "drug_count": 1, "polypharmacy_score": 0.5,
            "lab_abnormality_score": 0,
        }
        sick = {
            "age": 85, "gender": 1, "length_of_stay": 15.0,
            "heart_rate_mean": 115, "creatinine_max": 5.0,
            "wbc_mean": 20.0, "drug_count": 20, "polypharmacy_score": 10.0,
            "lab_abnormality_score": 6,
        }
        r1 = client.post("/predict", json=healthy)
        r2 = client.post("/predict", json=sick)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.json()["risk_score"] > r1.json()["risk_score"]


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST
# ══════════════════════════════════════════════════════════════════════════════
class TestEndToEnd:

    def test_full_pipeline_auc_above_threshold(self):
        """
        Full pipeline: load → engineer → train → predict.
        Asserts final AUC > 0.55 (well above random).
        """
        from data_loader          import load_data, FEATURE_COLS, TARGET_COL
        from feature_engineering  import engineer_features, build_preprocessing_pipeline
        from sklearn.ensemble      import ExtraTreesClassifier
        from sklearn.pipeline      import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics        import roc_auc_score

        df = load_data(source="synthetic", n=1500)
        X, _ = engineer_features(df, feature_cols=FEATURE_COLS)
        y    = df[TARGET_COL].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=7
        )

        pipe = Pipeline([
            ("prep", build_preprocessing_pipeline(use_clinical_features=False)),
            ("clf",  ExtraTreesClassifier(n_estimators=100, class_weight="balanced",
                                          n_jobs=1, random_state=7)),
        ])
        pipe.fit(X_train, y_train)
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

        assert auc > 0.55, f"AUC {auc:.4f} below threshold 0.55"

    def test_high_risk_patient_scores_higher(self):
        """Clinically high-risk patient must score above low-risk patient."""
        import joblib
        from feature_engineering import engineer_features
        from data_loader import FEATURE_COLS

        model_path = MODELS_DIR / "ae_model.pkl"
        if not model_path.exists():
            from train_model import main
            main(data_source="synthetic", cv_folds=2, n_patients=1000)

        model = joblib.load(model_path)

        low_risk = pd.DataFrame([{
            "age": 22, "gender": 0, "length_of_stay": 0.5,
            "heart_rate_mean": 65, "creatinine_max": 0.6,
            "wbc_mean": 6.0, "drug_count": 1, "polypharmacy_score": 0.5,
            "lab_abnormality_score": 0,
        }])
        high_risk = pd.DataFrame([{
            "age": 88, "gender": 1, "length_of_stay": 20.0,
            "heart_rate_mean": 120, "creatinine_max": 6.0,
            "wbc_mean": 22.0, "drug_count": 25, "polypharmacy_score": 10.0,
            "lab_abnormality_score": 7,
        }])

        X_low,  _ = engineer_features(low_risk,  feature_cols=FEATURE_COLS)
        X_high, _ = engineer_features(high_risk, feature_cols=FEATURE_COLS)

        score_low  = model.predict_proba(X_low)[0, 1]
        score_high = model.predict_proba(X_high)[0, 1]

        assert score_high > score_low, (
            f"High-risk patient scored {score_high:.4f} ≤ "
            f"low-risk {score_low:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
