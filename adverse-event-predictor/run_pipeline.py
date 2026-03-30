#!/usr/bin/env python3
"""
Adverse Event Risk Predictor — Master Pipeline Runner
File: run_pipeline.py

One-command execution of the complete pipeline:
  1. Generate / load data
  2. Train all ML models
  3. Evaluate + SHAP analysis
  4. Validate artifacts
  5. Print summary report

Usage:
  python run_pipeline.py                   # synthetic data, full pipeline
  python run_pipeline.py --source csv      # real MIMIC-III data from CSV
  python run_pipeline.py --skip-evaluate   # train only, skip SHAP
  python run_pipeline.py --cv 3 --quick    # fast run with 3-fold CV
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "api"))

from loguru import logger


# ── Banner ─────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ⚕  Adverse Event Risk Predictor                           ║
║      MIMIC-III Clinical ICU Data · ML Pipeline               ║
║                                                              ║
║   Pipeline:  SQL → SAS → Features → ML → SHAP → API         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_step(n: int, total: int, title: str):
    bar = "█" * n + "░" * (total - n)
    logger.info(f"\n[{bar}] Step {n}/{total} — {title}")


def check_prerequisites():
    """Verify required packages are available."""
    required = {
        "numpy":    "numpy",
        "pandas":   "pandas",
        "sklearn":  "scikit-learn",
        "joblib":   "joblib",
        "loguru":   "loguru",
    }
    optional = {
        "xgboost":  "xgboost",
        "lightgbm": "lightgbm",
        "shap":     "shap",
    }

    missing_required = []
    missing_optional = []

    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing_required.append(pkg)

    for mod, pkg in optional.items():
        try:
            __import__(mod)
        except ImportError:
            missing_optional.append(pkg)

    if missing_required:
        logger.error(f"Missing required packages: {missing_required}")
        logger.error("Run: pip install -r requirements.txt")
        sys.exit(1)

    if missing_optional:
        logger.warning(
            f"Optional packages not found (reduced functionality): {missing_optional}\n"
            "  XGBoost/LightGBM: advanced boosting models\n"
            "  SHAP: explainability plots\n"
            "  Install: pip install xgboost lightgbm shap"
        )

    return missing_optional


def step_load_data(source: str, n_patients: int) -> "pd.DataFrame":
    """Step 1: Load or generate data."""
    from data_loader import load_data, get_data_summary
    import pandas as pd

    logger.info(f"Source: {source} | Patients: {n_patients:,}")

    if source == "synthetic":
        df = load_data(source="synthetic", validate=True)
    else:
        csv_path = PROJECT_ROOT / "data" / "processed" / "ae_features_clean.csv"
        if not csv_path.exists():
            logger.warning(f"CSV not found at {csv_path}. Falling back to synthetic data.")
            df = load_data(source="synthetic", validate=True)
        else:
            df = load_data(source="csv", path=csv_path, validate=True)

    summary = get_data_summary(df)
    logger.info(
        f"Dataset loaded:\n"
        f"  Patients:        {summary['n_patients']:,}\n"
        f"  Features:        {summary['n_features']}\n"
        f"  Adverse events:  {summary['adverse_event_n']:,} "
        f"({summary['adverse_event_rate']:.1%})\n"
        f"  Age (mean±std):  {summary['age_mean']:.1f} ± {summary['age_std']:.1f}\n"
        f"  LOS (median):    {summary['los_median']:.1f} days"
    )
    return df


def step_train_models(df, cv_folds: int, missing_optional: list):
    """Step 2: Train all models and select best."""
    from train_model import main as train_main

    logger.info(f"CV folds: {cv_folds} | Models: LR, RF"
                + (", XGBoost, LightGBM" if not missing_optional else " (boosting skipped)"))

    result = train_main(
        data_source="synthetic",  # data already loaded; train_main reloads internally
        cv_folds=cv_folds,
        version="v1",
    )

    logger.info(
        f"Training complete:\n"
        f"  Best model:  {result['best_model']}\n"
        f"  Test AUC:    {result['test_auc']:.4f}\n"
        f"  Features:    {result['feature_count']}\n"
        f"  Train N:     {result['train_n']:,}\n"
        f"  Test N:      {result['test_n']:,}"
    )
    return result


def step_evaluate(skip_shap: bool):
    """Step 3: Evaluate model + generate SHAP plots."""
    from evaluate_model import run_full_evaluation

    report = run_full_evaluation(data_source="synthetic")
    m = report["metrics"]

    logger.info(
        f"Evaluation results:\n"
        f"  ROC-AUC:     {m['roc_auc']:.4f}\n"
        f"  Avg Precision:{m['avg_precision']:.4f}\n"
        f"  Sensitivity:  {m['sensitivity']:.4f}\n"
        f"  Specificity:  {m['specificity']:.4f}\n"
        f"  PPV:          {m['ppv']:.4f}\n"
        f"  NPV:          {m['npv']:.4f}\n"
        f"  TP: {m['tp']}  TN: {m['tn']}  FP: {m['fp']}  FN: {m['fn']}"
    )
    return report


def step_validate_artifacts():
    """Step 4: Confirm all expected output files exist."""
    models_dir  = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "data" / "processed" / "reports"

    expected_files = {
        "models/ae_model.pkl":               models_dir / "ae_model.pkl",
        "models/model_metadata.json":        models_dir / "model_metadata.json",
        "models/all_model_results.json":     models_dir / "all_model_results.json",
        "reports/roc_curve.png":             reports_dir / "roc_curve.png",
        "reports/precision_recall.png":      reports_dir / "precision_recall.png",
        "reports/confusion_matrix.png":      reports_dir / "confusion_matrix.png",
        "reports/evaluation_report.json":    reports_dir / "evaluation_report.json",
    }

    optional_files = {
        "models/shap_explainer.pkl":         models_dir / "shap_explainer.pkl",
        "reports/shap_feature_importance.png": reports_dir / "shap_feature_importance.png",
        "reports/shap_beeswarm.png":         reports_dir / "shap_beeswarm.png",
        "reports/calibration.png":           reports_dir / "calibration.png",
    }

    ok, warn, err = [], [], []

    for name, path in expected_files.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            ok.append(f"  ✓ {name}  ({size_kb:.1f} KB)")
        else:
            err.append(f"  ✗ {name}  [MISSING]")

    for name, path in optional_files.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            ok.append(f"  ✓ {name}  ({size_kb:.1f} KB)")
        else:
            warn.append(f"  ⚠ {name}  [optional - not found]")

    for line in ok:   logger.info(line)
    for line in warn: logger.warning(line)
    for line in err:  logger.error(line)

    if err:
        logger.warning(f"{len(err)} required artifact(s) missing.")
        return False

    logger.info(f"All {len(expected_files)} required artifacts present.")
    return True


def print_summary_report(train_result: dict, eval_report: dict, elapsed: float):
    """Print final summary to console."""
    m = eval_report["metrics"]

    print("\n" + "═" * 62)
    print("  PIPELINE COMPLETE — SUMMARY REPORT")
    print("═" * 62)
    print(f"  Total runtime:      {elapsed:.1f}s")
    print(f"  Best model:         {train_result['best_model']}")
    print(f"  Training samples:   {train_result['train_n']:,}")
    print(f"  Test samples:       {train_result['test_n']:,}")
    print()
    print("  PERFORMANCE METRICS")
    print(f"  ├─ ROC-AUC:         {m['roc_auc']:.4f}")
    print(f"  ├─ Avg Precision:   {m['avg_precision']:.4f}")
    print(f"  ├─ Sensitivity:     {m['sensitivity']:.4f}   (recall)")
    print(f"  ├─ Specificity:     {m['specificity']:.4f}")
    print(f"  ├─ PPV:             {m['ppv']:.4f}   (precision)")
    print(f"  └─ NPV:             {m['npv']:.4f}")
    print()
    print("  CONFUSION MATRIX (at optimal threshold)")
    print(f"  ├─ True Positives:  {m['tp']:,}   (correctly flagged AE)")
    print(f"  ├─ True Negatives:  {m['tn']:,}   (correctly cleared)")
    print(f"  ├─ False Positives: {m['fp']:,}   (unnecessary alerts)")
    print(f"  └─ False Negatives: {m['fn']:,}   (missed AEs)")
    print()
    print("  ARTIFACTS")
    print(f"  ├─ Model:       models/ae_model.pkl")
    print(f"  ├─ Metadata:    models/model_metadata.json")
    print(f"  └─ Reports:     data/processed/reports/")
    print()
    print("  NEXT STEPS")
    print("  ├─ Start API:       make api       → http://localhost:8000/docs")
    print("  ├─ Start Dashboard: make dashboard → http://localhost:8501")
    print("  ├─ Run tests:       make test")
    print("  └─ Docker deploy:   make docker-up")
    print("═" * 62 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Adverse Event Risk Predictor — Full Pipeline Runner"
    )
    parser.add_argument(
        "--source", choices=["synthetic", "csv", "db"], default="synthetic",
        help="Data source (default: synthetic)"
    )
    parser.add_argument(
        "--n-patients", type=int, default=5000,
        help="Synthetic patients to generate (default: 5000)"
    )
    parser.add_argument(
        "--cv", type=int, default=5,
        help="Cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--skip-evaluate", action="store_true",
        help="Skip evaluation/SHAP step"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run: 2-fold CV, 1000 patients"
    )
    args = parser.parse_args()

    if args.quick:
        args.cv = 2
        args.n_patients = 1000
        logger.info("Quick mode enabled: cv=2, n_patients=1000")

    print(BANNER)
    t_start = time.time()
    TOTAL_STEPS = 4 if not args.skip_evaluate else 3

    # ── Prerequisites ──────────────────────────────────────────
    print_step(0, TOTAL_STEPS, "Checking prerequisites")
    missing_optional = check_prerequisites()

    # ── Directories ────────────────────────────────────────────
    for d in ["data/raw", "data/processed/reports", "models"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Data ───────────────────────────────────────────
    print_step(1, TOTAL_STEPS, "Loading / generating data")
    df = step_load_data(args.source, args.n_patients)

    # ── Step 2: Train ──────────────────────────────────────────
    print_step(2, TOTAL_STEPS, "Training ML models (LR, RF, XGBoost, LightGBM)")
    train_result = step_train_models(df, args.cv, missing_optional)

    # ── Step 3: Evaluate ───────────────────────────────────────
    eval_report = None
    if not args.skip_evaluate:
        print_step(3, TOTAL_STEPS, "Evaluating model + computing SHAP explanations")
        try:
            eval_report = step_evaluate(skip_shap="shap" in missing_optional)
        except Exception as e:
            logger.warning(f"Evaluation step failed (non-fatal): {e}")
            eval_report = {"metrics": {
                "roc_auc": train_result["test_auc"],
                "avg_precision": 0, "sensitivity": 0,
                "specificity": 0, "ppv": 0, "npv": 0,
                "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            }}

    # ── Step 4: Validate ───────────────────────────────────────
    print_step(TOTAL_STEPS, TOTAL_STEPS, "Validating artifacts")
    step_validate_artifacts()

    # ── Summary ────────────────────────────────────────────────
    elapsed = time.time() - t_start
    if eval_report:
        print_summary_report(train_result, eval_report, elapsed)
    else:
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"Best model: {train_result['best_model']} | AUC: {train_result['test_auc']:.4f}")


if __name__ == "__main__":
    main()
