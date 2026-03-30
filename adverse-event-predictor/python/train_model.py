"""
Adverse Event Risk Predictor — MIMIC-III
python/train_model.py

Trains: Logistic Regression, Random Forest, Gradient Boosting, Extra Trees.
Falls back gracefully when xgboost/lightgbm are not installed.
Selects best model by 5-fold cross-validated ROC-AUC.
Saves: models/ae_model.pkl + models/model_metadata.json
Dependencies: numpy, pandas, sklearn, joblib.
"""

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
logger = logging.getLogger("ae.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Model Registry ────────────────────────────────────────────────────────────
def _try_import_boosters() -> Dict[str, Any]:
    """
    Try to import XGBoost and LightGBM.
    Returns dict of available boosting models.
    """
    boosters = {}
    try:
        import xgboost as xgb
        boosters["xgboost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=3,
            eval_metric="auc", random_state=42, n_jobs=1, verbosity=0,
        )
        logger.info("XGBoost available ✓")
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        boosters["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=50, subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True, random_state=42, n_jobs=1, verbose=-1,
        )
        logger.info("LightGBM available ✓")
    except ImportError:
        pass

    return boosters


def get_model_configs() -> Dict[str, Dict]:
    """Return all model configurations (sklearn + optional boosters)."""
    configs = {
        "logistic_regression": {
            "model": LogisticRegression(
                C=0.1, penalty="l2", solver="lbfgs",
                max_iter=2000, class_weight="balanced", random_state=42,
            ),
            "description": "L2-regularised Logistic Regression (clinical baseline)",
            "interpretable": True,
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=20,
                class_weight="balanced", n_jobs=1, random_state=42,
            ),
            "description": "Random Forest — balanced class weights",
            "interpretable": False,
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            ),
            "description": "Gradient Boosting (sklearn GBDT)",
            "interpretable": False,
        },
        "extra_trees": {
            "model": ExtraTreesClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=20,
                class_weight="balanced", n_jobs=1, random_state=42,
            ),
            "description": "Extremely Randomised Trees",
            "interpretable": False,
        },
    }

    # Add XGBoost / LightGBM if available
    for name, model in _try_import_boosters().items():
        configs[name] = {
            "model": model,
            "description": f"{name.title()} gradient boosting",
            "interpretable": False,
        }

    return configs


# ── Cross-validation ──────────────────────────────────────────────────────────
def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv:      int  = 5,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Manual stratified k-fold CV to avoid joblib subprocess issues
    with packages that mock loguru in some environments.
    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    aucs, aps, f1s, precs, recs = [], [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_val, y_prob)
        ap  = average_precision_score(y_val, y_prob)
        rep = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        f1  = rep["1"]["f1-score"]
        p   = rep["1"]["precision"]
        r   = rep["1"]["recall"]

        aucs.append(auc); aps.append(ap); f1s.append(f1); precs.append(p); recs.append(r)

    metrics = {
        "cv_roc_auc_mean":   float(np.mean(aucs)),
        "cv_roc_auc_std":    float(np.std(aucs)),
        "cv_ap_mean":        float(np.mean(aps)),
        "cv_f1_mean":        float(np.mean(f1s)),
        "cv_precision_mean": float(np.mean(precs)),
        "cv_recall_mean":    float(np.mean(recs)),
    }

    if verbose:
        logger.info(
            f"  CV({cv}-fold) AUC={metrics['cv_roc_auc_mean']:.4f}±{metrics['cv_roc_auc_std']:.4f}"
            f" | AP={metrics['cv_ap_mean']:.4f}"
            f" | F1={metrics['cv_f1_mean']:.4f}"
            f" | Prec={metrics['cv_precision_mean']:.4f}"
            f" | Rec={metrics['cv_recall_mean']:.4f}"
        )

    return metrics


# ── Train all models ──────────────────────────────────────────────────────────
def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    preprocessor,
    feature_names: List[str],
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Train all models with CV and test evaluation."""
    configs = get_model_configs()
    results = {}

    sep = "=" * 62
    logger.info(f"\n{sep}")
    logger.info(f"  Training {len(configs)} models | Train={len(X_train):,} | Test={len(X_test):,}")
    logger.info(f"  Class balance — 0:{(y_train==0).sum():,} | 1:{(y_train==1).sum():,}")
    logger.info(sep)

    for name, cfg in configs.items():
        t0 = time.time()
        logger.info(f"\n[{name.upper()}] {cfg['description']}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   cfg["model"]),
        ])

        # Cross-validation
        cv_metrics = cross_validate_pipeline(pipeline, X_train, y_train, cv=cv_folds)

        # Final fit
        pipeline.fit(X_train, y_train)

        # Test evaluation
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)
        test_ap  = average_precision_score(y_test, y_prob)

        # Optimal threshold (Youden's J)
        fpr, tpr, threshs = roc_curve(y_test, y_prob)
        j_idx   = np.argmax(tpr - fpr)
        opt_thr = float(threshs[j_idx])
        y_pred  = (y_prob >= opt_thr).astype(int)
        report  = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        elapsed = time.time() - t0
        logger.info(f"  TEST → AUC={test_auc:.4f} | AP={test_ap:.4f} | "
                    f"Threshold={opt_thr:.3f} | Time={elapsed:.1f}s")

        results[name] = {
            "pipeline":          pipeline,
            "cv_metrics":        cv_metrics,
            "test_auc":          float(test_auc),
            "test_ap":           float(test_ap),
            "optimal_threshold": opt_thr,
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            },
            "classification_report": report,
            "description":       cfg["description"],
            "interpretable":     cfg["interpretable"],
            "feature_names":     feature_names,
        }

    return results


# ── Model selection ───────────────────────────────────────────────────────────
def select_best_model(results: Dict[str, Any]) -> Tuple[str, Any]:
    """Select best model by CV ROC-AUC."""
    best_name = max(results, key=lambda k: results[k]["cv_metrics"]["cv_roc_auc_mean"])
    best      = results[best_name]

    logger.info(f"\n{'='*62}")
    logger.info("  MODEL COMPARISON")
    logger.info(f"{'='*62}")
    for name, r in sorted(results.items(),
                          key=lambda x: x[1]["cv_metrics"]["cv_roc_auc_mean"],
                          reverse=True):
        marker = "★ BEST" if name == best_name else "     "
        logger.info(
            f"  {marker} {name:25s} | "
            f"CV-AUC={r['cv_metrics']['cv_roc_auc_mean']:.4f} | "
            f"Test-AUC={r['test_auc']:.4f}"
        )
    logger.info(f"\n  Best: {best_name} "
                f"(CV-AUC={best['cv_metrics']['cv_roc_auc_mean']:.4f})")
    return best_name, best


# ── Save artifacts ────────────────────────────────────────────────────────────
def save_model(
    best_name:   str,
    best_result: Dict,
    all_results: Dict,
    version:     str = "v1",
) -> Path:
    """Persist model pipeline, metadata, and comparison JSON."""
    model_path      = MODELS_DIR / "ae_model.pkl"
    meta_path       = MODELS_DIR / "model_metadata.json"
    comparison_path = MODELS_DIR / "all_model_results.json"

    joblib.dump(best_result["pipeline"], model_path)
    logger.info(f"  Model saved → {model_path}  ({model_path.stat().st_size/1024:.0f} KB)")

    metadata = {
        "version":           version,
        "best_model_name":   best_name,
        "description":       best_result["description"],
        "feature_names":     best_result["feature_names"],
        "optimal_threshold": best_result["optimal_threshold"],
        "cv_metrics":        best_result["cv_metrics"],
        "test_metrics": {
            "roc_auc":       best_result["test_auc"],
            "avg_precision": best_result["test_ap"],
        },
        "classification_report": best_result["classification_report"],
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata  → {meta_path}")

    comparison = {
        name: {
            "cv_auc":    r["cv_metrics"]["cv_roc_auc_mean"],
            "test_auc":  r["test_auc"],
            "test_ap":   r["test_ap"],
            "roc_fpr":   r["roc_curve"]["fpr"],
            "roc_tpr":   r["roc_curve"]["tpr"],
        }
        for name, r in all_results.items()
    }
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"  Comparison→ {comparison_path}")

    return model_path


# ── Full training pipeline ────────────────────────────────────────────────────
def main(
    data_source: str = "synthetic",
    cv_folds:    int = 5,
    test_size:   float = 0.2,
    version:     str = "v1",
    n_patients:  int = 5000,
) -> Dict:
    """
    Full training pipeline.

    Returns:
        Dict with summary metrics.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from data_loader        import load_data, FEATURE_COLS, TARGET_COL
    from feature_engineering import engineer_features, build_preprocessing_pipeline

    logger.info("=" * 62)
    logger.info("  ADVERSE EVENT RISK PREDICTOR — Training Pipeline")
    logger.info("=" * 62)

    # Load data
    df = load_data(source=data_source, n=n_patients)

    # Feature engineering
    X, feat_names = engineer_features(df, feature_cols=FEATURE_COLS, add_interactions=True)
    y = df[TARGET_COL].astype(int)
    logger.info(f"Feature matrix: {X.shape} | Events: {y.sum():,}/{len(y):,} ({y.mean():.1%})")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    logger.info(f"Split: train={len(X_train):,} | test={len(X_test):,}")

    # Build preprocessor (no clinical scorer here — already done in engineer_features)
    preprocessor = build_preprocessing_pipeline(
        use_clinical_features=False,
        scaler_type="robust",
        imputer_strategy="median",
    )

    # Train all models
    all_results = train_all_models(
        X_train, y_train, X_test, y_test,
        preprocessor, feat_names, cv_folds,
    )

    # Select best and save
    best_name, best_result = select_best_model(all_results)
    model_path = save_model(best_name, best_result, all_results, version)

    return {
        "best_model":    best_name,
        "model_path":    str(model_path),
        "test_auc":      best_result["test_auc"],
        "feature_count": len(feat_names),
        "train_n":       len(X_train),
        "test_n":        len(X_test),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Adverse Event Risk Predictor")
    parser.add_argument("--source",     default="synthetic", choices=["synthetic", "csv", "db"])
    parser.add_argument("--cv",         type=int, default=5)
    parser.add_argument("--version",    default="v1")
    parser.add_argument("--n-patients", type=int, default=5000)
    args = parser.parse_args()

    result = main(
        data_source=args.source,
        cv_folds=args.cv,
        version=args.version,
        n_patients=args.n_patients,
    )

    print("\n" + "=" * 50)
    print("  TRAINING COMPLETE")
    print("=" * 50)
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
