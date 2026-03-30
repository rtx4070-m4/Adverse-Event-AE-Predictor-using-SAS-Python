"""
Adverse Event Risk Predictor — MIMIC-III
python/evaluate_model.py

Model evaluation, feature importance, and SHAP-equivalent explanations.
When shap is installed, uses real SHAP TreeExplainer / LinearExplainer.
Falls back to permutation importance when shap is unavailable.
Generates: ROC, PR, calibration, confusion matrix, feature importance plots.
Dependencies: numpy, pandas, sklearn, matplotlib. (shap optional)
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logger = logging.getLogger("ae.evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "data" / "processed" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dark theme palette ────────────────────────────────────────────────────────
DARK_BG    = "#0f1117"
PANEL_BG   = "#1a1d27"
BLUE       = "#4f9cf9"
RED        = "#ff4d6d"
GREEN      = "#39d98a"
YELLOW     = "#ffcc00"
PURPLE     = "#bf5af2"
MUTED      = "#8890a8"
TEXT       = "#e0e4f0"
GRID       = "#2d3141"

def _dark_fig(figsize=(9, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.5, linewidth=0.5)
    return fig, ax


# ── Core metrics ──────────────────────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """Compute comprehensive binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc":       float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "threshold":     float(threshold),
        "sensitivity":   float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity":   float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "ppv":           float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "npv":           float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_patients": int(len(y_true)),
        "n_adverse":  int(y_true.sum()),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    all_results: Optional[Dict] = None,
    model_name: str = "Best Model",
) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = _dark_fig()

    # Plot all model ROC curves if provided
    if all_results:
        colors = [PURPLE, YELLOW, MUTED, "#ff7a1a"]
        for (name, r), color in zip(all_results.items(), colors):
            if name != model_name.lower().replace(" ", "_"):
                ax.plot(r["roc_fpr"], r["roc_tpr"], color=color,
                        lw=1.2, alpha=0.5, ls="--",
                        label=f"{name} (AUC={r['cv_auc']:.3f})")

    ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"{model_name} (AUC={auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color=BLUE)
    ax.plot([0, 1], [0, 1], ":", color=MUTED, lw=1.2, label="Random Chance")

    ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=11, color=TEXT)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, color=TEXT)
    ax.set_title("ROC Curve — Adverse Event Risk Predictor", fontsize=13,
                 fontweight="bold", color=TEXT)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.02])

    # AUC annotation
    ax.text(0.62, 0.12, f"AUC = {auc:.4f}", transform=ax.transAxes,
            fontsize=13, color=GREEN, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_BG, edgecolor=GREEN, alpha=0.9))

    path = REPORTS_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"  ROC curve → {path}")
    return path


def plot_precision_recall(y_true: np.ndarray, y_prob: np.ndarray) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap       = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = _dark_fig()
    ax.plot(recall, precision, color=PURPLE, lw=2.5, label=f"Model (AP={ap:.3f})")
    ax.axhline(baseline, color=YELLOW, ls=":", lw=1.5, label=f"Baseline rate={baseline:.3f}")
    ax.fill_between(recall, precision, baseline, alpha=0.12, color=PURPLE)

    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, color=TEXT)
    ax.set_ylabel("Precision (PPV)", fontsize=11, color=TEXT)
    ax.set_title("Precision–Recall Curve", fontsize=13, fontweight="bold", color=TEXT)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
    ax.set_xlim([0, 1.01]); ax.set_ylim([0, 1.05])

    path = REPORTS_DIR / "precision_recall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"  PR curve  → {path}")
    return path


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Path:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values(): spine.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED)
        ax.grid(True, color=GRID, alpha=0.5, linewidth=0.5)

    # Calibration plot
    ax1.plot(prob_pred, prob_true, "o-", color=BLUE, lw=2, ms=6, label="Model")
    ax1.plot([0, 1], [0, 1], ":", color=MUTED, lw=1.5, label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Probability", color=TEXT, fontsize=10)
    ax1.set_ylabel("Fraction Positives", color=TEXT, fontsize=10)
    ax1.set_title("Calibration Curve", color=TEXT, fontsize=12, fontweight="bold")
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)

    # Score distribution
    ax2.hist(y_prob[y_true == 0], bins=40, alpha=0.65, color=GREEN,
             label="No Adverse Event", density=True)
    ax2.hist(y_prob[y_true == 1], bins=40, alpha=0.65, color=RED,
             label="Adverse Event", density=True)
    ax2.set_xlabel("Predicted Probability", color=TEXT, fontsize=10)
    ax2.set_ylabel("Density", color=TEXT, fontsize=10)
    ax2.set_title("Score Distribution by Outcome", color=TEXT, fontsize=12, fontweight="bold")
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle("Model Calibration Analysis", color=TEXT, fontsize=14, fontweight="bold")
    path = REPORTS_DIR / "calibration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"  Calibration → {path}")
    return path


def plot_confusion_matrix(metrics: Dict) -> Path:
    cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ])

    fig, ax = _dark_fig((7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, shrink=0.85)

    labels = [
        [f"TN\n{metrics['tn']:,}", f"FP\n{metrics['fp']:,}"],
        [f"FN\n{metrics['fn']:,}", f"TP\n{metrics['tp']:,}"],
    ]
    for i in range(2):
        for j in range(2):
            c = "white" if cm[i, j] > cm.max() * 0.5 else BLUE
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=14, color=c, fontweight="bold")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNo AE", "Predicted\nAE"], color=TEXT)
    ax.set_yticklabels(["Actual\nNo AE", "Actual\nAE"], color=TEXT)
    ax.set_title(
        f"Confusion Matrix  |  Sens={metrics['sensitivity']:.3f}  Spec={metrics['specificity']:.3f}",
        fontsize=12, fontweight="bold", color=TEXT,
    )

    path = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"  Confusion matrix → {path}")
    return path


# ── Feature Importance / SHAP ─────────────────────────────────────────────────
def _shap_importance(pipeline, X_sample: pd.DataFrame, feature_names: List[str]) -> Optional[np.ndarray]:
    """Try real SHAP values; return None if shap not installed."""
    try:
        import shap
        # Extract preprocessor and classifier
        steps     = list(pipeline.named_steps.items())
        preproc   = pipeline[:-1]
        clf       = pipeline[-1]
        X_pre     = preproc.transform(X_sample)

        model_type = type(clf).__name__.lower()
        if any(t in model_type for t in ["xgb", "lgbm", "lightgbm"]):
            explainer = shap.TreeExplainer(clf)
        elif "forest" in model_type or "tree" in model_type or "boosting" in model_type:
            explainer = shap.TreeExplainer(clf)
        else:
            explainer = shap.LinearExplainer(clf, X_pre)

        sv = explainer.shap_values(X_pre)
        if isinstance(sv, list):
            sv = sv[1]                 # binary — take positive class

        logger.info("  SHAP: real values computed ✓")
        return np.abs(sv).mean(axis=0)

    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"  SHAP failed ({e}) — using permutation importance")
        return None


def _permutation_importance(
    pipeline,
    X_test: pd.DataFrame,
    y_test:  pd.Series,
    n_repeats: int = 10,
) -> np.ndarray:
    """Sklearn permutation importance as SHAP fallback."""
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=n_repeats, random_state=42, scoring="roc_auc", n_jobs=1,
    )
    return result.importances_mean


def plot_feature_importance(
    pipeline,
    X_test:       pd.DataFrame,
    y_test:        pd.Series,
    feature_names: List[str],
    top_n:         int = 15,
    method:        str = "auto",
) -> Tuple[Path, np.ndarray, List[str]]:
    """
    Plot feature importance.
    Tries SHAP first; falls back to permutation importance.
    Returns: (plot_path, importance_values, feature_names_ordered)
    """
    n_feats = min(len(feature_names), X_test.shape[1])
    feat_names = feature_names[:n_feats]
    X_sample   = X_test.iloc[:500] if len(X_test) > 500 else X_test

    importance = None
    label = "SHAP |value|"

    if method in ("auto", "shap"):
        importance = _shap_importance(pipeline, X_sample, feat_names)

    if importance is None:
        logger.info("  Using permutation importance (install shap for SHAP values)")
        importance = _permutation_importance(pipeline, X_sample, y_test.iloc[:len(X_sample)])
        label = "Permutation Importance (ROC-AUC drop)"

    # Align lengths
    n = min(len(importance), len(feat_names))
    importance  = importance[:n]
    feat_names  = feat_names[:n]

    # Sort top-N
    idx_sorted  = np.argsort(importance)[-top_n:]
    imp_sorted  = importance[idx_sorted]
    name_sorted = [feat_names[i] for i in idx_sorted]
    colors      = [BLUE if v > imp_sorted.mean() else "#4a5070" for v in imp_sorted]

    fig, ax = _dark_fig((10, 7))
    bars = ax.barh(name_sorted, imp_sorted, color=colors, edgecolor="none", height=0.65)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + imp_sorted.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", fontsize=8, color=MUTED)

    ax.set_xlabel(label, fontsize=11, color=TEXT)
    ax.set_title("Feature Importance — Adverse Event Risk Prediction",
                 fontsize=13, fontweight="bold", color=TEXT)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values(): s.set_edgecolor(GRID)

    path = REPORTS_DIR / "shap_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"  Feature importance → {path}")

    return path, importance, feat_names


def plot_model_comparison(all_results: Dict) -> Path:
    """Bar chart comparing all models by CV-AUC and Test-AUC."""
    names    = list(all_results.keys())
    cv_aucs  = [all_results[n]["cv_auc"]   for n in names]
    te_aucs  = [all_results[n]["test_auc"]  for n in names]
    x        = np.arange(len(names))
    w        = 0.35

    fig, ax = _dark_fig((10, 5))
    bars1 = ax.bar(x - w/2, cv_aucs, w, color=BLUE,   label="CV AUC",   alpha=0.85)
    bars2 = ax.bar(x + w/2, te_aucs, w, color=PURPLE, label="Test AUC", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9, color=TEXT)
    ax.set_ylabel("ROC-AUC", color=TEXT, fontsize=11)
    ax.set_title("Model Comparison — All Classifiers", color=TEXT, fontsize=13, fontweight="bold")
    ax.set_ylim([max(0, min(cv_aucs + te_aucs) - 0.05), 1.0])
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)

    # Annotate bars
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=MUTED)

    path = REPORTS_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"  Model comparison → {path}")
    return path


# ── Patient-level explanation ─────────────────────────────────────────────────
def explain_patient(
    pipeline,
    patient_df:    pd.DataFrame,
    feature_names: List[str],
    top_n:         int = 8,
) -> Dict:
    """
    Generate a feature-contribution explanation for a single patient.
    Uses SHAP if available, else uses coefficient / feature importance proxy.
    """
    X_sample   = patient_df.iloc[:1]
    risk_score = float(pipeline.predict_proba(X_sample)[0, 1])

    # Try SHAP
    try:
        import shap
        preproc = Pipeline(list(pipeline.steps[:-1]))
        clf     = pipeline[-1]
        X_pre   = preproc.transform(X_sample)
        model_type = type(clf).__name__.lower()

        if any(t in model_type for t in ["xgb", "lgbm", "lightgbm", "forest", "tree", "boosting"]):
            explainer = shap.TreeExplainer(clf)
        else:
            bg = preproc.transform(patient_df)
            explainer = shap.LinearExplainer(clf, bg)

        sv = explainer.shap_values(X_pre)
        if isinstance(sv, list): sv = sv[1]
        sv = sv[0]

        n = min(len(sv), len(feature_names))
        order = np.argsort(np.abs(sv[:n]))[::-1][:top_n]
        contributions = [
            {
                "feature":    feature_names[i],
                "shap_value": float(sv[i]),
                "direction":  "↑ increases risk" if sv[i] > 0 else "↓ decreases risk",
                "magnitude":  "high" if abs(sv[i]) > 0.10 else "medium" if abs(sv[i]) > 0.02 else "low",
                "method":     "shap",
            }
            for i in order
        ]
        base_val = float(explainer.expected_value
                         if not isinstance(explainer.expected_value, (list, np.ndarray))
                         else explainer.expected_value[1])

    except (ImportError, Exception):
        # Fallback: use permutation importance proxy scaled to patient values
        preproc = Pipeline(list(pipeline.steps[:-1]))
        X_pre   = preproc.transform(X_sample)
        clf     = pipeline[-1]

        # Get feature importances from the classifier if available
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            imp = np.ones(X_pre.shape[1])

        n = min(len(imp), len(feature_names))
        order = np.argsort(imp[:n])[::-1][:top_n]
        contributions = [
            {
                "feature":    feature_names[i],
                "shap_value": float(imp[i] * (X_pre[0, i] if i < X_pre.shape[1] else 0)),
                "direction":  "↑ increases risk" if imp[i] > 0 else "↓ decreases risk",
                "magnitude":  "high" if imp[i] > np.mean(imp) else "low",
                "method":     "importance_proxy",
            }
            for i in order
        ]
        base_val = 0.0

    return {
        "risk_score":    risk_score,
        "base_value":    base_val,
        "contributions": contributions,
    }


# ── Full evaluation run ───────────────────────────────────────────────────────
def run_full_evaluation(
    data_source: str          = "synthetic",
    model_path:  Optional[Path] = None,
) -> Dict:
    """
    Load trained model, evaluate on test set, generate all plots and reports.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader         import load_data, FEATURE_COLS, TARGET_COL
    from feature_engineering  import engineer_features

    model_path = model_path or MODELS_DIR / "ae_model.pkl"
    meta_path  = MODELS_DIR / "model_metadata.json"
    comp_path  = MODELS_DIR / "all_model_results.json"

    if not model_path.exists():
        logger.warning("Model not found — running training first...")
        from train_model import main as train_main
        train_main(data_source=data_source)

    # Load model and metadata
    pipeline  = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    model_name = metadata["best_model_name"]
    threshold  = metadata["optimal_threshold"]
    feat_names = metadata["feature_names"]

    logger.info(f"Loaded: {model_name} | threshold={threshold:.3f}")

    # Load data and reproduce test split (same seed)
    df = load_data(source=data_source)
    X, _ = engineer_features(df, feature_cols=FEATURE_COLS, add_interactions=True)
    y    = df[TARGET_COL].astype(int)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = compute_metrics(y_test.values, y_prob, threshold=threshold)

    logger.info(
        f"\nEvaluation Results:\n"
        f"  ROC-AUC:      {metrics['roc_auc']:.4f}\n"
        f"  Avg Precision:{metrics['avg_precision']:.4f}\n"
        f"  Sensitivity:  {metrics['sensitivity']:.4f}\n"
        f"  Specificity:  {metrics['specificity']:.4f}\n"
        f"  PPV:          {metrics['ppv']:.4f}\n"
        f"  NPV:          {metrics['npv']:.4f}\n"
        f"  TP:{metrics['tp']} TN:{metrics['tn']} FP:{metrics['fp']} FN:{metrics['fn']}"
    )

    # Load comparison data for multi-model ROC
    all_results = None
    if comp_path.exists():
        with open(comp_path) as f:
            all_results = json.load(f)

    # Generate all plots
    logger.info("Generating plots...")
    plot_roc_curve(y_test.values, y_prob, all_results=all_results, model_name=model_name)
    plot_precision_recall(y_test.values, y_prob)
    plot_calibration(y_test.values, y_prob)
    plot_confusion_matrix(metrics)

    # Feature importance
    logger.info("Computing feature importance...")
    plot_feature_importance(pipeline, X_test, y_test, feat_names, top_n=15)

    # Model comparison
    if all_results:
        plot_model_comparison(all_results)

    # Save evaluation report
    report = {
        "model_name": model_name,
        "metrics":    metrics,
        "metadata":   metadata,
    }
    report_path = REPORTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"  Evaluation report → {report_path}")

    return report


if __name__ == "__main__":
    report = run_full_evaluation(data_source="synthetic")
    m = report["metrics"]
    print(f"\nROC-AUC:     {m['roc_auc']:.4f}")
    print(f"Sensitivity: {m['sensitivity']:.4f}")
    print(f"Specificity: {m['specificity']:.4f}")
    print(f"PPV:         {m['ppv']:.4f}")
    print(f"NPV:         {m['npv']:.4f}")
    print(f"\nPlots saved to: {REPORTS_DIR}")
