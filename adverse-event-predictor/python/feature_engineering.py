"""
Adverse Event Risk Predictor — MIMIC-III
python/feature_engineering.py

Feature engineering: clinical interaction terms, composite risk scores,
log-transforms, binary flags, sklearn-compatible preprocessing pipeline.
Dependencies: numpy, pandas, sklearn.
"""

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer

warnings.filterwarnings("ignore")
logger = logging.getLogger("ae.features")

# ── Column groups ─────────────────────────────────────────────────────────────
CONTINUOUS_COLS = [
    "age", "length_of_stay", "heart_rate_mean",
    "creatinine_max", "wbc_mean", "polypharmacy_score", "lab_abnormality_score",
]
BINARY_COLS = ["gender"]
COUNT_COLS  = ["drug_count"]
ALL_BASE    = CONTINUOUS_COLS + BINARY_COLS + COUNT_COLS   # 9 features


# ── Clinical Risk Scorer (custom transformer) ─────────────────────────────────
class ClinicalRiskScorer(BaseEstimator, TransformerMixin):
    """
    Derives 17 additional clinical features from base features.

    Adds:
      - Binary clinical flags (tachycardia, renal impairment, etc.)
      - Interaction terms (age × creatinine, HR × LOS)
      - Log-transforms for skewed variables
      - Composite clinical risk index
    """

    DERIVED_FEATURES = [
        "tachycardia_flag",
        "renal_impairment_flag",
        "leukocytosis_flag",
        "elderly_flag",
        "long_stay_flag",
        "high_polypharmacy_flag",
        "critical_labs_flag",
        "organ_failure_score",
        "composite_clinical_risk",
        "age_creatinine_interaction",
        "hr_los_interaction",
        "drug_lab_interaction",
        "poly_creatinine_interaction",
        "log_creatinine",
        "log_los",
        "log_wbc",
        "log_drugs",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self._input_cols if hasattr(self, "_input_cols") else
                             [f"f{i}" for i in range(X.shape[1])])
        else:
            X = pd.DataFrame(X).copy()
            self._input_cols = list(X.columns)

        def col(name, default=0.0):
            return X[name].fillna(default).values if name in X.columns else np.full(len(X), default)

        age   = col("age",                  65.0)
        hr    = col("heart_rate_mean",       80.0)
        crea  = col("creatinine_max",         1.0)
        wbc   = col("wbc_mean",               9.0)
        los   = col("length_of_stay",         2.0)
        drugs = col("drug_count",             5.0)
        lab   = col("lab_abnormality_score",  0.0)
        poly  = col("polypharmacy_score",     2.5)

        # ── Binary flags ──────────────────────────────────────────────────────
        tachy    = (hr    > 100).astype(float)
        renal    = (crea  > 1.5).astype(float)
        leuko    = (wbc   > 12).astype(float)
        elderly  = (age   >= 65).astype(float)
        longstay = (los   >= 7).astype(float)
        hipoly   = (poly  >= 7).astype(float)
        critlab  = (lab   >= 3).astype(float)

        # ── Composite scores ──────────────────────────────────────────────────
        organ_fail = renal + critlab + leuko
        composite  = (
            0.20 * tachy +
            0.25 * renal +
            0.20 * leuko +
            0.15 * (lab >= 2).astype(float) +
            0.10 * hipoly +
            0.10 * longstay
        ).clip(0, 1)

        # ── Interactions ──────────────────────────────────────────────────────
        age_crea  = (age / 65.0) * (crea / 1.2)
        hr_los    = (hr  / 80.0) * np.log1p(los)
        drug_lab  = drugs * (lab + 1)
        poly_crea = poly  * crea

        # ── Log transforms ────────────────────────────────────────────────────
        log_crea  = np.log1p(crea)
        log_los   = np.log1p(los)
        log_wbc   = np.log1p(wbc)
        log_drug  = np.log1p(drugs)

        derived = pd.DataFrame({
            "tachycardia_flag":          tachy,
            "renal_impairment_flag":     renal,
            "leukocytosis_flag":         leuko,
            "elderly_flag":              elderly,
            "long_stay_flag":            longstay,
            "high_polypharmacy_flag":    hipoly,
            "critical_labs_flag":        critlab,
            "organ_failure_score":       organ_fail,
            "composite_clinical_risk":   composite,
            "age_creatinine_interaction":age_crea,
            "hr_los_interaction":        hr_los,
            "drug_lab_interaction":      drug_lab,
            "poly_creatinine_interaction":poly_crea,
            "log_creatinine":            log_crea,
            "log_los":                   log_los,
            "log_wbc":                   log_wbc,
            "log_drugs":                 log_drug,
        }, index=X.index)

        result = pd.concat([X.reset_index(drop=True), derived.reset_index(drop=True)], axis=1)
        return result

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return base + self.DERIVED_FEATURES


# ── Outlier Clipper ───────────────────────────────────────────────────────────
class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clips numeric features to [lower_pct, upper_pct] percentile bounds."""

    def __init__(self, lower_pct: float = 0.005, upper_pct: float = 0.995):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = np.array(X, dtype=float)
        self.lower_ = np.nanpercentile(X, self.lower_pct * 100, axis=0)
        self.upper_ = np.nanpercentile(X, self.upper_pct * 100, axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float).copy()
        for i in range(X.shape[1]):
            X[:, i] = np.clip(X[:, i], self.lower_[i], self.upper_[i])
        return X


# ── Preprocessing Pipeline Builder ───────────────────────────────────────────
def build_preprocessing_pipeline(
    scaler_type:          str  = "robust",
    imputer_strategy:     str  = "median",
    use_clinical_features: bool = True,
) -> Pipeline:
    """
    Build a full sklearn preprocessing pipeline.

    Steps:
      1. (optional) ClinicalRiskScorer — adds 17 derived features
      2. SimpleImputer  — median/mean imputation
      3. OutlierClipper — winsorize to 0.5th/99.5th percentile
      4. Scaler         — RobustScaler | StandardScaler | PowerTransformer

    Returns: sklearn Pipeline
    """
    steps = []

    if use_clinical_features:
        steps.append(("clinical_scorer", ClinicalRiskScorer()))

    steps.append(("imputer", SimpleImputer(strategy=imputer_strategy)))
    steps.append(("clipper", OutlierClipper()))

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "power":
        scaler = PowerTransformer(method="yeo-johnson")
    else:
        scaler = RobustScaler()
    steps.append(("scaler", scaler))

    return Pipeline(steps)


# ── Feature engineering entry point ──────────────────────────────────────────
def engineer_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    add_interactions: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply feature engineering to a raw DataFrame.

    Returns:
        (X_engineered, feature_names_list)
    """
    from data_loader import FEATURE_COLS
    feature_cols = feature_cols or FEATURE_COLS

    # Fill any missing base feature columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            logger.warning(f"Column {col!r} not found — filled with 0")

    X = df[feature_cols].copy().astype(float)

    if add_interactions:
        scorer = ClinicalRiskScorer()
        X_out  = scorer.transform(X)
        feat_names = list(X_out.columns)
        logger.info(f"Feature engineering: {len(feature_cols)} → {len(feat_names)} features")
        return X_out, feat_names

    return X, feature_cols


def get_feature_metadata() -> pd.DataFrame:
    """Return a table describing base features and their clinical context."""
    records = [
        ("age",                   "continuous", "Demographics",  "Patient age (years)",                       18,  110),
        ("gender",                "binary",     "Demographics",  "Sex: 1=Male, 0=Female",                      0,    1),
        ("length_of_stay",        "continuous", "ICU",           "ICU length of stay (days)",                 0.1,  365),
        ("heart_rate_mean",       "continuous", "Vitals",        "Mean heart rate (bpm)",                     20,  300),
        ("creatinine_max",        "continuous", "Labs",          "Maximum serum creatinine (mg/dL)",          0.1,  30),
        ("wbc_mean",              "continuous", "Labs",          "Mean WBC count (K/µL)",                     0.1, 200),
        ("drug_count",            "count",      "Medications",   "Distinct drugs prescribed",                   0, 200),
        ("polypharmacy_score",    "continuous", "Medications",   "Polypharmacy burden score (0–10)",            0,  10),
        ("lab_abnormality_score", "count",      "Labs",          "Count of critical lab abnormalities",         0,  20),
    ]
    return pd.DataFrame(records,
                        columns=["name", "type", "category", "description", "min_val", "max_val"])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__.replace("feature_engineering.py", "")))
    from data_loader import load_data, FEATURE_COLS

    df = load_data(source="synthetic", n=500)
    X, names = engineer_features(df, feature_cols=FEATURE_COLS, add_interactions=True)
    print(f"Input:  {len(FEATURE_COLS)} features")
    print(f"Output: {len(names)} features")
    print(f"Shape:  {X.shape}")
    print(f"New features: {[n for n in names if n not in FEATURE_COLS]}")
