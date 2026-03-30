"""
Adverse Event Risk Predictor — MIMIC-III
python/data_loader.py

Clinical data ingestion: PostgreSQL / CSV / Synthetic.
Dependencies: numpy, pandas, sklearn (no external packages needed).
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ae.data")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"

FEATURE_COLS = [
    "age", "gender", "length_of_stay", "heart_rate_mean",
    "creatinine_max", "wbc_mean", "drug_count",
    "polypharmacy_score", "lab_abnormality_score",
]
TARGET_COL = "adverse_event"
ID_COLS    = ["subject_id", "hadm_id", "icustay_id"]

CLINICAL_RANGES = {
    "age":                   (18,  110),
    "gender":                (0,   1),
    "length_of_stay":        (0.1, 365),
    "heart_rate_mean":       (20,  300),
    "creatinine_max":        (0.1, 30),
    "wbc_mean":              (0.1, 200),
    "drug_count":            (0,   200),
    "polypharmacy_score":    (0,   10),
    "lab_abnormality_score": (0,   20),
}


def generate_synthetic_mimic(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate clinically realistic synthetic MIMIC-III data.

    The adverse event probability is a logistic function of known ICU
    risk factors, yielding ~22% event rate with realistic class separation.
    """
    rng = np.random.default_rng(seed)
    logger.info(f"Generating {n:,} synthetic patients (seed={seed})")

    # Demographics
    age    = rng.normal(65, 17, n).clip(18, 105)
    gender = rng.binomial(1, 0.55, n).astype(float)

    # ICU Stay
    los = rng.lognormal(1.4, 0.9, n).clip(0.1, 90)

    # Vitals
    heart_rate = rng.normal(88, 22, n).clip(30, 250)
    sbp        = rng.normal(118, 22, n).clip(70, 200)
    spo2       = rng.normal(97, 3, n).clip(60, 100)
    resp_rate  = rng.normal(18, 5, n).clip(8, 40)

    # Labs
    creatinine = rng.lognormal(0.3, 0.8, n).clip(0.1, 20)
    wbc        = rng.lognormal(2.2, 0.5, n).clip(0.5, 100)
    sodium     = rng.normal(139, 4, n).clip(125, 155)
    potassium  = rng.normal(4.1, 0.5, n).clip(2.5, 6.5)
    hemoglobin = rng.normal(10.5, 2.5, n).clip(5, 18)
    platelets  = rng.lognormal(5.0, 0.5, n).clip(10, 800)

    # Medications
    drug_ct   = rng.poisson(8, n).clip(0, 50).astype(float)
    poly_score = (drug_ct / 2).clip(0, 10)

    # Lab abnormality score
    lab_abn = (
        (creatinine > 1.2).astype(float) * 2 +
        (wbc > 11).astype(float) +
        (wbc < 4).astype(float) +
        ((sodium < 135) | (sodium > 145)).astype(float) +
        ((potassium < 3.5) | (potassium > 5.0)).astype(float) +
        (hemoglobin < 8).astype(float) * 2 +
        (platelets < 100).astype(float)
    ) + rng.poisson(0.3, n)
    lab_abn = lab_abn.clip(0, 20)

    # Adverse event via logistic model
    log_odds = (
        -3.8
        + 0.020 * (age - 65)
        - 0.030 * gender
        + 0.040 * los
        + 0.010 * (heart_rate - 88)
        + 0.220 * np.log1p(creatinine)
        + 0.080 * np.log1p(wbc)
        + 0.050 * drug_ct
        + 0.150 * lab_abn
        - 0.015 * (spo2 - 95)
        - 0.003 * (sbp - 120)
        + rng.normal(0, 0.4, n)
    )
    prob_ae       = 1 / (1 + np.exp(-log_odds))
    adverse_event = rng.binomial(1, prob_ae, n).astype(float)

    logger.info(f"AE rate={adverse_event.mean():.1%} | Age={age.mean():.0f} | LOS median={np.median(los):.1f}d")

    return pd.DataFrame({
        "subject_id":            np.arange(10000, 10000 + n),
        "hadm_id":               np.arange(200000, 200000 + n),
        "icustay_id":            np.arange(300000, 300000 + n),
        "age":                   age.round(1),
        "gender":                gender,
        "length_of_stay":        los.round(2),
        "heart_rate_mean":       heart_rate.round(1),
        "heart_rate_min":        (heart_rate - rng.uniform(5, 20, n)).clip(20, 280).round(1),
        "heart_rate_max":        (heart_rate + rng.uniform(5, 30, n)).clip(20, 300).round(1),
        "heart_rate_std":        rng.uniform(5, 25, n).round(2),
        "creatinine_max":        creatinine.round(3),
        "creatinine_mean":       (creatinine * 0.8).round(3),
        "wbc_mean":              wbc.round(2),
        "wbc_max":               (wbc * 1.2).round(2),
        "sodium_mean":           sodium.round(1),
        "potassium_mean":        potassium.round(2),
        "hemoglobin_min":        hemoglobin.round(1),
        "platelet_min":          platelets.round(0),
        "sbp_mean":              sbp.round(1),
        "spo2_min":              spo2.round(1),
        "resp_rate_mean":        resp_rate.round(1),
        "drug_count":            drug_ct.astype(int),
        "polypharmacy_score":    poly_score.round(2),
        "polypharmacy_flag":     (drug_ct >= 5).astype(float),
        "lab_abnormality_score": lab_abn.round(1),
        "adverse_event":         adverse_event,
    })


def load_data(
    source:   str            = "synthetic",
    path:     Optional[Path] = None,
    validate: bool           = True,
    n:        int            = 5000,
    seed:     int            = 42,
) -> pd.DataFrame:
    """
    Unified data loader.
    source: 'synthetic' | 'csv' | 'db'
    """
    if source == "synthetic":
        df = generate_synthetic_mimic(n=n, seed=seed)

    elif source == "csv":
        p = path or DATA_PROC / "ae_features_clean.csv"
        if not p.exists():
            logger.warning(f"CSV not found at {p}. Using synthetic data.")
            df = generate_synthetic_mimic(n=n, seed=seed)
        else:
            logger.info(f"Loading CSV: {p}")
            df = pd.read_csv(p, low_memory=False)

    elif source == "db":
        try:
            from sqlalchemy import create_engine
            url = os.environ.get("MIMIC_DB_URL", "postgresql://postgres:postgres@localhost:5432/mimic")
            engine = create_engine(url)
            df = pd.read_sql("SELECT * FROM ae_model_features", engine)
            logger.info(f"Loaded {len(df):,} rows from database")
        except Exception as e:
            logger.error(f"DB load failed: {e}. Using synthetic data.")
            df = generate_synthetic_mimic(n=n, seed=seed)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    # Coerce to numeric
    for col in FEATURE_COLS + [TARGET_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop missing target
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    if validate:
        _validate(df)

    logger.info(f"Dataset ready: {len(df):,} patients | AE rate: {df[TARGET_COL].mean():.1%}")
    return df


def _validate(df: pd.DataFrame):
    """Basic clinical range validation with warnings."""
    for col, (lo, hi) in CLINICAL_RANGES.items():
        if col in df.columns:
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_out > 0:
                logger.warning(f"  {col}: {n_out} values outside [{lo},{hi}]")


def split_features_target(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    cols = feature_cols or FEATURE_COLS
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy().astype(float), df[TARGET_COL].copy().astype(int)


def get_data_summary(df: pd.DataFrame) -> Dict:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return {
        "n_patients":         int(len(df)),
        "n_features":         len(cols),
        "adverse_event_n":    int(df[TARGET_COL].sum()),
        "adverse_event_rate": float(df[TARGET_COL].mean()),
        "age_mean":           float(df["age"].mean()) if "age" in df.columns else 0,
        "age_std":            float(df["age"].std())  if "age" in df.columns else 0,
        "los_median":         float(df["length_of_stay"].median()) if "length_of_stay" in df.columns else 0,
        "missing_pct":        float(df[cols].isnull().mean().mean()),
        "feature_names":      cols,
    }


if __name__ == "__main__":
    df = load_data(source="synthetic", n=5000)
    s  = get_data_summary(df)
    for k, v in s.items():
        if k != "feature_names":
            print(f"  {k:25s}: {v}")
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROC / "ae_dataset.csv", index=False)
    print(f"\nSaved → {DATA_PROC / 'ae_dataset.csv'}")
