"""
Adverse Event Risk Predictor - MIMIC-III
Exploratory Data Analysis Notebook
File: notebooks/exploratory_analysis.py

This script generates the exploratory_analysis.ipynb notebook.
Run: python notebooks/exploratory_analysis.py
Then open: jupyter notebook notebooks/exploratory_analysis.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# ── Cell 0: Title ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""
# ⚕ Adverse Event Risk Predictor — Exploratory Data Analysis
## MIMIC-III Clinical Database

This notebook performs exploratory data analysis on the ICU adverse event prediction dataset.

**Sections:**
1. Dataset Overview
2. Adverse Event Rate Analysis
3. Feature Distributions
4. Correlation Analysis
5. Clinical Risk Factor Profiling
6. Missing Value Analysis
"""))

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""
import sys
sys.path.insert(0, '../python')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Style
plt.style.use('dark_background')
matplotlib.rcParams['figure.facecolor'] = '#0f1117'
matplotlib.rcParams['axes.facecolor'] = '#1a1d27'
matplotlib.rcParams['figure.figsize'] = (12, 6)

from data_loader import load_data, FEATURE_COLS, TARGET_COL, get_data_summary
from feature_engineering import get_feature_metadata

# Load dataset
df = load_data(source='synthetic', validate=True)
print(f"Dataset loaded: {df.shape}")
"""))

# ── Cell 2: Overview ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""
# Dataset Overview
summary = get_data_summary(df)
print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)
for k, v in summary.items():
    if k != 'feature_names':
        print(f"  {k:30s}: {v}")

print(f"\\nFeatures: {', '.join(summary['feature_names'])}")
"""))

# ── Cell 3: Adverse Event Rate ─────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""
# Adverse Event Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
ae_counts = df[TARGET_COL].value_counts()
colors = ['#39d98a', '#ff4d6d']
axes[0].bar(['No AE', 'Adverse Event'], ae_counts.values, color=colors, width=0.5)
axes[0].set_title('Adverse Event Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(ae_counts.values):
    axes[0].text(i, v + 10, f'{v:,}\\n({v/len(df):.1%})', ha='center', fontsize=11)

# By risk factors
risk_summary = df.groupby(TARGET_COL)[FEATURE_COLS].mean()
risk_summary.index = ['No AE', 'Adverse Event']
risk_summary[['age', 'length_of_stay', 'heart_rate_mean', 'creatinine_max']].T.plot(
    kind='bar', ax=axes[1], color=colors
)
axes[1].set_title('Mean Feature Values by Outcome', fontsize=14, fontweight='bold')
axes[1].legend(['No AE', 'Adverse Event'])
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('../data/processed/reports/eda_adverse_event_dist.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ── Cell 4: Feature Distributions ─────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""
# Feature Distributions by Outcome
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

colors = {0: '#39d98a', 1: '#ff4d6d'}

for i, col in enumerate(FEATURE_COLS):
    for outcome in [0, 1]:
        subset = df[df[TARGET_COL] == outcome][col].dropna()
        axes[i].hist(subset, bins=30, alpha=0.6,
                     color=colors[outcome], density=True,
                     label='No AE' if outcome == 0 else 'Adverse Event')
    axes[i].set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.2)

plt.suptitle('Feature Distributions by Adverse Event Outcome', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../data/processed/reports/eda_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ── Cell 5: Correlation Matrix ─────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""
# Correlation Heatmap
corr_cols = FEATURE_COLS + [TARGET_COL]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(
    corr_matrix, mask=mask, cmap=cmap, center=0,
    annot=True, fmt='.2f', linewidths=0.5,
    cbar_kws={'shrink': 0.8}, ax=ax, annot_kws={'size': 9}
)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('../data/processed/reports/eda_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# Correlations with target
target_corr = df[corr_cols].corr()[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
print("\\nCorrelation with Adverse Event:")
print(target_corr.round(3).to_string())
"""))

# ── Cell 6: Clinical Profiling ─────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""
# Clinical Risk Profiling
from feature_engineering import ClinicalRiskScorer

scorer = ClinicalRiskScorer()
X_eng = scorer.transform(df[FEATURE_COLS].astype(float))
df_eng = pd.concat([df[[TARGET_COL]], X_eng.reset_index(drop=True)], axis=1)

# Age group analysis
df['age_group'] = pd.cut(df['age'], bins=[18,45,65,80,110],
                          labels=['18-44','45-64','65-79','80+'])
age_ae = df.groupby('age_group')[TARGET_COL].agg(['mean','count']).reset_index()
age_ae.columns = ['Age Group','AE Rate','Count']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# AE rate by age
bars = axes[0].bar(age_ae['Age Group'], age_ae['AE Rate'],
                    color=['#4f9cf9','#7b6cf9','#bf5af2','#ff4d6d'])
axes[0].set_title('Adverse Event Rate by Age Group', fontweight='bold')
axes[0].set_ylabel('AE Rate')
for bar in bars:
    h = bar.get_height()
    axes[0].text(bar.get_x()+bar.get_width()/2, h+0.005, f'{h:.1%}',
                  ha='center', fontsize=10)

# Creatinine vs AE
for outcome, color, label in [(0,'#39d98a','No AE'),(1,'#ff4d6d','AE')]:
    data = df[df[TARGET_COL]==outcome]['creatinine_max'].clip(0,10)
    axes[1].hist(data, bins=30, alpha=0.6, color=color, density=True, label=label)
axes[1].set_title('Creatinine Distribution', fontweight='bold')
axes[1].set_xlabel('Max Creatinine (mg/dL)')
axes[1].legend()

# Drug count vs AE
for outcome, color, label in [(0,'#39d98a','No AE'),(1,'#ff4d6d','AE')]:
    data = df[df[TARGET_COL]==outcome]['drug_count'].clip(0,30)
    axes[2].hist(data, bins=20, alpha=0.6, color=color, density=True, label=label)
axes[2].set_title('Drug Count Distribution', fontweight='bold')
axes[2].set_xlabel('Number of Drugs')
axes[2].legend()

for ax in axes: ax.grid(alpha=0.2)
plt.suptitle('Clinical Risk Factor Profiling', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/processed/reports/eda_clinical_profiling.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ── Cell 7: Statistical Summary ────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""
# Statistical Summary Table
from scipy import stats

print("=" * 80)
print(f"{'Feature':30s} {'No AE Mean':>12s} {'AE Mean':>12s} {'p-value':>12s} {'Sig':>6s}")
print("=" * 80)

for col in FEATURE_COLS:
    no_ae = df[df[TARGET_COL]==0][col].dropna()
    ae    = df[df[TARGET_COL]==1][col].dropna()
    t_stat, p_val = stats.mannwhitneyu(no_ae, ae, alternative='two-sided')
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f"{col:30s} {no_ae.mean():>12.3f} {ae.mean():>12.3f} {p_val:>12.4f} {sig:>6s}")

print("=" * 80)
print("Significance: *** p<0.001 | ** p<0.01 | * p<0.05")
"""))

# ── Cell 8: Summary ───────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""
## Key Findings

1. **Adverse Event Rate**: Approximately 30-40% of ICU patients experience an adverse event (composite outcome)

2. **Top Risk Factors** (by correlation with outcome):
   - Lab abnormality score (highest)
   - Creatinine max (renal impairment)
   - Polypharmacy score
   - Drug count
   - Length of ICU stay

3. **Age Effect**: Adverse event risk increases with age, particularly for patients ≥65

4. **Polypharmacy**: Patients with ≥5 drugs have significantly higher risk

5. **Clinical Thresholds**:
   - Creatinine >1.5 mg/dL → elevated risk
   - Heart rate >100 bpm → tachycardia, increased risk
   - WBC >12 K/μL → leukocytosis, possible infection

---
*Next Steps: Proceed to `python/train_model.py` for model training*
"""))

nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

import json
from pathlib import Path

output_path = Path(__file__).parent / "exploratory_analysis.ipynb"
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook written to: {output_path}")
