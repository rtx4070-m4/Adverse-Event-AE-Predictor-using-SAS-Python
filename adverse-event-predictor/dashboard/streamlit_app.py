"""
Adverse Event Risk Predictor - MIMIC-III
Streamlit Clinical Dashboard
File: dashboard/streamlit_app.py

Clinical ICU Risk Prediction Dashboard with:
- Patient risk scoring
- SHAP feature contributions
- Model performance overview
- Patient risk distribution
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ICU Adverse Event Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0f1117;
}

.main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border: 1px solid #1e3a4a;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
    text-align: center;
}

.main-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    color: #4f9cf9;
    font-size: 2.2rem;
    font-weight: 600;
    letter-spacing: -1px;
    margin: 0;
}

.main-header p {
    color: #8890a8;
    font-size: 0.95rem;
    margin: 0.5rem 0 0 0;
}

.risk-card {
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid;
    margin-bottom: 1rem;
}

.risk-low      { background: #0d2b1e; border-color: #1a5c3a; }
.risk-moderate { background: #2b2200; border-color: #5c4a00; }
.risk-high     { background: #2b0e00; border-color: #7a2000; }
.risk-critical { background: #1a0014; border-color: #6b0033; }

.risk-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.5rem;
    font-weight: 600;
    line-height: 1;
    margin: 0.5rem 0;
}

.risk-label {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0;
}

.clinical-msg {
    background: #131722;
    border-left: 3px solid #4f9cf9;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #c8d0e8;
}

.metric-card {
    background: #131722;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}

.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #4f9cf9;
}

.metric-label {
    font-size: 0.75rem;
    color: #8890a8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.shap-bar-pos { background: #ff4d6d; }
.shap-bar-neg { background: #39d98a; }

.feature-row {
    display: flex;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1e2535;
    gap: 1rem;
}

.feature-name { color: #c8d0e8; font-size: 0.85rem; min-width: 200px; }
.feature-val  { color: #8890a8; font-size: 0.8rem; font-family: monospace; min-width: 80px; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    color: #4f9cf9;
    border-bottom: 1px solid #1e2535;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton button {
    background: linear-gradient(135deg, #1a4a8a, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: all 0.2s;
}

.stButton button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    transform: translateY(-1px);
}

div[data-testid="stSidebarContent"] {
    background: #0a0e1a;
    border-right: 1px solid #1e2535;
}

.stSlider [data-testid="stSlider"] label { color: #8890a8; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────────────────────
def get_risk_color(category: str) -> str:
    return {
        "Low":      "#39d98a",
        "Moderate": "#ffcc00",
        "High":     "#ff7a1a",
        "Critical": "#ff4d6d",
    }.get(category, "#8890a8")


def call_api(endpoint: str, payload: dict) -> Optional[dict]:
    """Call the FastAPI backend."""
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Cannot connect to API. Make sure the FastAPI server is running:\n"
            "`cd api && uvicorn app:app --reload`"
        )
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def get_model_info() -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}/model-info", timeout=10)
        return r.json()
    except:
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 ICU Risk Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Patient Assessment", "Model Performance", "Batch Analysis", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # API Status
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success("✓ API Online")
        st.caption(f"Predictions served: {health.get('predictions_served', 0):,}")
    except:
        st.error("✗ API Offline")

    st.markdown("---")
    st.caption("MIMIC-III Clinical Database\nJohn Hopkins / MIT")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚕ ICU Adverse Event Risk Predictor</h1>
    <p>Machine Learning–powered clinical decision support for ICU patient risk stratification</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PATIENT ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
if page == "Patient Assessment":
    col_input, col_result = st.columns([1, 1.3], gap="large")

    with col_input:
        st.markdown('<div class="section-header">Patient Clinical Data</div>',
                    unsafe_allow_html=True)

        patient_id = st.text_input("Patient ID (optional)", placeholder="e.g., ICU-2024-001")

        st.markdown("**Demographics**")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=18, max_value=110, value=68, step=1)
        with c2:
            gender = st.selectbox("Sex", ["Male (1)", "Female (0)"])
            gender_val = 1.0 if "Male" in gender else 0.0

        st.markdown("**ICU Stay**")
        los = st.slider("Length of Stay (days)", 0.1, 30.0, 4.5, 0.1,
                        format="%.1f days")

        st.markdown("**Vital Signs**")
        hr = st.slider("Mean Heart Rate (bpm)", 30, 200, 88, 1)

        st.markdown("**Laboratory Values**")
        c3, c4 = st.columns(2)
        with c3:
            creatinine = st.number_input("Max Creatinine (mg/dL)", 0.1, 25.0, 1.2, 0.1,
                                          format="%.1f")
        with c4:
            wbc = st.number_input("Mean WBC (K/μL)", 0.5, 80.0, 9.5, 0.5,
                                   format="%.1f")

        lab_abn = st.slider("Lab Abnormality Count", 0, 10, 1, 1)

        st.markdown("**Medications**")
        c5, c6 = st.columns(2)
        with c5:
            drug_count = st.number_input("Drug Count", 0, 60, 7, 1)
        with c6:
            poly_score = st.slider("Polypharmacy Score", 0.0, 10.0, 3.5, 0.5)

        predict_btn = st.button("⚡ ASSESS RISK", use_container_width=True)
        explain_btn = st.button("🔬 EXPLAIN PREDICTION", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">Risk Assessment</div>',
                    unsafe_allow_html=True)

        payload = {
            "patient_id":           patient_id or None,
            "age":                  float(age),
            "gender":               gender_val,
            "length_of_stay":       float(los),
            "heart_rate_mean":      float(hr),
            "creatinine_max":       float(creatinine),
            "wbc_mean":             float(wbc),
            "drug_count":           float(drug_count),
            "polypharmacy_score":   float(poly_score),
            "lab_abnormality_score": float(lab_abn),
        }

        if predict_btn or explain_btn:
            with st.spinner("Analyzing patient data..."):
                result = call_api("/predict", payload)

            if result:
                risk_score = result["risk_score"]
                risk_cat   = result["risk_category"]
                flag       = result["adverse_event_flag"]
                color      = get_risk_color(risk_cat)
                css_class  = f"risk-{risk_cat.lower()}"

                # Risk Score Card
                st.markdown(f"""
                <div class="risk-card {css_class}">
                    <p class="risk-label" style="color:{color}">⬡ {risk_cat} Risk</p>
                    <div class="risk-score" style="color:{color}">{risk_score:.1%}</div>
                    <p style="color:#8890a8;font-size:0.8rem;margin:0.5rem 0 0 0">
                        Adverse Event Probability
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Risk gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    number={"suffix": "%", "font": {"size": 40, "color": color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8890a8",
                                 "tickfont": {"color": "#8890a8"}},
                        "bar":  {"color": color, "thickness": 0.3},
                        "bgcolor": "#131722",
                        "bordercolor": "#1e2535",
                        "steps": [
                            {"range": [0, 20],  "color": "#0d2b1e"},
                            {"range": [20, 40], "color": "#2b2200"},
                            {"range": [40, 65], "color": "#2b0e00"},
                            {"range": [65, 100],"color": "#1a0014"},
                        ],
                        "threshold": {
                            "line": {"color": "#ffffff", "width": 2},
                            "thickness": 0.8,
                            "value": result["threshold_used"] * 100,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="#0f1117",
                    font={"color": "#e0e4f0"},
                    height=220,
                    margin=dict(l=30, r=30, t=20, b=0),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Clinical message
                st.markdown(f"""
                <div class="clinical-msg">
                    <strong>Clinical Guidance:</strong><br>
                    {result['clinical_message']}
                </div>
                """, unsafe_allow_html=True)

                # Quick metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    flag_str = "HIGH RISK" if flag else "LOW RISK"
                    flag_color = "#ff4d6d" if flag else "#39d98a"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:{flag_color};font-size:1rem">
                            {'⚠' if flag else '✓'} {flag_str}
                        </div>
                        <div class="metric-label">Classification</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result['threshold_used']:.2f}</div>
                        <div class="metric-label">Decision Threshold</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    model_name = result.get("model_version", "Unknown")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:0.8rem">{model_name}</div>
                        <div class="metric-label">Model Version</div>
                    </div>""", unsafe_allow_html=True)

        # SHAP Explanation
        if explain_btn:
            with st.spinner("Computing SHAP explanation..."):
                expl_payload = {"patient": payload, "top_n": 8}
                expl = call_api("/explain", expl_payload)

            if expl:
                st.markdown('<div class="section-header">Feature Contributions (SHAP)</div>',
                            unsafe_allow_html=True)

                contribs = expl["explanation"]["contributions"]
                if contribs:
                    features = [c["feature"].replace("_", " ").title() for c in contribs]
                    shap_vals = [c["shap_value"] for c in contribs]
                    colors    = [get_risk_color("High") if v > 0 else get_risk_color("Low")
                                 for v in shap_vals]

                    fig_shap = go.Figure(go.Bar(
                        x=shap_vals,
                        y=features,
                        orientation="h",
                        marker_color=colors,
                        marker_line_width=0,
                    ))
                    fig_shap.update_layout(
                        paper_bgcolor="#0f1117",
                        plot_bgcolor="#131722",
                        font={"color": "#e0e4f0", "size": 11},
                        xaxis={"title": "SHAP Value (Impact on Risk)",
                               "gridcolor": "#1e2535"},
                        yaxis={"gridcolor": "#1e2535"},
                        height=320,
                        margin=dict(l=10, r=20, t=10, b=40),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

                    # Interpretation
                    st.markdown(f"""
                    <div class="clinical-msg">
                        <strong>SHAP Interpretation:</strong><br>
                        {expl['explanation']['interpretation']}
                    </div>
                    """, unsafe_allow_html=True)

        if not predict_btn and not explain_btn:
            st.info("👈 Enter patient data and click **ASSESS RISK** to generate a prediction.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    info = get_model_info()

    if info:
        st.markdown('<div class="section-header">Model Performance Overview</div>',
                    unsafe_allow_html=True)

        # Performance metrics
        m1, m2, m3, m4 = st.columns(4)
        perf = info.get("performance", {})
        with m1:
            st.metric("CV ROC-AUC",    f"{perf.get('cv_roc_auc', 0):.4f}",
                      delta=f"±{perf.get('cv_roc_auc_std', 0):.4f}")
        with m2:
            st.metric("Test ROC-AUC",  f"{perf.get('test_roc_auc', 0):.4f}")
        with m3:
            st.metric("Avg Precision", f"{perf.get('avg_precision', 0):.4f}")
        with m4:
            st.metric("Model",         info.get("model_name", "N/A"))

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            # ROC curve image
            roc_path = Path("data/processed/reports/roc_curve.png")
            if roc_path.exists():
                st.image(str(roc_path), caption="ROC Curve", use_column_width=True)
            else:
                st.info("ROC curve plot not found. Run evaluate_model.py to generate.")

        with col_b:
            pr_path = Path("data/processed/reports/precision_recall.png")
            if pr_path.exists():
                st.image(str(pr_path), caption="Precision-Recall Curve", use_column_width=True)
            else:
                st.info("PR curve not found. Run evaluate_model.py to generate.")

        col_c, col_d = st.columns(2)
        with col_c:
            shap_path = Path("data/processed/reports/shap_feature_importance.png")
            if shap_path.exists():
                st.image(str(shap_path), caption="SHAP Feature Importance", use_column_width=True)

        with col_d:
            cal_path = Path("data/processed/reports/calibration.png")
            if cal_path.exists():
                st.image(str(cal_path), caption="Calibration Curve", use_column_width=True)

        # Feature list
        st.markdown('<div class="section-header">Model Features</div>', unsafe_allow_html=True)
        feat_names = info.get("feature_names", [])
        if feat_names:
            feat_df = pd.DataFrame({
                "Feature": feat_names,
                "Index":   range(1, len(feat_names) + 1),
            })
            st.dataframe(feat_df.set_index("Index"), use_container_width=True, height=300)

    else:
        st.warning("Cannot load model info. Ensure the API is running.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Batch Analysis":
    st.markdown('<div class="section-header">Batch Patient Risk Analysis</div>',
                unsafe_allow_html=True)

    st.info(
        "Upload a CSV with patient features to score a cohort. "
        "Required columns: age, gender, length_of_stay, heart_rate_mean, "
        "creatinine_max, wbc_mean, drug_count, polypharmacy_score, lab_abnormality_score"
    )

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])

    # Demo with synthetic patients
    if st.button("📊 Generate Demo Cohort (50 patients)"):
        rng = np.random.default_rng(99)
        n   = 50
        demo_df = pd.DataFrame({
            "age":                  rng.normal(65, 15, n).clip(18, 100).round(0),
            "gender":               rng.binomial(1, 0.55, n).astype(float),
            "length_of_stay":       rng.lognormal(1.4, 0.9, n).clip(0.2, 30).round(1),
            "heart_rate_mean":      rng.normal(88, 22, n).clip(40, 180).round(0),
            "creatinine_max":       rng.lognormal(0.3, 0.8, n).clip(0.2, 15).round(2),
            "wbc_mean":             rng.normal(11, 5, n).clip(1, 60).round(1),
            "drug_count":           rng.poisson(8, n).clip(0, 40).astype(float),
            "polypharmacy_score":   rng.uniform(0, 10, n).round(1),
            "lab_abnormality_score": rng.poisson(1.5, n).clip(0, 8).astype(float),
        })
        uploaded_df = demo_df
    elif uploaded:
        uploaded_df = pd.read_csv(uploaded)
    else:
        uploaded_df = None

    if uploaded_df is not None:
        st.write(f"**{len(uploaded_df)} patients loaded**")
        st.dataframe(uploaded_df.head(10), use_container_width=True, height=280)

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            with st.spinner(f"Scoring {len(uploaded_df)} patients..."):
                patients = []
                for _, row in uploaded_df.iterrows():
                    patients.append({
                        "age":                  float(row.get("age", 65)),
                        "gender":               float(row.get("gender", 0)),
                        "length_of_stay":       float(row.get("length_of_stay", 3)),
                        "heart_rate_mean":      float(row.get("heart_rate_mean", 88)),
                        "creatinine_max":       float(row.get("creatinine_max", 1.0)),
                        "wbc_mean":             float(row.get("wbc_mean", 9.0)),
                        "drug_count":           float(row.get("drug_count", 5)),
                        "polypharmacy_score":   float(row.get("polypharmacy_score", 2.5)),
                        "lab_abnormality_score": float(row.get("lab_abnormality_score", 0)),
                    })

                batch_result = call_api("/predict/batch", {"patients": patients})

            if batch_result:
                preds = batch_result["predictions"]

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Patients",  batch_result["n_patients"])
                with m2:
                    st.metric("High Risk",       batch_result["n_high_risk"],
                              delta=f"{batch_result['n_high_risk']/batch_result['n_patients']:.0%}")
                with m3:
                    mean_risk = np.mean([p["risk_score"] for p in preds])
                    st.metric("Mean Risk Score", f"{mean_risk:.3f}")
                with m4:
                    st.metric("Processing",
                              f"{batch_result['processing_time_ms']:.0f}ms")

                # Results table
                result_df = pd.DataFrame([{
                    "Risk Score":   p["risk_score"],
                    "Category":     p["risk_category"],
                    "High Risk":    "⚠ Yes" if p["adverse_event_flag"] else "✓ No",
                } for p in preds])

                # Distribution
                fig_dist = px.histogram(
                    result_df, x="Risk Score", color="Category",
                    color_discrete_map={
                        "Low": "#39d98a", "Moderate": "#ffcc00",
                        "High": "#ff7a1a", "Critical": "#ff4d6d"
                    },
                    title="Risk Score Distribution",
                    nbins=20,
                )
                fig_dist.update_layout(
                    paper_bgcolor="#0f1117",
                    plot_bgcolor="#131722",
                    font={"color": "#e0e4f0"},
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                # Risk category pie
                cat_counts = result_df["Category"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                fig_pie = px.pie(
                    cat_counts, names="Category", values="Count",
                    color="Category",
                    color_discrete_map={
                        "Low": "#39d98a", "Moderate": "#ffcc00",
                        "High": "#ff7a1a", "Critical": "#ff4d6d"
                    },
                    title="Risk Category Breakdown",
                )
                fig_pie.update_layout(
                    paper_bgcolor="#0f1117",
                    font={"color": "#e0e4f0"},
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Full table
                result_df.index = range(1, len(result_df) + 1)
                st.dataframe(result_df, use_container_width=True, height=350)

                # Download
                csv = result_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv,
                    file_name="ae_risk_predictions.csv",
                    mime="text/csv",
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("""
    ## About This System

    ### Clinical Purpose
    The **ICU Adverse Event Risk Predictor** is a clinical decision support tool
    designed to help clinicians identify ICU patients at high risk of experiencing
    adverse events, enabling earlier intervention and improved outcomes.

    ### Adverse Event Definition
    An adverse event is defined as **any** of:
    - In-hospital mortality
    - ICU readmission within the same hospitalization
    - Sepsis/septic shock diagnosis (ICD-9)
    - ≥3 critical laboratory abnormalities

    ### Data Source
    Built on the [MIMIC-III](https://physionet.org/content/mimiciii/) Clinical Database
    — a large, freely available database comprising de-identified health-related data
    of ICU patients (2001–2012), developed by MIT and Beth Israel Deaconess Medical Center.

    ### Model Pipeline
    ```
    MIMIC-III PostgreSQL
         ↓
    SQL Feature Extraction
         ↓
    SAS Statistical Cleaning
         ↓
    Python Feature Engineering
         ↓
    ML Training (LR / RF / XGBoost / LightGBM)
         ↓
    Best Model Selection (CV ROC-AUC)
         ↓
    SHAP Explainability
         ↓
    FastAPI REST API
         ↓
    Streamlit Dashboard  ← You are here
    ```

    ### Disclaimer
    ⚠️ **This tool is for research and decision support only. It is NOT a substitute
    for clinical judgment. All predictions should be interpreted by qualified clinicians
    in the context of the full patient presentation.**

    ### Technical Stack
    - **ML:** scikit-learn, XGBoost, LightGBM
    - **Explainability:** SHAP (SHapley Additive exPlanations)
    - **API:** FastAPI + Uvicorn
    - **Dashboard:** Streamlit + Plotly
    - **Statistical:** SAS (baseline modeling)
    - **MLOps:** MLflow, Docker, GitHub Actions
    """)
