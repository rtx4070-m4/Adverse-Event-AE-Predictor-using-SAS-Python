/* ============================================================
   ADVERSE EVENT RISK PREDICTOR
   SAS Baseline Statistical Modeling
   File: sas/baseline_model.sas
   ============================================================
   Purpose:
     - Logistic regression baseline (clinical standard)
     - Univariate odds ratio analysis
     - AUROC calculation
     - Clinical score calibration (similar to NEWS/SOFA)
     - Export coefficients for comparison with ML models
   ============================================================ */

OPTIONS NODATE NONUMBER LINESIZE=120 PAGESIZE=60;
TITLE "MIMIC-III Adverse Event - SAS Baseline Statistical Models";

%LET DATA_PROC = /path/to/adverse-event-predictor/data/processed;

/* Load cleaned data */
PROC IMPORT
    DATAFILE = "&DATA_PROC./ae_cleaned.csv"
    OUT      = WORK.ae_model
    DBMS     = CSV
    REPLACE;
    GETNAMES = YES;
RUN;

/* ============================================================
   SECTION 1: UNIVARIATE ANALYSIS
   Odds Ratios for each predictor
   ============================================================ */
TITLE2 "Univariate Analysis - Odds Ratios";

%MACRO univariate_or(var=, label=);
    PROC LOGISTIC DATA=WORK.ae_model NOPRINT;
        MODEL adverse_event (EVENT='1') = &var / CLODDS=WALD;
        ODS OUTPUT OddsRatios=WORK.or_&var ParameterEstimates=WORK.pe_&var;
    RUN;
    DATA WORK.or_summary_&var;
        SET WORK.or_&var;
        variable = "&var";
        varLabel = "&label";
    RUN;
%MEND;

%univariate_or(var=age,                 label=Age (years));
%univariate_or(var=gender,              label=Gender (1=Male));
%univariate_or(var=length_of_stay,      label=ICU Length of Stay);
%univariate_or(var=heart_rate_mean,     label=Mean Heart Rate);
%univariate_or(var=creatinine_max,      label=Max Creatinine);
%univariate_or(var=wbc_mean,            label=Mean WBC);
%univariate_or(var=drug_count,          label=Drug Count);
%univariate_or(var=polypharmacy_score,  label=Polypharmacy Score);
%univariate_or(var=lab_abnormality_score label=Lab Abnormality Score);

/* Combine all OR summaries */
DATA WORK.all_or;
    SET WORK.or_summary_age
        WORK.or_summary_gender
        WORK.or_summary_length_of_stay
        WORK.or_summary_heart_rate_mean
        WORK.or_summary_creatinine_max
        WORK.or_summary_wbc_mean
        WORK.or_summary_drug_count
        WORK.or_summary_polypharmacy_score
        WORK.or_summary_lab_abnormality_score;
    OddsRatioEst = OddsRatioEst;  /* Already labeled from proc */
RUN;

PROC PRINT DATA=WORK.all_or LABEL NOOBS;
    VAR variable varLabel OddsRatioEst LowerCL UpperCL;
    FORMAT OddsRatioEst LowerCL UpperCL 8.3;
    TITLE2 "Univariate Odds Ratios with 95% Confidence Intervals";
RUN;


/* ============================================================
   SECTION 2: MULTIVARIATE LOGISTIC REGRESSION
   Full model with all clinical predictors
   ============================================================ */
TITLE2 "Multivariate Logistic Regression - Full Model";

PROC LOGISTIC DATA=WORK.ae_model DESCENDING;
    MODEL adverse_event =
        age
        gender
        length_of_stay
        heart_rate_mean
        creatinine_max
        wbc_mean
        drug_count
        polypharmacy_score
        lab_abnormality_score
        / SELECTION = STEPWISE
          SLENTRY   = 0.05
          SLSTAY    = 0.10
          CLODDS    = WALD
          LACKFIT               /* Hosmer-Lemeshow goodness-of-fit */
          RSQUARE               /* R-squared measures */
          CTABLE                /* Classification table */
          PPROB     = 0.5;      /* Cutoff for classification */
    OUTPUT OUT=WORK.ae_logit_pred P=pred_prob;
    ODS OUTPUT
        ParameterEstimates = WORK.logit_coefs
        OddsRatios         = WORK.logit_or
        Association        = WORK.logit_assoc
        FitStatistics      = WORK.logit_fit;
    TITLE2 "Stepwise Logistic Regression";
RUN;


/* ============================================================
   SECTION 3: AUROC CALCULATION
   ============================================================ */
TITLE2 "ROC Curve Analysis";

PROC LOGISTIC DATA=WORK.ae_model DESCENDING;
    MODEL adverse_event =
        age gender length_of_stay heart_rate_mean
        creatinine_max wbc_mean drug_count
        polypharmacy_score lab_abnormality_score;
    ROC "Full Model" PRED=pred_prob;
    ROCCONTRAST;
    ODS OUTPUT ROCAssociation=WORK.roc_stats;
RUN;

/* Extract C-statistic (AUROC) */
DATA _NULL_;
    SET WORK.logit_assoc;
    IF Label2 = "c" THEN
        CALL SYMPUTX("c_stat", nValue2);
RUN;
%PUT NOTE: C-Statistic (AUROC) = &c_stat;


/* ============================================================
   SECTION 4: CALIBRATION ANALYSIS
   Hosmer-Lemeshow Test + Calibration Plot Data
   ============================================================ */
TITLE2 "Model Calibration";

/* Create decile groups for calibration plot */
PROC RANK DATA=WORK.ae_logit_pred OUT=WORK.ae_ranked GROUPS=10;
    VAR pred_prob;
    RANKS prob_decile;
RUN;

PROC MEANS DATA=WORK.ae_ranked NOPRINT;
    CLASS prob_decile;
    VAR pred_prob adverse_event;
    OUTPUT OUT=WORK.calibration_data
        MEAN=mean_pred mean_obs
        N=n_patients;
RUN;

PROC PRINT DATA=WORK.calibration_data NOOBS;
    VAR prob_decile n_patients mean_pred mean_obs;
    FORMAT mean_pred mean_obs 8.4;
    TITLE2 "Calibration Table: Predicted vs Observed Rates by Decile";
RUN;


/* ============================================================
   SECTION 5: CLINICAL RISK SCORE
   Simplified integer score for bedside use
   ============================================================ */
TITLE2 "Clinical Risk Score Development";

/* Step 5a: Categorize continuous predictors */
DATA WORK.ae_scored;
    SET WORK.ae_model;

    /* Age Score (0-3 points) */
    IF age < 45 THEN age_score = 0;
    ELSE IF 45 <= age < 65  THEN age_score = 1;
    ELSE IF 65 <= age < 80  THEN age_score = 2;
    ELSE age_score = 3;

    /* Heart Rate Score (0-3 points) */
    IF heart_rate_mean < 60        THEN hr_score = 2;  /* Bradycardia = risk */
    ELSE IF 60 <= heart_rate_mean < 100 THEN hr_score = 0;  /* Normal */
    ELSE IF 100 <= heart_rate_mean < 130 THEN hr_score = 1;  /* Tachycardia */
    ELSE hr_score = 3;  /* Extreme */

    /* Creatinine Score (0-3 points) */
    IF creatinine_max <= 1.2       THEN crea_score = 0;
    ELSE IF 1.2 < creatinine_max <= 2.0 THEN crea_score = 1;
    ELSE IF 2.0 < creatinine_max <= 5.0 THEN crea_score = 2;
    ELSE crea_score = 3;

    /* WBC Score (0-2 points) */
    IF 4 <= wbc_mean <= 12         THEN wbc_score = 0;
    ELSE IF wbc_mean > 12          THEN wbc_score = 1;
    ELSE wbc_score = 2;  /* Leukopenia also risk */

    /* Drug/Polypharmacy Score (0-2 points) */
    IF drug_count < 5              THEN drug_score = 0;
    ELSE IF 5 <= drug_count < 10   THEN drug_score = 1;
    ELSE drug_score = 2;

    /* Lab Abnormality Score (0-3 points, already scored) */
    lab_score = MIN(lab_abnormality_score, 3);

    /* Total Clinical Risk Score (0-16 points) */
    ae_risk_score = age_score + hr_score + crea_score +
                    wbc_score + drug_score + lab_score;

    /* Risk Category */
    LENGTH risk_category $10;
    IF ae_risk_score <= 3       THEN risk_category = "Low";
    ELSE IF 4 <= ae_risk_score <= 7 THEN risk_category = "Moderate";
    ELSE IF 8 <= ae_risk_score <= 11 THEN risk_category = "High";
    ELSE risk_category = "Critical";
RUN;

/* Score performance */
PROC FREQ DATA=WORK.ae_scored;
    TABLES risk_category * adverse_event / NOCOL NOROW NOPERCENT;
    TITLE2 "Clinical Risk Score vs Adverse Event Outcome";
RUN;

PROC MEANS DATA=WORK.ae_scored N MEAN STDDEV;
    CLASS risk_category;
    VAR adverse_event;
    FORMAT adverse_event 8.3;
    TITLE2 "Adverse Event Rate by Risk Category";
RUN;


/* ============================================================
   SECTION 6: EXPORT RESULTS
   ============================================================ */

/* Export logistic regression coefficients */
PROC EXPORT
    DATA    = WORK.logit_coefs
    OUTFILE = "&DATA_PROC./sas_logit_coefficients.csv"
    DBMS    = CSV REPLACE;
RUN;

/* Export odds ratios */
PROC EXPORT
    DATA    = WORK.logit_or
    OUTFILE = "&DATA_PROC./sas_odds_ratios.csv"
    DBMS    = CSV REPLACE;
RUN;

/* Export calibration data */
PROC EXPORT
    DATA    = WORK.calibration_data
    OUTFILE = "&DATA_PROC./sas_calibration.csv"
    DBMS    = CSV REPLACE;
RUN;

/* Export scored dataset */
PROC EXPORT
    DATA    = WORK.ae_scored (KEEP=subject_id hadm_id icustay_id
                                    ae_risk_score risk_category adverse_event)
    OUTFILE = "&DATA_PROC./sas_clinical_scores.csv"
    DBMS    = CSV REPLACE;
RUN;

%PUT NOTE: SAS baseline modeling complete.;
%PUT NOTE: C-statistic (AUROC) = &c_stat;
TITLE;
