/* ============================================================
   ADVERSE EVENT RISK PREDICTOR
   SAS Statistical Processing & Data Cleaning
   File: sas/data_cleaning.sas
   ============================================================
   Purpose:
     - Import raw feature matrix from SQL extract
     - Perform clinical data validation
     - Handle missing values with clinical imputation
     - Detect and handle outliers
     - Create derived clinical variables
     - Export cleaned dataset for Python ML pipeline
   ============================================================ */

OPTIONS NODATE NONUMBER LINESIZE=120 PAGESIZE=60;
TITLE "MIMIC-III Adverse Event Risk Predictor - Data Cleaning";

/* ============================================================
   SECTION 0: MACRO DEFINITIONS
   ============================================================ */
%LET PROJECT_ROOT = /path/to/adverse-event-predictor;
%LET DATA_RAW     = &PROJECT_ROOT./data/raw;
%LET DATA_PROC    = &PROJECT_ROOT./data/processed;
%LET LOGFILE      = &PROJECT_ROOT./data/processed/cleaning_log.txt;

/* Macro: Print dataset info */
%MACRO dataset_info(dsname=);
    PROC SQL;
        SELECT COUNT(*) AS n_rows, NMISS(adverse_event) AS missing_label
        FROM &dsname;
    QUIT;
    PROC CONTENTS DATA=&dsname VARNUM; RUN;
%MEND dataset_info;

/* Macro: Winsorize a variable at specified percentiles */
%MACRO winsorize(dsn=, var=, lower_pct=0.01, upper_pct=0.99);
    PROC UNIVARIATE DATA=&dsn NOPRINT;
        VAR &var;
        OUTPUT OUT=_pct_&var
            PCTLPTS  = %SYSEVALF(&lower_pct * 100) %SYSEVALF(&upper_pct * 100)
            PCTLPRE  = p;
    RUN;
    DATA _NULL_;
        SET _pct_&var;
        CALL SYMPUTX('lower_&var', p%SYSEVALF(&lower_pct * 100));
        CALL SYMPUTX('upper_&var', p%SYSEVALF(&upper_pct * 100));
    RUN;
    DATA &dsn;
        SET &dsn;
        IF &var < &&lower_&var THEN &var = &&lower_&var;
        IF &var > &&upper_&var THEN &var = &&upper_&var;
    RUN;
    PROC DATASETS LIBRARY=WORK NOLIST;
        DELETE _pct_&var;
    RUN;
%MEND winsorize;


/* ============================================================
   SECTION 1: IMPORT RAW FEATURE MATRIX
   ============================================================ */
PROC IMPORT
    DATAFILE = "&DATA_RAW./ae_feature_matrix.csv"
    OUT      = WORK.ae_raw
    DBMS     = CSV
    REPLACE;
    GETNAMES = YES;
    GUESSINGROWS = 5000;
RUN;

%dataset_info(dsname=WORK.ae_raw);


/* ============================================================
   SECTION 2: INITIAL DATA VALIDATION
   ============================================================ */
/* Check for duplicate ICU stays */
PROC SORT DATA=WORK.ae_raw NODUPKEY DUPOUT=WORK.ae_duplicates
    OUT=WORK.ae_dedup;
    BY icustay_id;
RUN;

%LET n_dups = 0;
DATA _NULL_;
    IF 0 THEN SET WORK.ae_duplicates NOBS=n;
    CALL SYMPUTX('n_dups', n);
RUN;
%PUT NOTE: Removed &n_dups duplicate ICU stay records.;

/* Validate age range */
DATA WORK.ae_age_valid;
    SET WORK.ae_dedup;
    IF age < 18 OR age > 110 THEN DELETE;
RUN;

/* Validate length of stay */
DATA WORK.ae_los_valid;
    SET WORK.ae_age_valid;
    IF length_of_stay <= 0 OR length_of_stay > 365 THEN DELETE;
RUN;

/* Missing value summary before imputation */
PROC MEANS DATA=WORK.ae_los_valid NMISS N MEAN STDDEV MIN MAX MAXDEC=3;
    VAR age gender length_of_stay heart_rate_mean creatinine_max
        wbc_mean drug_count polypharmacy_score lab_abnormality_score;
    TITLE2 "Missing Value Summary - Pre-Imputation";
RUN;


/* ============================================================
   SECTION 3: OUTLIER DETECTION (IQR Method)
   ============================================================ */
PROC UNIVARIATE DATA=WORK.ae_los_valid;
    VAR heart_rate_mean creatinine_max wbc_mean length_of_stay;
    OUTPUT OUT=WORK.outlier_stats
        Q1=hr_q1 crea_q1 wbc_q1 los_q1
        Q3=hr_q3 crea_q3 wbc_q3 los_q3
        QRANGE=hr_iqr crea_iqr wbc_iqr los_iqr;
RUN;

DATA _NULL_;
    SET WORK.outlier_stats;
    /* Heart Rate */
    CALL SYMPUTX('hr_lo',   hr_q1   - 3 * hr_iqr);
    CALL SYMPUTX('hr_hi',   hr_q3   + 3 * hr_iqr);
    /* Creatinine */
    CALL SYMPUTX('crea_lo', crea_q1 - 3 * crea_iqr);
    CALL SYMPUTX('crea_hi', crea_q3 + 3 * crea_iqr);
    /* WBC */
    CALL SYMPUTX('wbc_lo',  wbc_q1  - 3 * wbc_iqr);
    CALL SYMPUTX('wbc_hi',  wbc_q3  + 3 * wbc_iqr);
RUN;

/* Flag and remove physiological outliers */
DATA WORK.ae_no_outliers;
    SET WORK.ae_los_valid;
    /* Clinical plausibility hard limits + statistical outlier removal */
    IF heart_rate_mean  < MAX(&hr_lo,  20)  THEN DELETE;
    IF heart_rate_mean  > MIN(&hr_hi,  280) THEN DELETE;
    IF creatinine_max   < 0                  THEN DELETE;
    IF creatinine_max   > MIN(&crea_hi, 30) THEN DELETE;
    IF wbc_mean         < 0                  THEN DELETE;
    IF wbc_mean         > MIN(&wbc_hi,  150) THEN DELETE;
RUN;


/* ============================================================
   SECTION 4: MISSING VALUE IMPUTATION
   Clinical rules-based + statistical imputation
   ============================================================ */

/* Step 4a: Compute medians by age group for stratified imputation */
PROC SORT DATA=WORK.ae_no_outliers; BY adverse_event; RUN;

PROC MEANS DATA=WORK.ae_no_outliers NOPRINT;
    BY adverse_event;
    VAR heart_rate_mean creatinine_max wbc_mean lactate_max
        hemoglobin_min platelets_min polypharmacy_score;
    OUTPUT OUT=WORK.strata_medians
        MEDIAN=med_hr med_crea med_wbc med_lac med_hgb med_plt med_poly;
RUN;

/* Step 4b: Imputation using stratified medians + clinical defaults */
DATA WORK.ae_imputed;
    SET WORK.ae_no_outliers;

    /* --- Heart Rate: impute with clinical reference 80 bpm if missing --- */
    IF MISSING(heart_rate_mean) THEN DO;
        heart_rate_mean = 80;
        hr_imputed = 1;
    END;
    ELSE hr_imputed = 0;

    /* --- Creatinine: impute with 1.0 (upper normal) if missing --- */
    IF MISSING(creatinine_max) THEN DO;
        creatinine_max  = 1.0;
        creatinine_mean = 1.0;
        crea_imputed = 1;
    END;
    ELSE crea_imputed = 0;

    /* --- WBC: impute with 9.0 (mid-normal) if missing --- */
    IF MISSING(wbc_mean) THEN DO;
        wbc_mean    = 9.0;
        wbc_max     = 9.0;
        wbc_imputed = 1;
    END;
    ELSE wbc_imputed = 0;

    /* --- Lactate: impute with 1.5 (low-normal) if missing --- */
    IF MISSING(lactate_max) THEN lactate_max = 1.5;

    /* --- Hemoglobin: impute with 12.0 if missing --- */
    IF MISSING(hemoglobin_min) THEN hemoglobin_min = 12.0;

    /* --- Platelets: impute with 200 if missing --- */
    IF MISSING(platelets_min) THEN platelets_min = 200;

    /* --- Drug count: impute with 0 if missing --- */
    IF MISSING(drug_count)        THEN drug_count        = 0;
    IF MISSING(polypharmacy_score) THEN polypharmacy_score = 0;
    IF MISSING(polypharmacy_flag)  THEN polypharmacy_flag  = 0;

    /* --- Gender: impute with mode (0=Female most common in MIMIC) --- */
    IF MISSING(gender) THEN gender = 0;

    /* --- Lab Abnormality Score: default 0 --- */
    IF MISSING(lab_abnormality_score) THEN lab_abnormality_score = 0;

    /* Track imputation */
    any_imputed = MAX(hr_imputed, crea_imputed, wbc_imputed);

    LABEL
        hr_imputed   = "Heart Rate Was Imputed"
        crea_imputed = "Creatinine Was Imputed"
        wbc_imputed  = "WBC Was Imputed"
        any_imputed  = "Any Value Was Imputed";
RUN;

/* Missing value summary AFTER imputation */
PROC MEANS DATA=WORK.ae_imputed NMISS N MAXDEC=3;
    VAR age gender length_of_stay heart_rate_mean creatinine_max
        wbc_mean drug_count polypharmacy_score lab_abnormality_score;
    TITLE2 "Missing Value Summary - Post-Imputation";
RUN;


/* ============================================================
   SECTION 5: DERIVED CLINICAL VARIABLES
   ============================================================ */
DATA WORK.ae_features;
    SET WORK.ae_imputed;

    /* --- Age Groups --- */
    LENGTH age_group $10;
    SELECT;
        WHEN (age < 45)              age_group = "Young";
        WHEN (45 <= age < 65)        age_group = "Middle";
        WHEN (65 <= age < 80)        age_group = "Senior";
        WHEN (age >= 80)             age_group = "Elderly";
        OTHERWISE                    age_group = "Unknown";
    END;

    /* --- Length of Stay Category --- */
    LENGTH los_category $10;
    SELECT;
        WHEN (length_of_stay < 1)    los_category = "Short";
        WHEN (1 <= length_of_stay < 3) los_category = "Medium";
        WHEN (3 <= length_of_stay < 7) los_category = "Long";
        WHEN (length_of_stay >= 7)   los_category = "Very Long";
        OTHERWISE                    los_category = "Unknown";
    END;

    /* --- Tachycardia Flag (HR > 100) --- */
    tachycardia = (heart_rate_mean > 100);

    /* --- Renal Impairment (Creatinine > 1.5) --- */
    renal_impairment = (creatinine_max > 1.5);

    /* --- Leukocytosis (WBC > 12) --- */
    leukocytosis = (wbc_mean > 12);

    /* --- Composite Clinical Risk Index --- */
    /* Weighted sum of adverse physiological indicators */
    clinical_risk_index =
        (0.25 * tachycardia)            +
        (0.30 * renal_impairment)       +
        (0.20 * leukocytosis)           +
        (0.15 * polypharmacy_flag)      +
        (0.10 * (lab_abnormality_score >= 2));

    /* --- Normalized Creatinine (z-score approximation) --- */
    /* Mean ~1.1, SD ~1.4 in ICU population */
    creatinine_zscore = (creatinine_max - 1.1) / 1.4;

    /* --- Heart Rate Variability Category --- */
    LENGTH hr_category $10;
    IF heart_rate_mean < 60      THEN hr_category = "Brady";
    ELSE IF heart_rate_mean < 100 THEN hr_category = "Normal";
    ELSE IF heart_rate_mean < 130 THEN hr_category = "Tachy";
    ELSE                              hr_category = "Extreme";

    LABEL
        age_group          = "Age Group Category"
        los_category       = "ICU Length of Stay Category"
        tachycardia        = "Tachycardia Indicator (HR>100)"
        renal_impairment   = "Renal Impairment (Creat>1.5)"
        leukocytosis       = "Leukocytosis (WBC>12)"
        clinical_risk_index = "Composite Clinical Risk Index"
        creatinine_zscore  = "Standardized Creatinine"
        hr_category        = "Heart Rate Category"
        adverse_event      = "Adverse Event Outcome (0/1)";
RUN;


/* ============================================================
   SECTION 6: WINSORIZATION (Cap Extreme Values)
   ============================================================ */
%winsorize(dsn=WORK.ae_features, var=length_of_stay,   lower_pct=0.01, upper_pct=0.99);
%winsorize(dsn=WORK.ae_features, var=heart_rate_mean,  lower_pct=0.01, upper_pct=0.99);
%winsorize(dsn=WORK.ae_features, var=creatinine_max,   lower_pct=0.01, upper_pct=0.99);
%winsorize(dsn=WORK.ae_features, var=wbc_mean,         lower_pct=0.01, upper_pct=0.99);
%winsorize(dsn=WORK.ae_features, var=drug_count,       lower_pct=0.00, upper_pct=0.99);


/* ============================================================
   SECTION 7: FINAL CLEANED DATASET
   ============================================================ */
DATA WORK.ae_cleaned;
    SET WORK.ae_features;
    /* Keep only complete cases for core ML features */
    IF NMISS(age, gender, length_of_stay, heart_rate_mean,
             creatinine_max, wbc_mean, drug_count,
             polypharmacy_score, lab_abnormality_score,
             adverse_event) = 0;
RUN;

/* Final dataset summary */
PROC MEANS DATA=WORK.ae_cleaned N MEAN STDDEV MIN MAX MAXDEC=3;
    VAR age gender length_of_stay heart_rate_mean creatinine_max
        wbc_mean drug_count polypharmacy_score lab_abnormality_score
        clinical_risk_index;
    CLASS adverse_event;
    TITLE2 "Final Cleaned Feature Summary by Outcome";
RUN;

/* Adverse event rate */
PROC FREQ DATA=WORK.ae_cleaned;
    TABLES adverse_event / NOCUM;
    TITLE2 "Adverse Event Distribution";
RUN;


/* ============================================================
   SECTION 8: EXPORT FOR PYTHON ML PIPELINE
   ============================================================ */
PROC EXPORT
    DATA    = WORK.ae_cleaned
    OUTFILE = "&DATA_PROC./ae_cleaned.csv"
    DBMS    = CSV
    REPLACE;
RUN;

/* Export feature definitions */
DATA WORK.feature_dict;
    LENGTH variable $40 description $200 type $20 missing_action $50;
    INPUT variable $ description $ type $ missing_action $;
    DATALINES;
age "Patient age in years" continuous "Clinical reference 65"
gender "Patient sex (1=Male, 0=Female)" binary "Mode imputation"
length_of_stay "ICU length of stay in days" continuous "Exclude negatives"
heart_rate_mean "Mean heart rate during ICU stay" continuous "Clinical default 80"
creatinine_max "Maximum serum creatinine" continuous "Clinical default 1.0"
wbc_mean "Mean white blood cell count" continuous "Clinical default 9.0"
drug_count "Total distinct drugs prescribed" count "Zero imputation"
polypharmacy_score "Polypharmacy burden score 0-10" continuous "Zero imputation"
lab_abnormality_score "Count of critical lab values" count "Zero imputation"
adverse_event "Composite adverse outcome label" binary "Target variable"
;
RUN;

PROC EXPORT
    DATA    = WORK.feature_dict
    OUTFILE = "&DATA_PROC./feature_dictionary.csv"
    DBMS    = CSV
    REPLACE;
RUN;

%PUT NOTE: Data cleaning complete. Output: &DATA_PROC./ae_cleaned.csv;
TITLE;
