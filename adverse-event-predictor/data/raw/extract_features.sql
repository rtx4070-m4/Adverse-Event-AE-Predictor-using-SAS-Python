-- ============================================================
-- MIMIC-III: Adverse Event Risk Predictor
-- SQL Feature Extraction Layer
-- ============================================================
-- Target: PostgreSQL (MIMIC-III standard setup)
-- Schema: mimiciii
-- ============================================================

-- ============================================================
-- STEP 1: Base Cohort (ICU patients)
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS ae_cohort CASCADE;
CREATE MATERIALIZED VIEW ae_cohort AS
WITH base AS (
    SELECT
        p.subject_id,
        p.gender,
        p.dob,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.deathtime,
        a.hospital_expire_flag,
        a.diagnosis,
        i.icustay_id,
        i.intime  AS icu_intime,
        i.outtime AS icu_outtime,
        i.los     AS icu_los_days,
        EXTRACT(EPOCH FROM (a.admittime - p.dob)) / (365.25 * 86400) AS age_years
    FROM mimiciii.patients   p
    JOIN mimiciii.admissions  a ON p.subject_id = a.subject_id
    JOIN mimiciii.icustays    i ON a.hadm_id    = i.hadm_id
    WHERE
        a.has_chartevents_data = 1
        AND i.los IS NOT NULL
        AND i.los > 0
        AND EXTRACT(EPOCH FROM (a.admittime - p.dob)) / (365.25 * 86400) BETWEEN 18 AND 110
)
SELECT * FROM base;

CREATE INDEX ON ae_cohort (subject_id);
CREATE INDEX ON ae_cohort (hadm_id);
CREATE INDEX ON ae_cohort (icustay_id);


-- ============================================================
-- STEP 2: Vital Signs - Heart Rate (CHARTEVENTS)
-- itemid 211  = Heart Rate (CareVue)
-- itemid 220045 = Heart Rate (Metavision)
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS ae_heart_rate CASCADE;
CREATE MATERIALIZED VIEW ae_heart_rate AS
SELECT
    c.icustay_id,
    AVG(c.valuenum)  AS heart_rate_mean,
    MIN(c.valuenum)  AS heart_rate_min,
    MAX(c.valuenum)  AS heart_rate_max,
    STDDEV(c.valuenum) AS heart_rate_std
FROM mimiciii.chartevents c
JOIN ae_cohort            co ON c.icustay_id = co.icustay_id
WHERE
    c.itemid IN (211, 220045)
    AND c.valuenum IS NOT NULL
    AND c.valuenum BETWEEN 20 AND 300   -- physiologically plausible range
    AND c.error IS DISTINCT FROM 1
GROUP BY c.icustay_id;

CREATE INDEX ON ae_heart_rate (icustay_id);


-- ============================================================
-- STEP 3: Lab Values (LABEVENTS)
-- Creatinine: itemid 50912
-- WBC:        itemid 51301
-- Lactate:    itemid 50813
-- Hemoglobin: itemid 51222
-- Platelets:  itemid 51265
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS ae_labs CASCADE;
CREATE MATERIALIZED VIEW ae_labs AS
SELECT
    le.hadm_id,
    -- Creatinine
    MAX(CASE WHEN le.itemid = 50912 THEN le.valuenum END) AS creatinine_max,
    AVG(CASE WHEN le.itemid = 50912 THEN le.valuenum END) AS creatinine_mean,
    -- WBC
    AVG(CASE WHEN le.itemid = 51301 THEN le.valuenum END) AS wbc_mean,
    MAX(CASE WHEN le.itemid = 51301 THEN le.valuenum END) AS wbc_max,
    -- Lactate
    MAX(CASE WHEN le.itemid = 50813 THEN le.valuenum END) AS lactate_max,
    AVG(CASE WHEN le.itemid = 50813 THEN le.valuenum END) AS lactate_mean,
    -- Hemoglobin
    MIN(CASE WHEN le.itemid = 51222 THEN le.valuenum END) AS hemoglobin_min,
    AVG(CASE WHEN le.itemid = 51222 THEN le.valuenum END) AS hemoglobin_mean,
    -- Platelets
    MIN(CASE WHEN le.itemid = 51265 THEN le.valuenum END) AS platelets_min,
    AVG(CASE WHEN le.itemid = 51265 THEN le.valuenum END) AS platelets_mean,
    -- Lab Abnormality Score (count of critical values)
    SUM(CASE
            WHEN le.itemid = 50912 AND le.valuenum > 3.5  THEN 1
            WHEN le.itemid = 51301 AND (le.valuenum > 12 OR le.valuenum < 3) THEN 1
            WHEN le.itemid = 50813 AND le.valuenum > 4    THEN 1
            WHEN le.itemid = 51222 AND le.valuenum < 7    THEN 1
            WHEN le.itemid = 51265 AND le.valuenum < 80   THEN 1
            ELSE 0
        END) AS lab_abnormality_score
FROM mimiciii.labevents le
JOIN ae_cohort          co ON le.hadm_id = co.hadm_id
WHERE
    le.itemid IN (50912, 51301, 50813, 51222, 51265)
    AND le.valuenum IS NOT NULL
    AND le.valuenum > 0
GROUP BY le.hadm_id;

CREATE INDEX ON ae_labs (hadm_id);


-- ============================================================
-- STEP 4: Medications / Prescriptions
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS ae_medications CASCADE;
CREATE MATERIALIZED VIEW ae_medications AS
SELECT
    pr.hadm_id,
    COUNT(DISTINCT pr.drug)          AS drug_count,
    COUNT(DISTINCT pr.formulary_drug_cd) AS unique_drug_formulations,
    -- Polypharmacy score (weighted by drug class risk)
    COUNT(DISTINCT CASE
        WHEN LOWER(pr.drug) LIKE '%anticoagul%' THEN pr.drug
        WHEN LOWER(pr.drug) LIKE '%warfarin%'   THEN pr.drug
        WHEN LOWER(pr.drug) LIKE '%heparin%'    THEN pr.drug
        WHEN LOWER(pr.drug) LIKE '%insulin%'    THEN pr.drug
        WHEN LOWER(pr.drug) LIKE '%opioid%'     THEN pr.drug
        WHEN LOWER(pr.drug) LIKE '%morphine%'   THEN pr.drug
        WHEN LOWER(pr.drug) LIKE '%sedative%'   THEN pr.drug
    END) AS high_risk_drug_count,
    -- Polypharmacy binary flag (>=5 drugs = polypharmacy)
    CASE WHEN COUNT(DISTINCT pr.drug) >= 5 THEN 1 ELSE 0 END AS polypharmacy_flag,
    -- Polypharmacy score (0-10 scale)
    LEAST(COUNT(DISTINCT pr.drug) / 2.0, 10) AS polypharmacy_score
FROM mimiciii.prescriptions pr
JOIN ae_cohort              co ON pr.hadm_id = co.hadm_id
WHERE
    pr.drug IS NOT NULL
    AND pr.startdate IS NOT NULL
GROUP BY pr.hadm_id;

CREATE INDEX ON ae_medications (hadm_id);


-- ============================================================
-- STEP 5: Adverse Event Label Definition
-- An adverse event is defined as ANY of:
--   (a) In-hospital death (hospital_expire_flag = 1)
--   (b) ICU readmission within same hospitalization
--   (c) Critical lab abnormality score >= 3
--   (d) Sepsis/shock diagnosis codes (ICD9)
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS ae_labels CASCADE;
CREATE MATERIALIZED VIEW ae_labels AS
WITH icu_readmit AS (
    -- Detect ICU readmissions within same hospitalization
    SELECT
        hadm_id,
        COUNT(*) AS icu_stays_count,
        CASE WHEN COUNT(*) > 1 THEN 1 ELSE 0 END AS icu_readmission
    FROM mimiciii.icustays
    GROUP BY hadm_id
),
sepsis_dx AS (
    -- ICD-9 codes for sepsis / septic shock / organ failure
    SELECT DISTINCT hadm_id, 1 AS has_sepsis
    FROM mimiciii.diagnoses_icd
    WHERE icd9_code IN (
        '99591','99592',                         -- Sepsis / Severe sepsis
        '78552',                                 -- Septic shock
        '99811','99812',                         -- Acute organ failure
        '5849','5845','5846','5847',             -- Acute kidney injury
        '99859',                                 -- Other septicemia
        '99802'                                  -- Unspecified septicemia
    )
)
SELECT
    co.icustay_id,
    co.hadm_id,
    co.subject_id,
    co.hospital_expire_flag,
    COALESCE(ir.icu_readmission, 0)  AS icu_readmission,
    COALESCE(sd.has_sepsis, 0)       AS has_sepsis,
    COALESCE(la.lab_abnormality_score >= 3, FALSE)::INT AS critical_labs,
    -- FINAL LABEL: adverse_event = 1 if any adverse outcome
    GREATEST(
        co.hospital_expire_flag,
        COALESCE(ir.icu_readmission, 0),
        COALESCE(sd.has_sepsis, 0),
        (COALESCE(la.lab_abnormality_score, 0) >= 3)::INT
    ) AS adverse_event
FROM ae_cohort     co
LEFT JOIN icu_readmit ir ON co.hadm_id = ir.hadm_id
LEFT JOIN sepsis_dx   sd ON co.hadm_id = sd.hadm_id
LEFT JOIN ae_labs     la ON co.hadm_id = la.hadm_id;

CREATE INDEX ON ae_labels (icustay_id);
CREATE INDEX ON ae_labels (hadm_id);


-- ============================================================
-- STEP 6: Final Feature Matrix
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS ae_feature_matrix CASCADE;
CREATE MATERIALIZED VIEW ae_feature_matrix AS
SELECT
    -- Identifiers
    co.subject_id,
    co.hadm_id,
    co.icustay_id,

    -- Demographics
    ROUND(co.age_years::NUMERIC, 1)        AS age,
    CASE WHEN co.gender = 'M' THEN 1
         WHEN co.gender = 'F' THEN 0
         ELSE NULL END                      AS gender,  -- 1=Male, 0=Female

    -- ICU Stay
    ROUND(co.icu_los_days::NUMERIC, 2)     AS length_of_stay,

    -- Vital Signs
    ROUND(hr.heart_rate_mean::NUMERIC, 2)  AS heart_rate_mean,
    ROUND(hr.heart_rate_min::NUMERIC, 2)   AS heart_rate_min,
    ROUND(hr.heart_rate_max::NUMERIC, 2)   AS heart_rate_max,
    ROUND(hr.heart_rate_std::NUMERIC, 2)   AS heart_rate_std,

    -- Lab Results
    ROUND(la.creatinine_max::NUMERIC, 3)   AS creatinine_max,
    ROUND(la.creatinine_mean::NUMERIC, 3)  AS creatinine_mean,
    ROUND(la.wbc_mean::NUMERIC, 3)         AS wbc_mean,
    ROUND(la.wbc_max::NUMERIC, 3)          AS wbc_max,
    ROUND(la.lactate_max::NUMERIC, 3)      AS lactate_max,
    ROUND(la.hemoglobin_min::NUMERIC, 3)   AS hemoglobin_min,
    ROUND(la.platelets_min::NUMERIC, 2)    AS platelets_min,
    COALESCE(la.lab_abnormality_score, 0)  AS lab_abnormality_score,

    -- Medications
    COALESCE(me.drug_count, 0)             AS drug_count,
    COALESCE(me.high_risk_drug_count, 0)   AS high_risk_drug_count,
    COALESCE(me.polypharmacy_flag, 0)      AS polypharmacy_flag,
    ROUND(COALESCE(me.polypharmacy_score, 0)::NUMERIC, 2) AS polypharmacy_score,

    -- Outcome / Label
    lb.adverse_event,
    lb.hospital_expire_flag,
    lb.icu_readmission,
    lb.has_sepsis,
    lb.critical_labs,

    -- Metadata
    co.admittime,
    co.icu_intime,
    co.diagnosis

FROM ae_cohort    co
LEFT JOIN ae_heart_rate hr ON co.icustay_id = hr.icustay_id
LEFT JOIN ae_labs       la ON co.hadm_id    = la.hadm_id
LEFT JOIN ae_medications me ON co.hadm_id   = me.hadm_id
LEFT JOIN ae_labels      lb ON co.icustay_id = lb.icustay_id
WHERE
    co.age_years BETWEEN 18 AND 110
    AND lb.adverse_event IS NOT NULL;

CREATE INDEX ON ae_feature_matrix (subject_id);
CREATE INDEX ON ae_feature_matrix (hadm_id);
CREATE INDEX ON ae_feature_matrix (icustay_id);
CREATE INDEX ON ae_feature_matrix (adverse_event);

-- ============================================================
-- STEP 7: Export Query for Python ingestion
-- ============================================================
-- Run this to export to CSV:
-- COPY (SELECT * FROM ae_feature_matrix ORDER BY icustay_id)
-- TO '/tmp/ae_feature_matrix.csv' WITH CSV HEADER;

-- Summary statistics
SELECT
    COUNT(*)                              AS total_patients,
    SUM(adverse_event)                    AS adverse_events,
    ROUND(100.0 * AVG(adverse_event), 2) AS adverse_event_rate_pct,
    ROUND(AVG(age), 1)                   AS mean_age,
    ROUND(AVG(length_of_stay), 2)        AS mean_icu_los_days
FROM ae_feature_matrix;
