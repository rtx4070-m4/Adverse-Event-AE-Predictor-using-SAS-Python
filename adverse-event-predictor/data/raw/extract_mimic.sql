-- ============================================================
-- MIMIC-III Adverse Event Risk Predictor
-- SQL Extraction Layer
-- ============================================================

-- Step 1: Core patient cohort (adult ICU patients)
DROP TABLE IF EXISTS adverse_event_cohort;

CREATE TABLE adverse_event_cohort AS
SELECT
    p.subject_id,
    a.hadm_id,
    i.icustay_id,

    -- Demographics
    EXTRACT(EPOCH FROM (a.admittime - p.dob)) / (365.25 * 24 * 3600) AS age,
    CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS gender,

    -- ICU stay info
    EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600.0 AS icu_los_hours,
    EXTRACT(EPOCH FROM (a.dischtime - a.admittime)) / 86400.0 AS hosp_los_days,

    -- Admission info
    a.admission_type,
    a.insurance,
    a.ethnicity,
    a.hospital_expire_flag,

    -- Discharge disposition as adverse event proxy
    CASE
        WHEN a.hospital_expire_flag = 1 THEN 1
        WHEN a.discharge_location IN ('DEAD/EXPIRED', 'HOSPICE-MEDICAL FACILITY', 'HOSPICE-HOME') THEN 1
        ELSE 0
    END AS adverse_event

FROM mimiciii.patients p
JOIN mimiciii.admissions a ON p.subject_id = a.subject_id
JOIN mimiciii.icustays i ON a.hadm_id = i.hadm_id

WHERE
    EXTRACT(EPOCH FROM (a.admittime - p.dob)) / (365.25 * 24 * 3600) >= 18
    AND i.los >= 1  -- at least 1 day in ICU
    AND a.has_chartevents_data = 1;


-- Step 2: Lab values (creatinine, WBC)
DROP TABLE IF EXISTS ae_lab_features;

CREATE TABLE ae_lab_features AS
SELECT
    le.hadm_id,

    -- Creatinine (itemid 50912)
    MAX(CASE WHEN le.itemid = 50912 THEN le.valuenum END) AS creatinine_max,
    AVG(CASE WHEN le.itemid = 50912 THEN le.valuenum END) AS creatinine_mean,

    -- WBC (itemid 51300, 51301)
    AVG(CASE WHEN le.itemid IN (51300, 51301) THEN le.valuenum END) AS wbc_mean,
    MAX(CASE WHEN le.itemid IN (51300, 51301) THEN le.valuenum END) AS wbc_max,

    -- Sodium (itemid 50983)
    AVG(CASE WHEN le.itemid = 50983 THEN le.valuenum END) AS sodium_mean,

    -- Potassium (itemid 50971)
    AVG(CASE WHEN le.itemid = 50971 THEN le.valuenum END) AS potassium_mean,

    -- Hemoglobin (itemid 51222)
    AVG(CASE WHEN le.itemid = 51222 THEN le.valuenum END) AS hemoglobin_mean,
    MIN(CASE WHEN le.itemid = 51222 THEN le.valuenum END) AS hemoglobin_min,

    -- Platelet (itemid 51265)
    MIN(CASE WHEN le.itemid = 51265 THEN le.valuenum END) AS platelet_min,

    -- Glucose (itemid 50931)
    MAX(CASE WHEN le.itemid = 50931 THEN le.valuenum END) AS glucose_max,
    AVG(CASE WHEN le.itemid = 50931 THEN le.valuenum END) AS glucose_mean,

    -- BUN (itemid 51006)
    MAX(CASE WHEN le.itemid = 51006 THEN le.valuenum END) AS bun_max,

    -- Lab abnormality count
    SUM(CASE
        WHEN le.itemid = 50912 AND le.valuenum > 1.2 THEN 1  -- high creatinine
        WHEN le.itemid IN (51300, 51301) AND le.valuenum > 11 THEN 1  -- high WBC
        WHEN le.itemid = 50983 AND (le.valuenum < 135 OR le.valuenum > 145) THEN 1  -- abnormal sodium
        WHEN le.itemid = 50971 AND (le.valuenum < 3.5 OR le.valuenum > 5.0) THEN 1  -- abnormal potassium
        WHEN le.itemid = 51222 AND le.valuenum < 8 THEN 1  -- low hemoglobin
        WHEN le.itemid = 51265 AND le.valuenum < 100 THEN 1  -- low platelets
        ELSE 0
    END) AS lab_abnormality_count

FROM mimiciii.labevents le
WHERE le.valuenum IS NOT NULL
  AND le.valuenum > 0
GROUP BY le.hadm_id;


-- Step 3: Vital signs (heart rate from chartevents)
DROP TABLE IF EXISTS ae_vitals_features;

CREATE TABLE ae_vitals_features AS
SELECT
    ce.icustay_id,

    -- Heart Rate (itemids: 211, 220045)
    AVG(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) AS heart_rate_mean,
    MAX(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) AS heart_rate_max,
    MIN(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) AS heart_rate_min,
    STDDEV(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) AS heart_rate_std,

    -- Systolic BP (itemids: 51, 442, 455, 6701, 220179, 220050)
    AVG(CASE WHEN ce.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN ce.valuenum END) AS sbp_mean,
    MIN(CASE WHEN ce.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN ce.valuenum END) AS sbp_min,

    -- SpO2 (itemids: 646, 220277)
    AVG(CASE WHEN ce.itemid IN (646, 220277) THEN ce.valuenum END) AS spo2_mean,
    MIN(CASE WHEN ce.itemid IN (646, 220277) THEN ce.valuenum END) AS spo2_min,

    -- Temperature (itemids: 678, 223761)
    AVG(CASE WHEN ce.itemid IN (678, 223761) THEN ce.valuenum END) AS temp_mean,
    MAX(CASE WHEN ce.itemid IN (678, 223761) THEN ce.valuenum END) AS temp_max,

    -- Respiratory Rate (itemids: 615, 618, 220210, 224690)
    AVG(CASE WHEN ce.itemid IN (615, 618, 220210, 224690) THEN ce.valuenum END) AS resp_rate_mean,
    MAX(CASE WHEN ce.itemid IN (615, 618, 220210, 224690) THEN ce.valuenum END) AS resp_rate_max

FROM mimiciii.chartevents ce
WHERE ce.valuenum IS NOT NULL
  AND ce.error IS DISTINCT FROM 1
GROUP BY ce.icustay_id;


-- Step 4: Prescriptions / Drug count
DROP TABLE IF EXISTS ae_drug_features;

CREATE TABLE ae_drug_features AS
SELECT
    hadm_id,
    COUNT(DISTINCT drug) AS drug_count,
    COUNT(DISTINCT drug_type) AS drug_type_count,
    -- Polypharmacy: >= 5 drugs simultaneously
    CASE WHEN COUNT(DISTINCT drug) >= 5 THEN 1 ELSE 0 END AS polypharmacy_flag,
    COUNT(DISTINCT drug) AS polypharmacy_score

FROM mimiciii.prescriptions
WHERE drug IS NOT NULL
GROUP BY hadm_id;


-- Step 5: Final merged feature table
DROP TABLE IF EXISTS ae_model_features;

CREATE TABLE ae_model_features AS
SELECT
    c.subject_id,
    c.hadm_id,
    c.icustay_id,
    c.age,
    c.gender,
    c.icu_los_hours,
    c.hosp_los_days,
    c.admission_type,
    c.hospital_expire_flag,
    c.adverse_event,

    -- Lab features
    COALESCE(l.creatinine_max, 1.0) AS creatinine_max,
    COALESCE(l.creatinine_mean, 1.0) AS creatinine_mean,
    COALESCE(l.wbc_mean, 8.0) AS wbc_mean,
    COALESCE(l.wbc_max, 8.0) AS wbc_max,
    COALESCE(l.sodium_mean, 140.0) AS sodium_mean,
    COALESCE(l.potassium_mean, 4.0) AS potassium_mean,
    COALESCE(l.hemoglobin_mean, 12.0) AS hemoglobin_mean,
    COALESCE(l.hemoglobin_min, 12.0) AS hemoglobin_min,
    COALESCE(l.platelet_min, 200.0) AS platelet_min,
    COALESCE(l.glucose_max, 100.0) AS glucose_max,
    COALESCE(l.bun_max, 15.0) AS bun_max,
    COALESCE(l.lab_abnormality_count, 0) AS lab_abnormality_score,

    -- Vital features
    COALESCE(v.heart_rate_mean, 80.0) AS heart_rate_mean,
    COALESCE(v.heart_rate_max, 80.0) AS heart_rate_max,
    COALESCE(v.heart_rate_std, 10.0) AS heart_rate_std,
    COALESCE(v.sbp_mean, 120.0) AS sbp_mean,
    COALESCE(v.sbp_min, 120.0) AS sbp_min,
    COALESCE(v.spo2_mean, 98.0) AS spo2_mean,
    COALESCE(v.spo2_min, 98.0) AS spo2_min,
    COALESCE(v.temp_mean, 37.0) AS temp_mean,
    COALESCE(v.resp_rate_mean, 16.0) AS resp_rate_mean,

    -- Drug features
    COALESCE(d.drug_count, 0) AS drug_count,
    COALESCE(d.polypharmacy_score, 0) AS polypharmacy_score,
    COALESCE(d.polypharmacy_flag, 0) AS polypharmacy_flag

FROM adverse_event_cohort c
LEFT JOIN ae_lab_features l ON c.hadm_id = l.hadm_id
LEFT JOIN ae_vitals_features v ON c.icustay_id = v.icustay_id
LEFT JOIN ae_drug_features d ON c.hadm_id = d.hadm_id;
