# Generates data from adult ICUs including demographics, lab results and vital measurements

import argparse
import os
import random

import numpy as np
import pandas as pd
import psycopg2


def replace(group):
    """ Replace missing values in measurements using mean imputation
    takes in a pandas group, and replaces the null value with the mean of the none null
    values of the same group
    """
    mask = group.isnull()
    group[mask] = group[~mask].mean()
    return group


def main(sqluser, sqlpass):
    random.seed(22891)
    # Output directory to generate the files
    mimicdir = "./data/mimic/"
    if not os.path.exists(mimicdir):
        os.mkdir(mimicdir)

    # create a database connection and connect to local postgres version of mimic
    dbname = "mimic"
    schema_name = "mimiciii"
    con = psycopg2.connect(dbname=dbname, user=sqluser, host="127.0.0.1", password=sqlpass)
    cur = con.cursor()
    cur.execute("SET search_path to " + schema_name)

    # ========get the icu details

    # this query extracts the following:
    #   Unique ids for the admission, patient and icu stay
    #   Patient gender
    #   diagnosis
    #   age
    #   ethnicity
    #   admission type
    #   first hospital stay
    #   first icu stay?
    #   mortality within a week

    denquery = """
    --ie is the icustays table
    --adm is the admissions table

    SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
    , pat.gender
    , adm.admittime, adm.dischtime, adm.diagnosis
    , ROUND( (CAST(adm.dischtime AS DATE) - CAST(adm.admittime AS DATE)) , 4) AS los_hospital
    , ROUND( (CAST(adm.admittime AS DATE) - CAST(pat.dob AS DATE))  / 365, 4) AS age
    , adm.ethnicity, adm.ADMISSION_TYPE
    --, adm.hospital_expire_flag
    , CASE when adm.deathtime between ie.intime and ie.outtime THEN 1 ELSE 0 END AS mort_icu
    , DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS hospstay_seq
    , CASE
        WHEN DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) = 1 THEN 1
        ELSE 0 END AS first_hosp_stay
    -- icu level factors
    , ie.intime, ie.outtime
    , ie.FIRST_CAREUNIT
    , ROUND( (CAST(ie.outtime AS DATE) - CAST(ie.intime AS DATE)) , 4) AS los_icu
    , DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq
    , CASE
        WHEN adm.deathtime between ie.intime and ie.intime + interval '168' hour THEN 1 ELSE 0 END AS mort_week

    -- first ICU stay *for the current hospitalization*
    , CASE
        WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN 1
        ELSE 0 END AS first_icu_stay

    FROM icustays ie
    INNER JOIN admissions adm
        ON ie.hadm_id = adm.hadm_id
    INNER JOIN patients pat
        ON ie.subject_id = pat.subject_id
    WHERE adm.has_chartevents_data = 1
    ORDER BY ie.subject_id, adm.admittime, ie.intime;
    """

    den = pd.read_sql_query(denquery, con)

    ## drop patients with less than 48 hour
    den["los_icu_hr"] = (den.outtime - den.intime).astype("timedelta64[h]")
    den = den[(den.los_icu_hr >= 48)]
    den = den[(den.age < 300)]
    den.drop("los_icu_hr", 1, inplace=True)
    ## clean up
    den["adult_icu"] = np.where(den["first_careunit"].isin(["PICU", "NICU"]), 0, 1)
    den["gender"] = np.where(den["gender"] == "M", 1, 0)
    den.ethnicity = den.ethnicity.str.lower()
    den.ethnicity.loc[(den.ethnicity.str.contains("^white"))] = "white"
    den.ethnicity.loc[(den.ethnicity.str.contains("^black"))] = "black"
    den.ethnicity.loc[(den.ethnicity.str.contains("^hisp")) | (den.ethnicity.str.contains("^latin"))] = "hispanic"
    den.ethnicity.loc[(den.ethnicity.str.contains("^asia"))] = "asian"
    den.ethnicity.loc[~(den.ethnicity.str.contains("|".join(["white", "black", "hispanic", "asian"])))] = "other"

    den.drop(
        [
            "hospstay_seq",
            "los_icu",
            "icustay_seq",
            "admittime",
            "dischtime",
            "los_hospital",
            "intime",
            "outtime",
            "first_careunit",
        ],
        1,
        inplace=True,
    )

    # ========= 48 hour vitals query
    # these are the normal ranges. useful to clean up the data

    vitquery = """
    -- This query pivots the vital signs for the first 48 hours of a patient's stay
    -- Vital signs include heart rate, blood pressure, respiration rate, and temperature
    -- DROP MATERIALIZED VIEW IF EXISTS vitalsfirstday CASCADE;
    -- create materialized view vitalsfirstday as

    SELECT pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.VitalID, pvt.VitalValue, pvt.VitalChartTime

    FROM  (
        select ie.subject_id, ie.hadm_id, ie.icustay_id, ce.charttime as VitalChartTime
        , case
            when itemid in (211,220045) and valuenum > 0 and valuenum < 300 then 'HeartRate'
            when itemid in (51,442,455,6701,220179,220050) and valuenum > 0 and valuenum < 400 then 'SysBP'
            when itemid in (8368,8440,8441,8555,220180,220051) and valuenum > 0 and valuenum < 300 then 'DiasBP'
            when itemid in (456,52,6702,443,220052,220181,225312) and valuenum > 0 and valuenum < 300 then 'MeanBP'
            when itemid in (615,618,220210,224690) and valuenum > 0 and valuenum < 70 then 'RespRate'
            when itemid in (223761,678) and valuenum > 70 and valuenum < 120  then 'Temp' -- converted to degC in valuenum call
            when itemid in (223762,676) and valuenum > 10 and valuenum < 50  then 'Temp'
            when itemid in (646,220277) and valuenum > 0 and valuenum <= 100 then 'SpO2'
            when itemid in (807,811,1529,3745,3744,225664,220621,226537) and valuenum > 0 then 'Glucose'

            else null end as VitalID


            , case
            when itemid in (211,220045) and valuenum > 0 and valuenum < 300 then valuenum -- HeartRate
            when itemid in (51,442,455,6701,220179,220050) and valuenum > 0 and valuenum < 400 then valuenum -- SysBP
            when itemid in (8368,8440,8441,8555,220180,220051) and valuenum > 0 and valuenum < 300 then valuenum -- DiasBP
            when itemid in (456,52,6702,443,220052,220181,225312) and valuenum > 0 and valuenum < 300 then valuenum -- MeanBP
            when itemid in (615,618,220210,224690) and valuenum > 0 and valuenum < 70 then valuenum -- RespRate
            when itemid in (223761,678) and valuenum > 70 and valuenum < 120  then (valuenum-32)/1.8 -- TempF, convert to degC
            when itemid in (223762,676) and valuenum > 10 and valuenum < 50  then valuenum -- TempC
            when itemid in (646,220277) and valuenum > 0 and valuenum <= 100 then valuenum -- SpO2
            when itemid in (807,811,1529,3745,3744,225664,220621,226537) and valuenum > 0 then valuenum -- Glucose

            else null end as VitalValue


        from icustays ie
        left join chartevents ce
        on ie.subject_id = ce.subject_id and ie.hadm_id = ce.hadm_id and ie.icustay_id = ce.icustay_id
        and ce.charttime between ie.intime and ie.intime + interval '48' hour
        -- exclude rows marked as error
        and ce.error IS DISTINCT FROM 1
        where ce.itemid in
        (
        -- HEART RATE
        211, --"Heart Rate"
        220045, --"Heart Rate"

        -- Systolic/diastolic
        51, --	Arterial BP [Systolic]
        442, --	Manual BP [Systolic]
        455, --	NBP [Systolic]
        6701, --	Arterial BP #2 [Systolic]
        220179, --	Non Invasive Blood Pressure systolic
        220050, --	Arterial Blood Pressure systolic

        8368, --	Arterial BP [Diastolic]
        8440, --	Manual BP [Diastolic]
        8441, --	NBP [Diastolic]
        8555, --	Arterial BP #2 [Diastolic]
        220180, --	Non Invasive Blood Pressure diastolic
        220051, --	Arterial Blood Pressure diastolic

        -- MEAN ARTERIAL PRESSURE
        456, --"NBP Mean"
        52, --"Arterial BP Mean"
        6702, --	Arterial BP Mean #2
        443, --	Manual BP Mean(calc)
        220052, --"Arterial Blood Pressure mean"
        220181, --"Non Invasive Blood Pressure mean"
        225312, --"ART BP mean"

        -- RESPIRATORY RATE
        618,--	Respiratory Rate
        615,--	Resp Rate (Total)
        220210,--	Respiratory Rate
        224690, --	Respiratory Rate (Total)

        -- SPO2, peripheral
        646, 220277,

        -- GLUCOSE, both lab and fingerstick
        807,--	Fingerstick Glucose
        811,--	Glucose (70-105)
        1529,--	Glucose
        3745,--	BloodGlucose
        3744,--	Blood Glucose
        225664,--	Glucose finger stick
        220621,--	Glucose (serum)
        226537,--	Glucose (whole blood)

        -- TEMPERATURE
        223762, -- "Temperature Celsius"
        676,	-- "Temperature C"
        223761, -- "Temperature Fahrenheit"
        678 --	"Temperature F"

        )
    ) pvt
    where VitalID is not null
    order by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.VitalID, pvt.VitalChartTime;
    """
    vit48 = pd.read_sql_query(vitquery, con)
    vit48.isnull().sum()

    # ===============48 hour labs query
    # This query extracts the lab events in the first 48 hours
    labquery = """
    WITH pvt AS (
        --- ie is the icu stay
        --- ad is the admissions table
        --- le is the lab events table
        SELECT ie.subject_id, ie.hadm_id, ie.icustay_id, le.charttime as LabChartTime
        , CASE
            when le.itemid = 50868 then 'ANION GAP'
            when le.itemid = 50862 then 'ALBUMIN'
            when le.itemid = 50882 then 'BICARBONATE'
            when le.itemid = 50885 then 'BILIRUBIN'
            when le.itemid = 50912 then 'CREATININE'
            when le.itemid = 50806 then 'CHLORIDE'
            when le.itemid = 50902 then 'CHLORIDE'
            when le.itemid = 50809 then 'GLUCOSE'
            when le.itemid = 50931 then 'GLUCOSE'
            when le.itemid = 50810 then 'HEMATOCRIT'
            when le.itemid = 51221 then 'HEMATOCRIT'
            when le.itemid = 50811 then 'HEMOGLOBIN'
            when le.itemid = 51222 then 'HEMOGLOBIN'
            when le.itemid = 50813 then 'LACTATE'
            when le.itemid = 50960 then 'MAGNESIUM'
            when le.itemid = 50970 then 'PHOSPHATE'
            when le.itemid = 51265 then 'PLATELET'
            when le.itemid = 50822 then 'POTASSIUM'
            when le.itemid = 50971 then 'POTASSIUM'
            when le.itemid = 51275 then 'PTT'
            when le.itemid = 51237 then 'INR'
            when le.itemid = 51274 then 'PT'
            when le.itemid = 50824 then 'SODIUM'
            when le.itemid = 50983 then 'SODIUM'
            when le.itemid = 51006 then 'BUN'
            when le.itemid = 51300 then 'WBC'
            when le.itemid = 51301 then 'WBC'
        ELSE null
        END AS label

        , -- add in some sanity checks on the values
        CASE
            when le.itemid = 50862 and le.valuenum >    10 then null -- g/dL 'ALBUMIN'
            when le.itemid = 50868 and le.valuenum > 10000 then null -- mEq/L 'ANION GAP'
            when le.itemid = 50882 and le.valuenum > 10000 then null -- mEq/L 'BICARBONATE'
            when le.itemid = 50885 and le.valuenum >   150 then null -- mg/dL 'BILIRUBIN'
            when le.itemid = 50806 and le.valuenum > 10000 then null -- mEq/L 'CHLORIDE'
            when le.itemid = 50902 and le.valuenum > 10000 then null -- mEq/L 'CHLORIDE'
            when le.itemid = 50912 and le.valuenum >   150 then null -- mg/dL 'CREATININE'
            when le.itemid = 50809 and le.valuenum > 10000 then null -- mg/dL 'GLUCOSE'
            when le.itemid = 50931 and le.valuenum > 10000 then null -- mg/dL 'GLUCOSE'
            when le.itemid = 50810 and le.valuenum >   100 then null -- % 'HEMATOCRIT'
            when le.itemid = 51221 and le.valuenum >   100 then null -- % 'HEMATOCRIT'
            when le.itemid = 50811 and le.valuenum >    50 then null -- g/dL 'HEMOGLOBIN'
            when le.itemid = 51222 and le.valuenum >    50 then null -- g/dL 'HEMOGLOBIN'
            when le.itemid = 50813 and le.valuenum >    50 then null -- mmol/L 'LACTATE'
            when le.itemid = 50960 and le.valuenum >    60 then null -- mmol/L 'MAGNESIUM'
            when le.itemid = 50970 and le.valuenum >    60 then null -- mg/dL 'PHOSPHATE'
            when le.itemid = 51265 and le.valuenum > 10000 then null -- K/uL 'PLATELET'
            when le.itemid = 50822 and le.valuenum >    30 then null -- mEq/L 'POTASSIUM'
            when le.itemid = 50971 and le.valuenum >    30 then null -- mEq/L 'POTASSIUM'
            when le.itemid = 51275 and le.valuenum >   150 then null -- sec 'PTT'
            when le.itemid = 51237 and le.valuenum >    50 then null -- 'INR'
            when le.itemid = 51274 and le.valuenum >   150 then null -- sec 'PT'
            when le.itemid = 50824 and le.valuenum >   200 then null -- mEq/L == mmol/L 'SODIUM'
            when le.itemid = 50983 and le.valuenum >   200 then null -- mEq/L == mmol/L 'SODIUM'
            when le.itemid = 51006 and le.valuenum >   300 then null -- 'BUN'
            when le.itemid = 51300 and le.valuenum >  1000 then null -- 'WBC'
            when le.itemid = 51301 and le.valuenum >  1000 then null -- 'WBC'
        ELSE le.valuenum
        END AS LabValue


        FROM icustays ie
        LEFT JOIN labevents le
            ON le.subject_id = ie.subject_id
            AND le.hadm_id = ie.hadm_id
            AND le.charttime between (ie.intime) AND (ie.intime + interval '48' hour)
            AND le.itemid IN
            (
                -- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
                50868, -- ANION GAP | CHEMISTRY | BLOOD | 769895
                50862, -- ALBUMIN | CHEMISTRY | BLOOD | 146697
                50882, -- BICARBONATE | CHEMISTRY | BLOOD | 780733
                50885, -- BILIRUBIN, TOTAL | CHEMISTRY | BLOOD | 238277
                50912, -- CREATININE | CHEMISTRY | BLOOD | 797476
                50902, -- CHLORIDE | CHEMISTRY | BLOOD | 795568
                50806, -- CHLORIDE, WHOLE BLOOD | BLOOD GAS | BLOOD | 48187
                50931, -- GLUCOSE | CHEMISTRY | BLOOD | 748981
                50809, -- GLUCOSE | BLOOD GAS | BLOOD | 196734
                51221, -- HEMATOCRIT | HEMATOLOGY | BLOOD | 881846
                50810, -- HEMATOCRIT, CALCULATED | BLOOD GAS | BLOOD | 89715
                51222, -- HEMOGLOBIN | HEMATOLOGY | BLOOD | 752523
                50811, -- HEMOGLOBIN | BLOOD GAS | BLOOD | 89712
                50813, -- LACTATE | BLOOD GAS | BLOOD | 187124
                50960, -- MAGNESIUM | CHEMISTRY | BLOOD | 664191
                50970, -- PHOSPHATE | CHEMISTRY | BLOOD | 590524
                51265, -- PLATELET COUNT | HEMATOLOGY | BLOOD | 778444
                50971, -- POTASSIUM | CHEMISTRY | BLOOD | 845825
                50822, -- POTASSIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 192946
                51275, -- PTT | HEMATOLOGY | BLOOD | 474937
                51237, -- INR(PT) | HEMATOLOGY | BLOOD | 471183
                51274, -- PT | HEMATOLOGY | BLOOD | 469090
                50983, -- SODIUM | CHEMISTRY | BLOOD | 808489
                50824, -- SODIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 71503
                51006, -- UREA NITROGEN | CHEMISTRY | BLOOD | 791925
                51301, -- WHITE BLOOD CELLS | HEMATOLOGY | BLOOD | 753301
                51300  -- WBC COUNT | HEMATOLOGY | BLOOD | 2371
            )
        AND le.valuenum IS NOT null
        AND le.valuenum > 0 -- lab values cannot be 0 and cannot be negative

        LEFT JOIN admissions ad
        ON ie.subject_id = ad.subject_id
        AND ie.hadm_id = ad.hadm_id


    )
    SELECT pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.LabChartTime, pvt.label, pvt.LabValue
    From pvt
    where pvt.label is not NULL
    ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.label, pvt.LabChartTime;

    """

    lab48 = pd.read_sql_query(labquery, con)

    # =====combine all variables
    mort_vital = den.merge(vit48, how="left", on=["subject_id", "hadm_id", "icustay_id"])
    mort_lab = den.merge(lab48, how="left", on=["subject_id", "hadm_id", "icustay_id"])

    # create means by age group and gender
    mort_vital["age_group"] = pd.cut(
        mort_vital["age"],
        [-1, 5, 10, 15, 20, 25, 40, 60, 80, 200],
        labels=["l5", "5_10", "10_15", "15_20", "20_25", "25_40", "40_60", "60_80", "80p"],
    )
    mort_lab["age_group"] = pd.cut(
        mort_lab["age"],
        [-1, 5, 10, 15, 20, 25, 40, 60, 80, 200],
        labels=["l5", "5_10", "10_15", "15_20", "20_25", "25_40", "40_60", "60_80", "80p"],
    )

    # one missing variable
    adult_vital = mort_vital[(mort_vital.adult_icu == 1)]
    adult_lab = mort_lab[(mort_lab.adult_icu == 1)]
    adult_vital.drop(columns=["adult_icu"], inplace=True)
    adult_lab.drop(columns=["adult_icu"], inplace=True)

    adult_vital.to_csv(os.path.join(mimicdir, "adult_icu_vital.gz"), compression="gzip", index=False)
    mort_lab.to_csv(os.path.join(mimicdir, "adult_icu_lab.gz"), compression="gzip", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Query ICU mortality data from mimic database")
    parser.add_argument("--sqluser", type=str, default="mimicuser", help="postgres user to access mimic database")
    parser.add_argument(
        "--sqlpass", type=str, default="Iv7bahqu", help="postgres user password to access mimic database"
    )
    args = parser.parse_args()
    main(args.sqluser, args.sqlpass)
