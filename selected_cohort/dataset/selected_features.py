# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import numpy as np

class SMM_feature():
    def __init__(self, categorical_vars, continuous_vars, cat_length, cot_length, 
                 feature_name=None, cat_list=None):
        self.categorical_vars = categorical_vars
        self.continuous_vars = continuous_vars

        self.cat_length = cat_length
        self.cot_length = cot_length

        self.hospital_features_name = ['LEVEL', 'SYSNAME', 'MAPP8', 'OBBD', 'LBEDSA', 'OPRA', 'NICHOS', 'ADMTOT', 
                                       'MCDIPD', 'LINGHOS', 'FTTOT', 'FTMTF', 'FTRNTF', 'TEOB', 'TPOB', 'CMRPAY', 
                                       'FAMADV', 'CNTYNAME', "FTMDTF", "HOS_DELI_Volume", "HOS_DELI_Medicaid", 
                                       "HOS_DELI_Minority"]
        self.feature_name = feature_name
        self.cat_list = cat_list

def get_feature_name(categorical_vars, continuous_vars, cat_list):
    assert len(categorical_vars) == len(cat_list), f"ERROR: different len(categorical_vars)={len(categorical_vars)} with len(cat_list)={len(cat_list)}"
    feature_name = []
    cat_list = np.array(cat_list)
    cat_list = np.ceil(np.log(cat_list + 1) / np.log(2)).astype(np.int)
    for (cat_name, cat_cnt) in zip(categorical_vars, cat_list):
        for j in range(cat_cnt):
            feature_name.append(cat_name + f"_{j}")
    for cot_name in continuous_vars:
        feature_name.append(cot_name)

    return feature_name

def get_SMM_features(mode: str='default'):
    # categorical count for each category
    cats_length_65_11 = [4, 7, 3, 3, 5, 2, 5, 5, 32, 4, 10, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 32, 17, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    cats_length_16_2 = [4, 7, 3, 5, 2, 5, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2]
    cat_list = None
    if mode == 'Categorical_65+Continuous_11' or mode == 'default':
        # Original total 75 + ["comorbidities_delivery"]
        categorical_vars = [
            "YEAR", "AGE_CAT", "PRIMLANG_CAT", "MARITAL_CAT", "PAY_CAT", "Homeless", 
            "MEDINCSTQ_CAT", "UR_CAT", "PSTCO", "LEVEL", "SYSNAME", "MAPP8", "NICHOS", 
            "LINGHOS", "CMRPAY", "FAMADV", "Encounter_Cat_past30days", "SMOKING_delivery",
            "OTHERSUBSTANCE_delivery", "OBESITY_delivery", "MULTIPLE_GESTATION_delivery",
            "ATYPE", "TRANSFER", "NEOPLASM_delivery", "BLOOD_delivery", "ENDOCRINE_delivery",
            "MENTAL_delivery", "RHEUMATIC_delivery", "HYPERTENSIVE_delivery", "ISCHEMIC_delivery",
            "PULMONARY_delivery", "OTHERHEART_delivery", "CEREBROVASCULAR_delivery", "ARTERY_delivery",
            "VEIN_delivery", "OTHERCIRCULATORY_delivery", "CHRONIC_delivery", "LIVER_delivery",
            "RENAL_delivery", "POISONING_NARCOTICS_delivery", "POISONING_OTHER_delivery", 
            "EXTERNAL_MORBIDITY_delivery", "HIGHRISK_delivery", "HEMORRHAGE_delivery",
            "EXCESSIVE_delivery", "PREDIABETES_delivery", "GESTATIONAL_delivery", "MALNUTRITION_delivery",
            "OTHERCOMPLICATION_delivery", "FETUS_delivery", "RACE_CAT", "HOSPNAME", "CNTYNAME",
            "HIGHRISK_past30days", "HYPERTENSION_past30days", "HEMORRHAGE_past30days", 
            "EXCESSIVE_past30days", "VENOUS_past30days", "GENITOURINARY_past30days", 
            "PREDIABETES_past30days", "GESTATIONAL_past30days", "MATERNAL_past30days", 
            "ABNORMAL_past30days", "FETUS_past30days", "comorbidities_delivery"
        ]
        # Original 11
        continuous_vars = [
            "OBBD", "LBEDSA", "OPRA", "ADMTOT", "MCDIPD", "FTTOT", "FTMDTF", "FTRNTF", 
            "HOS_DELI_Volume", "HOS_DELI_Medicaid", "HOS_DELI_Minority"
        ]
        cat_length, cot_length = [152, 11]
        cat_list = cats_length_65_11
    elif mode == '16+2':
        # cat vars and cot vars
        categorical_vars = [
            'YEAR', 'AGE_CAT', 'PRIMLANG_CAT', 'PAY_CAT', 'Homeless', 'MEDINCSTQ_CAT',
            'LEVEL', 'MAPP8', 'CMRPAY', 'FAMADV', 'Encounter_Cat_past30days', 'OBESITY_delivery',
            'MULTIPLE_GESTATION_delivery', 'HIGHRISK_past30days', 'HYPERTENSION_past30days',
            'comorbidities_delivery'
        ]
        continuous_vars = [
            "HOS_DELI_Volume", "HOS_DELI_Minority"
        ]
        cat_length, cot_length = [38, 2]
        cat_list = cats_length_16_2

    else:
        assert False, 'Error: mode={}'.format(mode)

    feature_name = get_feature_name(categorical_vars, continuous_vars, cat_list)
    return SMM_feature(categorical_vars, continuous_vars, cat_length, cot_length, 
                       feature_name=feature_name, cat_list=cat_list)