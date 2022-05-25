# Author : Sandeep Ramachandra (sandeep.ramachandra@ugent.be)
# File contains queries to produce lung cancer patients and their medical info from MIMIC III dataset

import pandas as pd
import os

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import tqdm
import datetime as dt
from pprint import pprint
import re

def get_list_subjects(patient_csv :str, admission_csv : str, icu_csv : str, diagnosis_csv : str) -> pd.DataFrame:
    """
    get a filtered list of subjectids, admission ids and icu ids given the mimic csv file paths 
    patient_csv: str, path to PATIENTS.csv file of MIMIC III
    admission_csv: str, path to ADMISSIONS.csv file of MIMIC III
    icu_csv: str, path to ICUSTAYS.csv file of MIMIC III
    diagnosis_csv: str, path to ADMISSIONS.csv file of MIMIC III
    """
    adm_df = pd.read_csv(admission_csv)
    # filter out patient who did not die during admission
    sl = adm_df[adm_df.HOSPITAL_EXPIRE_FLAG == 0]
    # look for cancer in the admission DIAGNOSIS and filter for those admissions
    sl = sl[sl.DIAGNOSIS.str.contains('cancer',case=False,regex=False).fillna(False)].HADM_ID
    # note all rows in icu_df have all subject_id, hadm_id and diagnosis_id so that step is skipped
    icu_df = pd.read_csv(icu_csv)
    diag = pd.read_csv(diagnosis_csv)
    # For selected admissions, find mentions of lung cancer
    diag = diag[diag.HADM_ID.isin(sl)]
    filt = diag.dropna().groupby('SUBJECT_ID', as_index = True)
    def find_lung_cancer(df):
        for i,item in enumerate(df.ICD9_CODE.to_list()):
            if str(item).startswith('162'): # diagnosis is lung cancer
                return True
        return False
    
    s = filt.filter(find_lung_cancer)
    # return necessary info
    los = icu_df[icu_df.SUBJECT_ID.isin(s.SUBJECT_ID)][['SUBJECT_ID','HADM_ID','ICUSTAY_ID','LOS','INTIME']]
    return los
    
    
def get_medications(list_icu_id : list, prescriptions_csv : str = "../Data/MIMIC/PRESCRIPTIONS.csv") -> dict:
    '''
    Get the binary classification if patients took drug from preselected drug list
    Use list_icustay_id to filter the prescription_csv for patient info
    list_icu_id: list of icu ids for one patient
    prescription_csv: path of PRESCRIPTIONS.csv file of MIMIC III
    returns a dictionary with key of drug name and 1 if patient took said drug else 0
    '''
    meds = pd.read_csv(prescriptions_csv, usecols=['SUBJECT_ID','ICUSTAY_ID','DRUG'], )
    
    sub_meds = meds[meds.ICUSTAY_ID.isin(list_icu_id)]
    sel_meds = ["amiodarone","ampicillinsulbactam","atropine","calciumgluconate","carvedilol","cefazolin",
                "cefepime","ceftriaxone","clonazepam","clopidogrel","dextrose","diazepam","digoxin","diltiazem",
                "diphenhydramine","enoxaparin","fentanyl","fentanylcitrate","fluconazole","fondaparinux","furosemide",
                "glucagon","haloperidol","heparin","hydralazine","hydromorphone","insulin","levofloxacin","levothyroxine",
                "metoclopramide","metoprolol","metronidazole","midazolam","nitroglycerin","nitroprusside","norepinephrine",
                "ondansetron","phenytoin","piperacillin","potassium","prednisone","propofol","vancomycin"]
    # remove spaces and make all lower case
    sub_meds = sub_meds.DRUG.str.replace(' ','').str.lower()
    list_meds = sub_meds.to_list()
    res = {}
    for a in sel_meds:
        if any(a in x for x in list_meds): # if x is part of any medication in the list of medications given to patient 
            res[a+'_y'] = 1
        else:
            res[a+'_y'] = 0
    return res


def get_general_info(subject_id : int, hadm_id : int,
                     admission_csv : str = "../Data/MIMIC/ADMISSIONS.csv", 
                     patient_csv : str = "../Data/MIMIC/PATIENTS.csv") -> dict:
    '''
    get gender, age and admission type info regarding a single patient 'subject_id' from admission_csv and patient_csv
    subject_id: patient id from MIMIC III obtained after filtering
    hadm_id: admission id from MIMIC III obtained after filtering
    admission_csv: str, path to ADMISSIONS.csv file of MIMIC III
    patient_csv: str, path to PATIENTS.csv file of MIMIC III
    '''
    patient = pd.read_csv(patient_csv)
    # get gender, date of birth from patient file
    patient = patient[patient.SUBJECT_ID == subject_id]
    gender = 1 if patient.GENDER.values == 'M' else 0
    dob = patient.DOB.values
    dob = [dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S") for x in dob][0]
    admission = pd.read_csv(admission_csv)
    # get admit time from admissions and calculate age
    admission = admission[admission.HADM_ID == hadm_id]
    admit_time = admission.ADMITTIME.values
    admit_time = dt.datetime.strptime(admit_time[0],"%Y-%m-%d %H:%M:%S")
    age = (admit_time - dob).total_seconds()/(365*24*3600)
    # categorise patient as senior citizen if age > 60
    age_senior = 1 if age > 60 else 0
    # get admission type 
    admit_type_dict = {"ELECTIVE":0,"EMERGENCY":0,"URGENT":0}
    admit_type = admission.ADMISSION_TYPE.values[0]
    admit_type_dict[admit_type] += 1 # remainder from when there were multiple admission allowed for each patient, right now if it aint broke
    res = {
        "gender":gender,
        "age_senior":age_senior,
        **admit_type_dict
    }
    return res


def verify_carevue_metavision(l : list) -> bool:
    '''
    Function to check if each feature list has terms from both carevue and metavision database. 
    (MIMIC III has a combination of the two icu databases and they have different codes for each labs)
    This ensures that each features has atleast one code for each database (One lab can have multiple names even within the same database)
    '''
    carevue = False
    metavision = False
    for a in l:
        # carevue codes are all below 22000 and metavision are all above 220000
        if a[0]<22000:
            carevue=True
        elif a[0]>=22000:
            metavision=True
    return (carevue and metavision)


pb = ProgressBar()
pb.register()

def get_itemid_from_features(d_items_csv : str = "../Data/MIMIC/D_ITEMS.csv") -> dict:
    """
    get a dictionary of lists with the features collecting all related itemids in a list
    d_items_csv: path to D_ITEMS.csv file of MIMIC III
    """
    # features = {"RBC":"(?<!P)RBC", # Strict filtering for labs
    #             "WBC":"WBC",
    #             "Platelets":"Platelets",
    #             "Hemoglobin":"Hemoglobin",
    #             "Hematocrit":"(Hematocrit)|(Hematocrit \(whole blood - calc\))",
    #             "Bands":"(?<!L&D )Bands",
    #             "Neutrophils":"Neuts",
    #             "Temperature F":"Temperature F",
    #             "Heart rate":"Heart rate",
    #             "Respiratory rate":"Respiratory rate",
    #             "Blood pressure systolic":"blood pressure Systolic$",
    #             "Blood pressure diatolic":"blood pressure Diastolic$",
    #             "Pulseoxymetry":"Pulseoxymetry",
    #             "troponin":"troponin",
    #             "BUN":"BUN",
    #             "INR":"INR",
    #             "PTT":"PTT",
    #             "creatinine":"creatinine",
    #             "glucose":"(glucose)|(glucose \(whole blood\))",
    #             "sodium":"(sodium)|(sodium \(whole blood\))",
    #             "potassium":"(potassium)|(potassium \(whole blood\))",
    #             "chloride":"(chloride)|(chloride \(whole blood\))",
    #             "PEEP set":"PEEP set",
    #             "tidal volume":"^tidal volume",
    #             "anion gap":"anion gap",
    #             "O2 Fraction":"O2 Fraction"}
    features = {"RBC":"(?<!P)RBC", # lax filtering for labs
                "WBC":"WBC",
                "Platelets":"Platelets",
                "Hemoglobin":"Hemoglobin",
                "Hematocrit":"Hematocrit",
                "Bands":"(?<!L&D )Bands",
                "Neutrophils":"Neuts",
                "Temperature F":"Temperature F",
                "Heart rate":"Heart rate",
                "Respiratory rate":"Respiratory rate",
                "Blood pressure systolic":"blood pressure Systolic$",
                "Blood pressure diatolic":"blood pressure Diastolic$",
                "Pulseoxymetry":"Pulseoxymetry",
                "troponin":"troponin",
                "BUN":"BUN",
                "INR":"INR",
                "PTT":"PTT",
                "creatinine":"creatinine",
                "glucose":"glucose",
                "sodium":"sodium",
                "potassium":"potassium",
                "chloride":"chloride",
                "PEEP set":"PEEP set",
                "tidal volume":"^tidal volume",
                "anion gap":"anion gap",
                "O2 Fraction":"O2 Fraction"}
        
    d_items = pd.read_csv(d_items_csv,usecols=['ITEMID','LABEL','LINKSTO'])
    # select from chartevent items only since the features will be seen from there
    d_items = d_items[d_items.LINKSTO == 'chartevents']
    d_items = d_items.drop('LINKSTO', axis = 1)
    
    item_ids = {}
    for a in features:
        # look for exact matches for each feature with atleast one carevue match and one metavision match
        tmp = list(d_items[d_items.LABEL.str.fullmatch(features[a], case = False).fillna(False)].itertuples(index=False, name=None))
        if not verify_carevue_metavision(tmp):
            # if one from both databases are not found, then use a little relaxed matching
            tmp = list(d_items[d_items.LABEL.str.contains(features[a], flags = re.IGNORECASE, regex=True).fillna(False)].itertuples(index=False, name=None))
        item_ids[a] = tmp

    return item_ids

def get_all_labs(list_icu_id : list, 
                 feature_dict : dict,
                 csv_file : str = "../Data/MIMIC/CHARTEVENTS.csv",
                 usecols : list= ['ICUSTAY_ID','ITEMID','VALUE','VALUENUM','VALUEUOM'],
                 dtype : dict = {"ICUSTAY_ID":'float64','ITEMID':'float64','VALUE':'object','VALUENUM':'float64','VALUEUOM':'object'},
                 output_file : str = "../gen/reduced_chartevents.csv") -> pd.DataFrame:
    """
    chartevents is a giant csv file so we reduce the csv using dask and then read into pandas dataframe.
    list_icu_id is a list of all icuids in the filtered patient list
    feature_dict is the dictionary of all relevant itemids for all features
    csv_file is the path to chartevents.csv
    usecols is a list of column names to load selected columns from csv
    dtype is a dictionary with column names as key and their data types (not needed to be changed)
    output_file is name of generated(reduced) file which the function will check for before loading the csv
    """
    if os.path.exists(output_file): # reduced file is already found
        return pd.read_csv(output_file)
    charts = dd.read_csv(csv_file, usecols = usecols, dtype=dtype)
    # reduce to list of icu ids of all patients
    charts = charts[charts.ICUSTAY_ID.isin(list_icu_id)]
    # reduce to list of feature names
    feature_list = [i[0] for v in feature_dict.values() for i in v]
    charts = charts[charts.ITEMID.isin(feature_list)].compute()
    # save file for future use directly
    charts.to_csv(output_file)
    return charts

def get_lab_info(list_icu_id : list, charts_panda_file : pd.DataFrame, d_items_csv : str = "../Data/MIMIC/D_ITEMS.csv"):
    '''
    get lab info for a single patient's icu 'icu_id' from labs_csv 
    list_icu_ids: list of icu stays of one patient 
    charts_pandas_file: panda csv obtained by reducing chartevents.csv in get_all_labs function
    d_labs_csv: path to D_ITEMS.csv in MIMIC III to convert the labids in labs_csv to human understandable tags
    '''
    # get list of features
    d_items = get_itemid_from_features(d_items_csv)
    # reduce to single patient info
    charts = charts_panda_file[charts_panda_file.ICUSTAY_ID.isin(list_icu_id)]
    
    res={}
    # get mean (single value) for all lab results found
    for feature, items in d_items.items():
        list_itemids = [a[0] for a in items]
        measurements = charts[charts.ITEMID.isin(list_itemids)].VALUENUM
        if not measurements.empty:
            res[feature] = measurements.mean()

    return res
    
def main(root_dir : str = "../Data/MIMIC/") -> dict:
    '''
    Main function to return a dictionary of all needed patient info after filtering MIMIC III. 
    Can be read into pandas by pandas.DataFrame.from_records(dict).T()
    root_dir: path to the extracted MIMIC III data
    '''
    subjects = get_list_subjects(root_dir+"PATIENTS.csv",root_dir+"ADMISSIONS.csv",root_dir+"ICUSTAYS.csv",root_dir+"DIAGNOSES_ICD.csv")
    filtered_subjects = subjects.sort_values('INTIME').drop_duplicates(subset='SUBJECT_ID',keep='last')
    d_items = get_itemid_from_features(root_dir+"D_ITEMS.csv")
    charts = get_all_labs(subjects.ICUSTAY_ID.to_list(), d_items, root_dir+'CHARTEVENTS.csv')
    
    res = {}
    
    for _, subject in tqdm.tqdm(filtered_subjects['SUBJECT_ID'].iteritems(),
                                                   total = len(filtered_subjects)):
        # we only want the general patient info for the last admission so we use filtered subjects which has droped all except the last ones
        hadm_id = filtered_subjects[filtered_subjects.SUBJECT_ID == subject].HADM_ID.item()
        los = filtered_subjects[filtered_subjects.SUBJECT_ID == subject].LOS.item()
        general_info = get_general_info(subject, hadm_id, root_dir+"ADMISSIONS.csv", root_dir+"PATIENTS.csv" )
        
        # medications and labs are taken for all icu stays
        icu_id = subjects[subjects.SUBJECT_ID == subject].ICUSTAY_ID.to_list()
        meds_info = get_medications(icu_id, root_dir+"PRESCRIPTIONS.csv")
        labs_info = get_lab_info(icu_id, charts, root_dir+"D_ITEMS.csv")
        
        # the features percentage present in the patient info (for thresholding purposes)
        features = d_items.keys()
        feature_pc = sum([x in labs_info for x in features])/len(features)
        
        # if feature_pc < 0.7:# uncomment to enable thresholding
        #     continue
        # NOTE !!! feature_pc is passed but is not needed as a feature for the model. Having said this, it is surprisingly predictive for the LOS prediction task mentioned.
        res[subject] = {'feature_pc': feature_pc,'los':1 if los > 7 else 0, **general_info, **meds_info, **labs_info}
        
    return res
    
if __name__ == "__main__":
    print(main())
