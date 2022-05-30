# Perfectly predicting ICU length of stay: too good to be true
Code to reproduce [1]. See commentary[2]. Expected folder structure is as follows: \
  . \
  |-Data \
  |  |-MIMIC \
  |  |  |-PATIENTS.csv \
  |  |  |-ADMISSIONS.csv \
  |  |  |-DIAGNOSES_ICD.csv \
  |  |  |-ICUSTAYS.csv \
  |  |  |-D_ITEMS.csv \
  |  |  |-CHARTEVENTS.csv \
  |  |  |-PRESCRIPTIONS.csv \
  |-gen \
  |  |- \
  |-src \
  |  |-MIMIC_queries.py \
  |  |-Reproduction.ipynb 
## Data
  The MIMIC III dataset has to downloaded independently due to licensing. See https://mimic.mit.edu/docs/gettingstarted/ to get access. Extract the csv to   the Data/MIMIC folder.
## Usage
  The code was tested using Python 3.9. Run pip install -r pip_requirements.txt ,preferably in a virtual environemnt, to get the requirements.
  Run Reproduction.ipynb in jupyter step by step
## References
  [1]. Alsinglawi, B., Alshari, O., Alorjani, M. et al. An explainable machine learning framework for lung cancer hospital length of stay prediction. Sci Rep 12, 607 (2022). https://doi.org/10.1038/s41598-021-04608-7 \
  [2]. 
    
