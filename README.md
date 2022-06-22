# Perfectly predicting ICU length of stay: too good to be true
Code to reproduce [1]. See commentary[2]. \
## Abstract
A paper of Alsinglawi et al.[1] was recently accepted and published in Scientific Reports. In this paper, the authors aim to
predict length of stay (LOS), discretized into either long (> 7 days) or short stays (≤ 7 days), of lung cancer patients in an ICU
department using various machine learning techniques. The authors claim to achieve perfect results with an Area Under the
Receiver Operating Characteristic curve (AUROC) of 100% with a Random Forest classifier with ADASYN class balancing
over sampling, which if accurate could have significant implications for hospital management. However, we have identified
several methodological flaws within the manuscript which cause the results to be overly optimistic and would have serious
consequences if used in a clinical practice. Moreover, the reporting of the methodology is unclear and many important details are
missing from the manuscript, which makes reproduction extremely difficult. In the paper, we have tried to reproduce the dataset and run a RF classifier on the data to identify the effect of these methodological flaws on the classfier's performance.
## Data
  The MIMIC III dataset has to downloaded independently due to licensing. See https://mimic.mit.edu/docs/gettingstarted/ to get access. Extract the csv to   the Data/MIMIC folder. Required csv files are : PATIENTS.csv, ADMISSIONS.csv, DIAGNOSES_ICD.csv, ICUSTAYS.csv, D_ITEMS.csv, CHARTEVENTS.csv, PRESCRIPTIONS.csv.
## Usage
  The code was tested using Python 3.8.10. Run pip install -r pip_requirements.txt ,preferably in a virtual environemnt, to get the requirements.
  Run Reproduction.ipynb in jupyter step by step
## References
  [1]. Alsinglawi, B., Alshari, O., Alorjani, M. et al. An explainable machine learning framework for lung cancer hospital length of stay prediction. Sci Rep 12, 607 (2022). https://doi.org/10.1038/s41598-021-04608-7 \
  [2]. Placeholder for paper
    
