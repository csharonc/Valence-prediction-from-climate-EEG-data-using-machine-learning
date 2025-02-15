# Lists for feature selection

#Based on wave frequency and location

import pandas as pd

features = pd.read_csv("../features_tweaked.csv")


metadata_columns = ['Respondent', 'Participant_ID', 'mean_rating', 'Participant_ID_short',
       'image_number', 'individual_ratings', 'ID', 'Type', "class", "Missing", "FAA"]


frontal_groups = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"]
central_groups = ["Cz", "C3", "C4", "T3", "T4", "T5", "T6"]
parietal_groups = ["P3", "P4", "Pz", "POz", "O1", "O2"]

_frontal = []
_central = []
_parietal = []

_theta = [] #4-7
_alpha = [] #8-12
_beta = [] #13-30
_low_gamma = [] #31-50
_high_gamma = [] #51-80

for col in features.columns:
    if col not in metadata_columns:
        if col[:2] in frontal_groups or col[:3] in frontal_groups:
            _frontal.append(col)
        elif col[:2] in central_groups:
            _central.append(col)
        elif col[:2] in parietal_groups or col[:3] in parietal_groups:
            _parietal.append(col)


for col in features.columns:
    if col not in metadata_columns: 
        if col[-4] == "_":
            if int(col[-3]) in range(4,8):
                _theta.append(col)
            elif col[-3] == "9":
                _alpha.append(col)
        else:
            if int(col[-4:-2]) in range(10, 13):
                _alpha.append(col)
            elif int(col[-4:-2]) in range(13, 31):
                _beta.append(col)
            elif int(col[-4:-2]) in range(31, 51):
                _low_gamma.append(col)
            elif int(col[-4:-2]) in range(51, 81):
                _high_gamma.append(col)