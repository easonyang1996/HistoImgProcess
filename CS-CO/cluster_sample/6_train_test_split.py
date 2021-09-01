import numpy as np
import os
import pandas as pd 

MAG = '0.5_um'
PATH = '../../roi_features/{}/'.format(MAG)
CLINICAL = '../../lihc_clinical.csv'

#process clinical data
df = pd.read_csv(CLINICAL)
df.set_index('bcr_patient_barcode', inplace=True)
x = df[['PFI', 'PFI.time']].dropna(axis=0)      # TCGA-2V-A95S

print(x.head())

nan_patient = 'TCGA-2V-A95S'

# get slide information and roi feature list
feature_list = os.listdir(PATH+'all/')
slide_dict = {}
for item in feature_list:
    name = item.split('_')[0][:12]
    if name != nan_patient:
        if name not in slide_dict.keys():
            slide_dict[name] = [item]
        else:
            slide_dict[name].append(item)

#print(slide_dict)



np.save(PATH+'slide_dict.npy', slide_dict)
x.to_csv(PATH+'clinical.csv')
