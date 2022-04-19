import pandas as pd
import numpy as np
from helper import plot_symptom_stats

# -------------- Controller ----------------
load_binary_variables = True
load_symptom_frequencies = True


# -------------- Load Binary Data ----------------
data = pd.read_stata("NCSR/20240-0002-Data.dta")

if load_binary_variables:
    anxiety_symptoms = pd.read_csv("NCSR/anx_binary.txt", header=None)[0].to_list()
    depression_symptoms = pd.read_csv("NCSR/dep_binary.txt", header=None)[0].to_list()
    
else:
    anxiety_symptoms = pd.read_csv("NCSR/anx_all.txt", header=None)[0].to_list()
    depression_symptoms = pd.read_csv("NCSR/dep_all.txt", header=None)[0].to_list()

    # Find unique values for each variable in the questionnaire
    f = open("NCSR/anxiety_variables.txt", "a")
    for col in list(anxiety_symptoms):
        f.write(col+": "+str(list(pd.unique(data[col])))+"\n")
    f.close()

    f = open("NCSR/depression_variables.txt", "a")
    for col in list(depression_symptoms):
        f.write(col+": "+str(list(pd.unique(data[col])))+"\n")
    f.close()
    
    # Remove every non-binary variable from the dataset
    f = open("NCSR/anx_binary.txt", "a")
    for col in list(anxiety_symptoms):
        values = set(pd.unique(data[col]))
        if 'YES' in values and 'NO' in values:
            f.write(col+"\n")
    f.close()
    
    f = open("NCSR/dep_binary.txt", "a")
    for col in list(depression_symptoms):
        values = set(pd.unique(data[col]))
        if 'YES' in values and 'NO' in values:
            f.write(col+"\n")
    f.close()        


# -------------- Calculate frequency of each binary variable ---------------- 
if load_symptom_frequencies:
    anx_freqs = pd.read_csv('NCSR/anx_freqs.csv').to_numpy()
    dep_freqs = pd.read_csv('NCSR/dep_freqs.csv').to_numpy()
    
else:
    anx_freqs = np.zeros((3, len(anxiety_symptoms)))  
    for i_symp, anx_symp in enumerate(anxiety_symptoms):
        freqs = data[anx_symp].value_counts(normalize=True, dropna=False)
        anx_freqs[0,i_symp] = freqs['YES']
        anx_freqs[1,i_symp] = freqs['NO']
        anx_freqs[2,i_symp] = 1-freqs['YES']-freqs['NO']
    anx = pd.DataFrame(anx_freqs, index=['YES','NO','NaN'], columns=anxiety_symptoms)
    anx.to_csv('NCSR/anx_freqs.csv')

    dep_freqs = np.zeros((3, len(depression_symptoms)))  
    for i_symp, dep_symp in enumerate(depression_symptoms):
        freqs = data[dep_symp].value_counts(normalize=True, dropna=False)
        dep_freqs[0,i_symp] = freqs['YES']
        dep_freqs[1,i_symp] = freqs['NO']
        dep_freqs[2,i_symp] = 1-freqs['YES']-freqs['NO']
    dep = pd.DataFrame(dep_freqs, index=['YES','NO','NaN'], columns=depression_symptoms)
    dep.to_csv('NCSR/dep_freqs.csv')  
    

# -------------------- Select symptoms of interests ---------------------
# Currently feature selection solely based on its frequency
# TODO: feature-selection that rely on more informative / reliable criteria

anx_SOI_idx = anx_freqs[2,1:]<0.8
dep_SOI_idx = anx_freqs[2,1:]<0.8
anx_SOI = [anxiety_symptoms[i] for i in range(len(anx_SOI_idx)) if anx_SOI_idx[i]]
dep_SOI = [depression_symptoms[i] for i in range(len(dep_SOI_idx)) if dep_SOI_idx[i]]

# -------------------- Select samples based on symptoms of interest ----------------

data_anx = data[anx_SOI]
data_anx = data_anx.dropna()
data_anx = data_anx.to_numpy() == 'YES'
np.savetxt('X.csv', data_anx.astype('int'))

'''
data_dep = data[dep_SOI]
data_dep = data_dep.dropna()
print(data_dep)
'''