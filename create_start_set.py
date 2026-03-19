import numpy as np
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os


type = sys.argv[1]
data = sys.argv[2]
np.set_printoptions(threshold=sys.maxsize)
with open(os.path.join( f'Data/'+data+'/'+data+'_actives_smiles'+type+'.smi')) as f:
    actives=pd.DataFrame([[lines.split(':')[0],pd.Series(json.loads(lines.split(':')[1])),1] for lines in f], columns=['Smile','FingerPrint','Active'])
with open(os.path.join( f'Data/'+data+'/'+data+'_inactives_smiles'+type+'.smi')) as f:
    inactives=pd.DataFrame([[lines.split(':')[0],pd.Series(json.loads(lines.split(':')[1])),0] for lines in f], columns=['Smile','FingerPrint','Active'])

def start_and_rest_set():
  for i in range(10):
    _, startTrainActive = train_test_split(actives, test_size=1)
    _, startTrainInactive = train_test_split(inactives, test_size=9) 
    startTrain = pd.concat([startTrainActive,startTrainInactive])

    startTrain[['Smile','Active']].to_csv('Data/'+data+'/Start_mols/'+data+'file'+str(i)+'_10startMol.txt', sep=';', index=False)

start_and_rest_set()