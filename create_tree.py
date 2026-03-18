import numpy as np
import json
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from joblib import dump



data = 'ALDH1'
type = '1024'

with open(os.path.join( f'Data/'+data+'/'+data+'_actives_smiles'+type+'.smi')) as f:
  actives=pd.DataFrame([[lines.split(':')[0],json.loads(lines.split(':')[1]),1] for lines in f], columns=['Smile','desc','Active'])
with open(os.path.join( f'Data/'+data+'/'+data+'_inactives_smiles'+type+'.smi')) as f:
  inactives=pd.DataFrame([[lines.split(':')[0],json.loads(lines.split(':')[1]),0] for lines in f], columns=['Smile','desc','Active'])


all_mol = pd.concat([actives,inactives], ignore_index=True)
all_mol["desc"] = all_mol["desc"].apply(lambda x: np.array(x)) 
X = pd.DataFrame(all_mol["desc"].tolist())  
y = all_mol["Active"]


clf = DecisionTreeClassifier()
clf.fit(X, y)
dump(clf, 'Data/decision_tree_'+data+'_'+type+'.joblib')