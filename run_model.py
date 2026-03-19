import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
import pandas as pd
import json

import os

np.set_printoptions(threshold=sys.maxsize)

def dataPrep(startTrain, restTrain):
  x_train = pd.DataFrame(list(startTrain['desc']), index = startTrain['desc'].index)
  y_train = startTrain['Active']
  x_val = pd.DataFrame(list(restTrain['desc']), index = restTrain['desc'].index)
  return (x_train, y_train, x_val)

def predict(X, regr, acq_model):
    preds = np.zeros((len(X), len(regr.estimators_)))
    for j, submodel in enumerate(regr.estimators_):
        preds[:, j] = submodel.predict(X)
        y_pred = np.mean(preds, axis=1) 
    sigma = np.std(preds, axis=1)
    if acq_model == "greedy":
       return y_pred 
    elif acq_model == "UCB":
        a =(y_pred+(2*sigma))
        return a
    elif acq_model == "opp_UCB":
        a =(y_pred-(2*sigma))
        return a


def modelrun(x_train, y_train, x_val, model, acq_model):
    if model == "RF":
        regr = RandomForestRegressor()
        regr.fit(x_train, y_train)
        a = pd.Series(predict(x_val, regr, acq_model), name="Predicted")
        return a
    elif model == "NN":
        model = Sequential([
            Dense(len(x_train[0])//2, activation='sigmoid', input_shape = (x_train.shape[1],)),
            BatchNormalization(axis=1),
            Dense(1, activation='sigmoid'),
        ])

        #Compile the model
        model.compile(
            optimizer = 'adam',
            loss = 'mean_squared_error',
            metrics=['accuracy']
        )

        #Train the model
        model.fit(
            x_train,
            y_train,
            epochs=16,
      batch_size=8,
      verbose=0
    )
    return pd.Series(sum(model.predict(x_val).tolist(),[]),name="Predicted")
       

def sort_split_data(data, predicted, n):
  df = pd.concat([(data).reset_index(drop=True),predicted], axis=1)
  df = df.sort_values(by=['Predicted'],ascending=False)
  return (df.iloc[:n,:][['Smile','desc','Active']],df.iloc[n:,:][['Smile','desc','Active']])

def predicted_true(train_set):
  return train_set['Active'].value_counts()[1]

def train_run_model(start_n, n_add, iter, data, model = "RF", acq_model="greedy" ):
  start_molecules_file = 'Data/'+data+'/Start_mols/'+data+'file'+str(start_n)+'_10startMol.txt' 

  with open(os.path.join( start_molecules_file)) as f:
    start_mols=[lines.split(';')[0] for lines in f.readlines()[1:]]
  list_index =[all_mol[all_mol['Smile'] == mol].index[0] for mol in start_mols]
  startTrain = all_mol.iloc[list_index]

  restTrain = all_mol.drop(index=list_index)
  n_iter = iter

  x_train, y_train, x_val = dataPrep(startTrain, restTrain)

  correct_predicted = []
  correct_predicted.append(int(predicted_true(startTrain)))

  for _ in range(n_iter):
    y_predict = modelrun(x_train, y_train, x_val, model, acq_model)
    new_start,restTrain = sort_split_data(restTrain, y_predict, n_add)
    startTrain = pd.concat([startTrain,new_start])
    x_train, y_train, x_val = dataPrep(startTrain, restTrain)
    correct_predicted.append(int(predicted_true(startTrain)))
  new_f = open('Results/'+ data +'_'+type+'_'+str(n_add)+'add_'+model+'_'+acq_model+'.txt','a')
  new_f.write('startmol.'+str(start_n)+' '+str(correct_predicted)+': '+str(startTrain['Smile'].tolist())+'\n')


from multiprocessing import Pool

type = sys.argv[1]
data = sys.argv[2]

acc_size = sys.argv[3]
n_iter = 100//acc_size

with open(os.path.join( f'Data/'+data+'/'+data+'_actives_smiles'+type+'.smi')) as f:
  actives=pd.DataFrame([[lines.split(':')[0],pd.Series(json.loads(lines.split(':')[1])),1] for lines in f], columns=['Smile','desc','Active'])
with open(os.path.join( f'Data/'+data+'/'+data+'_inactives_smiles'+type+'.smi')) as f:
  inactives=pd.DataFrame([[lines.split(':')[0],pd.Series(json.loads(lines.split(':')[1])),0] for lines in f], columns=['Smile','desc','Active'])


all_mol = pd.concat([actives,inactives], ignore_index=True)

if __name__ == '__main__':
  with Pool(1) as p:
    p.starmap(train_run_model, [(i, acc_size, n_iter, data) for i in range(10)])


