import numpy as np
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors,MACCSkeys
from rdkit.Chem import AllChem
from rdkit import DataStructs
import os


np.set_printoptions(threshold=sys.maxsize)
def toBit(smile, desc):
  fp_arr = np.zeros((1,))
  if desc == '1024':
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, nBits=1024), fp_arr)
  elif desc == '512':
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, nBits=512), fp_arr)
  elif desc == 'MACCS':
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile)), fp_arr)
  elif desc == 'desc':
    DataStructs.ConvertToNumpyArray(Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(smile)), fp_arr)
  elif desc == 'MQN':
    DataStructs.ConvertToNumpyArray(rdMolDescriptors.MQNs_(Chem.MolFromSmiles(smile)), fp_arr)


  return fp_arr

type = sys.argv[1]
data = sys.argv[2]
with open(os.path.join( f'Data/'+data+'/actives.smi')) as f:
  new_f=open('Data/'+data+'/'+data+'_actives_smiles'+type+'.smi','a')
  for smile in f.readlines():
    new_f.write(str((smile.strip().split()[0])+":"+str((toBit(smile.strip().split()[0],type)).tolist())+"\n"))
with open(os.path.join( f'Data/'+data+'/inactives.smi')) as f:
  new_f=open('Data/'+data+'/'+data+'_inactives_smiles'+type+'.smi','a')
  for smile in f.readlines():
    new_f.write(str((smile.strip().split()[0])+":"+str((toBit(smile.strip().split()[0],type)).tolist())+"\n"))