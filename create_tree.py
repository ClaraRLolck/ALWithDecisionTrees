import numpy as np
import json
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
import sys


type = sys.argv[1]
data = sys.argv[2]

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

np.set_printoptions(threshold=sys.maxsize)
def find_path_to_node_with_directions(tree, target_node_id):
    path = []

    def recurse(node_id, current_path):
        if node_id == target_node_id:
            current_path.append((node_id, 'leaf'))
            path.extend(current_path)
            return True

        left_id = tree.children_left[node_id]
        right_id = tree.children_right[node_id]

        if left_id != -1:
            if recurse(left_id, current_path + [(node_id, 'left')]):
                return True
        if right_id != -1:
            if recurse(right_id, current_path + [(node_id, 'right')]):
                return True

        return False

    recurse(0, [])
    return path




def thresholds_with_directions(clf, target_node_id):
    tree = clf.tree_
    feature = tree.feature
    threshold = tree.threshold

    path = find_path_to_node_with_directions(tree, target_node_id)

    path_node_feature_direction = []
    for node_id, direction in path:
        path_node_feature_direction.append([node_id, feature[node_id], '>' if direction=='right' else '<='])
    return path_node_feature_direction


def calculate_list_distance_vector(id, clf, fp_len):
    path = thresholds_with_directions(clf, id)
    vector = np.zeros(1024) #len of fp
    for node in path:
        if node[1] !=-2:
            if node[2]=='>':
                vector[node[1]-1] = 1
            elif node[2] == '<=':
                vector[node[1]-1] = -1
    return vector

def calculate_distance(vector1, vector2):
    return np.sum(np.abs(vector1-vector2))

feature = clf.tree_.feature
n_nodes = clf.tree_.node_count
fp_len = len(X[0])

list_of_id = [i for i in range(n_nodes) if feature[i] == -2] 

n=len(list_of_id)

list_distance_vector = [calculate_list_distance_vector(x, clf, fp_len) for x in list_of_id] 


distance_matrix = np.empty(shape=(n, n), dtype='object')

for i in range(n):
    for j in range(n):
        if i>j:
            distance = calculate_distance(list_distance_vector[i], list_distance_vector[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

with open('Data/distanceMatrixDecisionTree_'+data+'_'+type+'.npy', 'wb') as f:
    np.save(f, list_of_id)
    np.save(f, distance_matrix)