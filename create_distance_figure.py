from joblib import load
import numpy as np
import os
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem import rdFingerprintGenerator
import itertools
import ast
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import random
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

np.set_printoptions(threshold=sys.maxsize)

data = 'ALDH1'
desc = '1024'
n_add= 2
model = 'RF'
acq_model = 'greedy'
mol = 5

clf = load("Data/decision_tree_" + data + "_" + desc + ".joblib")
with open('Data/distanceMatrixDecisionTree_'+data+'_'+desc+'.npy', 'rb') as f:
    list_id = np.load(f, allow_pickle=True)
    distance_matrix = np.load(f, allow_pickle=True)
with open(os.path.join( f'Data/'+data+'/'+data+'_actives_smiles'+desc+'.smi')) as f:
  actives=pd.DataFrame([[lines.split(':')[0],json.loads(lines.split(':')[1]),1] for lines in f], columns=['Smile','desc','Active'])
with open(os.path.join( f'Data/'+data+'/'+data+'_inactives_smiles'+desc+'.smi')) as f:
  inactives=pd.DataFrame([[lines.split(':')[0],json.loads(lines.split(':')[1]),0] for lines in f], columns=['Smile','desc','Active'])

all_mol = pd.concat([actives,inactives], ignore_index=True)

all_mol["desc"] = all_mol["desc"].apply(lambda x: np.array(x)) 
X = pd.DataFrame(all_mol["desc"].tolist())  
y = all_mol["Active"]

def find_closest_leaf(target_id, number_of_leafs, only_active=True, list_id = list_id,clf = clf, distance_matrix = distance_matrix):

    list_id = np.array(list_id)
    values = clf.tree_.value

    if target_id not in list_id:
        raise ValueError(f"Target ID {target_id} not found in list_id.")

    target_index = np.where(list_id == target_id)[0][0]
    distances = distance_matrix[target_index]

    if only_active:
        actives = [
            (i, distances[i]) 
            for i in range(len(list_id))
            if list_id[i] != target_id and values[list_id[i]][0][1] > 0.0
        ]
    else:
        actives = [
            (i, distances[i]) 
            for i in range(len(list_id)) 
            if list_id[i] != target_id
        ]

    # Sort by distance
    actives.sort(key=lambda x: x[1])

    # Take the top closest
    closest_indices = [i for i, _ in actives[:number_of_leafs]]
    closest_leaf_ids = list_id[closest_indices].tolist()
    
    return closest_leaf_ids

def find_distance_index(id1, id2, list_id = list_id, distance_matrix = distance_matrix):
    index1 = np.where(list_id==id1)[0][0]
    index2 = np.where(list_id==id2)[0][0]
    return distance_matrix[index1][index2]

fpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024)

def fp(smiles):
  fp = all_mol[all_mol['Smile'] == smiles]['desc'].tolist()[0]
  return fp


def find_index(smile, clf=clf):
    sample = np.array([fp(smile)])
    leaf_id = clf.apply(sample)[0]
    return leaf_id


file_path = 'Results/'+ data +'_'+desc+'_'+str(n_add)+'add_'+model+'_'+acq_model+'.txt'
def load_smiles_file(filepath, mol_nr):
    with open(filepath, "r") as file:
        for line in file:
            parts = line.strip().split(";")
            if parts[0]=="file"+str(mol_nr):
                return ast.literal_eval(parts[2]), ast.literal_eval(parts[1])
        
aquired_mols,number_list = load_smiles_file(file_path, mol)
sizes = clf.tree_.n_node_samples
values = clf.tree_.value

iteration_idx = [0 for _ in range(10)] + [val for val in np.arange(50)+1 for _ in (0, 1)]


relevant_clusters = []
for it_idx, s in zip(iteration_idx, aquired_mols):
    index = find_index(s)
    if values[index][0][1] ==1:
        relevant_clusters.append((index, it_idx))

aquired_mols_index = [find_index(smile) for smile in aquired_mols]
cluster_sizes = [sizes[id] for id in range(len(clf.tree_.n_node_samples))]


unique_clusters = list(dict.fromkeys([x for x,y in relevant_clusters]))
cluster_edges = []
for i,j in itertools.combinations(unique_clusters, 2):
    if i > j:
        cluster_edges.append((i,j,find_distance_index(i,j)))
    elif i!=j:
        cluster_edges.append((j,i,find_distance_index(i,j)))

cluster_edges = list(dict.fromkeys([x for x in cluster_edges]))


start_cluster = relevant_clusters[0][0]
important_cluster_edges = cluster_edges[:5]
temp_cluster_list = important_cluster_edges
important_cluster_edges2=sorted(temp_cluster_list, key=lambda x: x[2])[:10]
rest_of_clusters = cluster_edges[5:]

closest_leaves = set(find_closest_leaf(start_cluster, 20,only_active=False)+[rc[0] for rc in relevant_clusters])
other_cluster_edges = []
for i,j in itertools.combinations(closest_leaves, 2):
    if i > j:
        other_cluster_edges.append((i,j,find_distance_index(i,j)))
    elif i!=j:
        other_cluster_edges.append((j,i,find_distance_index(i,j)))
for i,j in itertools.combinations(list(closest_leaves)+[rc[0] for rc in relevant_clusters], 2):
        if i > j:
                rest_of_clusters.append((i,j,find_distance_index(i,j)))
        elif i!=j:
                rest_of_clusters.append((j,i,find_distance_index(i,j)))

seed = 2  # Seed random number generators for reproducibility
G = nx.MultiDiGraph()

times_dist=1
cluster_size = 200

color_list = ['red', 'darkblue' , 'green', 'magenta', 'orange']
fig, ax = plt.subplots()


for x, y, w in important_cluster_edges:
    G.add_edge(x,y, len=w*times_dist)  
            

list.sort(important_cluster_edges2,key=lambda x: x[2])
concat_clusters = important_cluster_edges2 + rest_of_clusters+other_cluster_edges 
concat_clusters_unique = list(dict.fromkeys([x for x in concat_clusters]))
for x, y, w in concat_clusters_unique:
    G.add_edge(x,y, len=w+times_dist) 

for x, y, w in set(other_cluster_edges):
    G.add_edge(x,y, len=w+times_dist)  

for x, y, w in set(rest_of_clusters):
    G.add_edge(x,y, len=w+times_dist)

pos = nx.nx_agraph.graphviz_layout(G, prog='neato')


alphas = []

label_color={}
label_sizes={}
colors = []
sizes = []
for n in G.nodes:
    if n in [rc[0] for rc in relevant_clusters]:
        alphas.append(1)
        label_color[n] = 'black'
        label_sizes[n]=18
        colors.append('lightgrey')
        sizes.append(cluster_size*cluster_sizes[n]) #ALDH1 200 

    elif values[n][0][1]==1:
        alphas.append(0.2)
        label_color[n] = 'grey'
        label_sizes[n]=0
        colors.append('lightgrey')
        sizes.append(cluster_size*cluster_sizes[n])
    else:
        alphas.append(0.8)
        label_color[n] = 'grey'
        label_sizes[n]=0
        colors.append('red')
        sizes.append(10)       

cmap = plt.cm.winter
nodes = nx.draw_networkx_nodes(G, pos, node_size=sizes, margins=0.12, node_color=colors, alpha=alphas)
pos_important = {k: pos[k] for k in list(G.nodes)}
labels_important = {n: n for n in list(G.nodes)}

for n, label in labels_important.items():
    nx.draw_networkx_labels(G, {n: pos[n]}, labels={n: label},
                            font_color=label_color[n], font_size=label_sizes[n], ax=ax)



fig.set_size_inches(10, 10)
#use pm 30 in each direction when plotting actives sampled
xs = []
ys = []
cs = []
for node, idx in relevant_clusters:
    x,y = pos[node]
    min_offset = cluster_sizes[node]*cluster_size *0.05
    x_offset = (random.random()-0.5)*min_offset
    y_offset = (random.random()-0.5)*min_offset
    xs.append(x+x_offset)
    ys.append(y+y_offset)
    cs.append(idx)
p=ax.scatter(xs, ys, c=cs, zorder=2, alpha=0.5, cmap="plasma", s=150)

pos1 = pos[unique_clusters[0]]
pos2 = pos[unique_clusters[1]] 
dist = ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)**0.5

# Add scale bar
fontprops = fm.FontProperties(size=16)
scalebar = AnchoredSizeBar(ax.transData,
                           dist,  # length of the bar in data units
                           "Distance = " +str(int(important_cluster_edges[0][2])),  # label
                           'lower right',  # location
                           pad=0.8,
                           color='black',
                           frameon=False,
                           size_vertical=1.0,
                           fontproperties=fontprops)

ax.add_artist(scalebar)


plt.show()
plt.savefig("Figure/distance_figure_"+str(mol)+'_'+ data +'_'+desc+'_'+str(n_add)+'add_'+model+'_'+acq_model+".pdf")

import numpy as np

smiles_list = aquired_mols
active_nodes = []
active_nodes_size = []

for smile in smiles_list:
    sample = np.array([fp(smile)])
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value
    samples = clf.tree_.n_node_samples

    node_indicator = clf.decision_path(sample)
    leaf_id = clf.apply(sample)

    sample_id = 0
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]
    for node in node_index:
        if values[node][0][1] ==1:
            node_id = node
            break


    if values[node_index[-1]][0][1] ==1:

        active_nodes_size.append(round(samples[node_id]*values[node_id][0][1],0))
        active_nodes.append(node_id)


import matplotlib.pyplot as plt

plt.plot(np.linspace(0,100,51),[ x-1 for x in number_list])
plt.xlabel('Screened molecules')
plt.ylabel('Active molecules')

plt.ylim(top=70)
plt.show()
