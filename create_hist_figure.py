
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import ast
import sys

desc = sys.argv[1]
data = sys.argv[2]
n_add = sys.argv[3]
model = "RF"
acq_model = 'greedy'

def load_data(data, desc_type):
    with open(f'Data/{data}/{data}_actives_smiles{desc_type}.smi') as f:
        actives = pd.DataFrame(
            [[line.split(':')[0], json.loads(line.split(':')[1]), 1] for line in f],
            columns=['Smile', 'desc', 'Active']
        )
    all_mol = pd.concat([actives], ignore_index=True)
    all_mol["desc"] = all_mol["desc"].apply(lambda x: np.array(x))
    X = pd.DataFrame(all_mol["desc"].tolist())
    y = all_mol["Active"]

    return X, y, all_mol

def open_start_mol(start_molecules_file):
  with open(os.path.join( start_molecules_file)) as f:
    start_mols=[lines.split(';')[0] for lines in f.readlines()[1:]]
  list_index =[all_mol[all_mol['Smile'] == mol].index[0] for mol in start_mols]
  smiles = all_mol.iloc[list_index]['Smile'].tolist()
  return (smiles[0], smiles[1:])

def fp(smiles):
  fp = all_mol[all_mol['Smile'] == smiles]['desc'].tolist()[0]
  return fp

def print_start_fp(start_file):
  active,inactive=open_start_mol(start_file)
  active_index=fp(active)
  return active_index

def build_plot_data(clf, start_mols, values, sizes, node_depth, is_leaves, n_nodes, idx):
    leaf_id_for_visit = []
    nodes_sizes = []
    nodes_depth = []
    color_list = []
    size_list = []
    alpha_list = []
    feature = clf.tree_.feature

    for smile, id in zip(start_mols, idx):
        if smile in all_mol['Smile'].tolist():
            sample = np.array([fp(smile)])
            leaf_id = clf.apply(sample)[0]
            leaf_id_for_visit.append(leaf_id)
            print("leafId: "+str(leaf_id), str(id), "Size: "+str(sizes[leaf_id]* values[leaf_id][0][1]), "Depth: "+ str(node_depth[leaf_id]))

    for i in range(n_nodes):
        active_count = values[i][0][1]
        if active_count ==1.0 and feature[i] == -2 or i in leaf_id_for_visit:
            nodes_sizes.append(sizes[i] * active_count)
            nodes_depth.append(node_depth[i])
            if i in leaf_id_for_visit:
                color_list.append('red')
                size_list.append(20)
                alpha_list.append(1)
            else:
                color_list.append('blue')
                size_list.append(10)
                alpha_list.append(0.1)

    # Sort: active (red) last
    zipped = list(zip(nodes_sizes, nodes_depth, color_list, size_list, alpha_list))
    zipped.sort(key=lambda x: x[2] == 'red')
    return map(list, zip(*zipped))

def load_smiles_file(filepath):
    with open(filepath, "r") as file:
        number_list_list =[]
        smiles_list_list = []
        idx = []
        for line in file:
            parts = line.strip().split(";")
            idx.append(parts[0])
            number_list_list.append(ast.literal_eval(parts[1]))
            smiles_list_list.append(ast.literal_eval(parts[2])[-100:])

    return idx,number_list_list, smiles_list_list

# === Main ===
X, y, all_mol = load_data(data, desc)

clf = load("Data/decision_tree_" + data + "_" + desc + ".joblib")
file_path = file_path = 'Results/'+ data +'_'+desc+'_'+str(n_add)+'add_'+model+'_'+acq_model+'.txt'
idx, number_list_list, smiles_list_list = load_smiles_file(file_path)
active_smiles_list_list = []
for smiles_list in smiles_list_list:
    active_smiles_list_list.append([smile for smile in smiles_list if smile in all_mol['Smile'].tolist()])
smiles_list_list = active_smiles_list_list
highest_found_number = []
found_total = []
ids = []
smiles_higestest = []
for i in range(len(number_list_list)):
    dict={}

    start_mol = i
    
    smiles_list = smiles_list_list[start_mol]
    active_nodes = []
    active_nodes_size = []

    if number_list_list[start_mol][-1] > 1:
        for smile in smiles_list:
            try:
                sample = np.array([fp(smile)])
                node_id = clf.apply(sample)
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
                    if feature[node] == -2:
                        node_id = node
                        break


                # check if value of the split feature for sample 0 is below threshold
                if sample[sample_id, feature[node_id]] <= threshold[node_id]:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"

                if feature[node] == -2 and values[node_id][0][1] > 0.0:

                    active_nodes_size.append(int(round(samples[node_id]*values[node_id][0][1],0)))
                    active_nodes.append(node_id)
                    if node_id not in dict:
                        dict[node_id] = smile
            except:
                continue


        sorted_found = sorted([(x, active_nodes.count(x)) for x in set(active_nodes)], key=lambda x: x[1], reverse=True)
        if len(sorted_found) == 0:
            continue
        else:
            highest_found_number.append(sorted_found[0][1])
            ids.append(sorted_found[0][1])
            smiles_higestest.append(dict[sorted_found[0][0]])
            found_total.append([x-1 for x in number_list_list[start_mol]][1::][-1])


start_mols = smiles_higestest
n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
values = clf.tree_.value
sizes = clf.tree_.n_node_samples

node_depth = np.zeros(n_nodes, dtype=np.int64)
is_leaves = np.zeros(n_nodes, dtype=bool)
stack = [(0, 0)]
while stack:
    node_id, depth = stack.pop()
    node_depth[node_id] = depth
    if children_left[node_id] != children_right[node_id]:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True

# Build plot data

sizes_plot, depths_plot, colors, sizes_dots, alphas = build_plot_data(
    clf, start_mols, values, sizes, node_depth, is_leaves, n_nodes, idx
)


# Plot
plt.scatter(sizes_plot, depths_plot, alpha=alphas, color=colors, s=sizes_dots)
plt.xlabel("Active in leaf", fontsize=18)
plt.ylabel("Depth" , fontsize=18)
title = f"{data} - {'MACCS' if desc == 'MACCS' else 'ECFP4'}"
plt.title(title, fontsize=20)
new_f = open(f'Results/{data}_{desc}_depth_size.txt','w')
new_f.write(data+','+desc+':'+str(sizes_plot)+';'+str(depths_plot)+';'+str(alphas)+';'+str(colors)+';'+str(sizes_dots)+'\n')
new_f.close()
plt.rcParams.update({'font.size': 16})

plt.tight_layout()
plt.show()

