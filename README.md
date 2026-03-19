# ALWithDecisionTrees
## Before running the model
The start sets and the molecular descriptors need to be defined before running the model. To create the files used in the model, run: 
```
create_start_set.py "descriptor_type" "data"
```
and
```
create_fp.py "descriptor_type" "data"
```
The descriptor type can be "1024", "512", "MACCS", "desc", or "MQN", and the data can be "ALDH1", "PKM2", or "VDR".
## Run model
Run the model with the created start sets and the molecular descriptors.
```
run_model.py "descriptor_type" "data" acc_size
```
The "acc_size" is the acquisition size, and should be a number under 100. The model is run by default with a Random Forest and the Greedy acquisition function, but this can be changed in the code.
The code creates the folder "Results" and a file in the folder with the number of active molecules found per iteration, and a list of the checked molecules per start set.
## Analyse Results
### Create decision tree
To create a decision tree for a data set and a molecular descriptor, run:
```
create_tree.py "descriptor_type" "data"
```
This creates the decision tree and a distance Matrix made from the tree. These are saved in the Data folder
### Create figures
To create a scatterplot of the size and depth of the active leaves in the decision tree, run:
```
figure_scatter.py "descriptor_type" "data" acc_size
```
This prints the scatterplot, which is created by default for the Random Forest and the Greedy acquisition function results, but this can be changed in the code. The leaves that contribute most to the number of active molecules found per start set are marked in red in the scatterplot. Besides this, the leaf ID, start set ID, size, and depth are printed for the leaves with the largest contribution to the number of found active molecules per start set.

The distance figure is created by:
```
figure_distance.py "descriptor_type" "data" acc_size start_set_number
```
The start set is numbered from 0 to 9, and the figure is created for a specific run. This returns a figure of the active leaves visited in the run, how many molecules are found in the leaf, and the distance between these leaves. It also prints a plot of the number of found active molecules per iteration for the run.

