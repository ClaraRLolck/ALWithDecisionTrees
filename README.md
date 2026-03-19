# ALWithDecisionTrees
## Before running the model
Run 
```
create_start_set.py "descriptor_type" "data"
```
and
```
create_fp.py "descriptor_type" "data"
```
Where the descriptor type could be "1024", "512", "MACCS", "desc", or "MQN", and data is "ALDH1", "PKM2", or "VDR".
## Run model
Run the model with the created start sets and the molecular descriptors.
```
run_model.py "descriptor_type" "data" acc_size
```
The "acc_size" is the acquisition size, and should be a number under 100. The model is in default run with Random Forest and the acquisition function Greedy, but this can be changed in the code.
The code creates the folder "Results" and a file in the folder with the number of active molecules found per iteration, and a list of the checked molecules per start set.
## Analyse Results
### Create decision tree
To create a decision tree for a data set and a molecular descriptor, run:
```
create_tree.py "descriptor_type" "data"
```
This creates the decision tree and a distance Matrix made from the tree. These are saved in the Data folder
### Create figures
