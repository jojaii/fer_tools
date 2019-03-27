# fer_tools

Fer tools contains scripts for 
1. dataset aggregation,
2. model training and testing, and
3. embedding generation of the different Facial Expression Recognition datasets.

The architecture currently being used for this implementation is miniXception by github user oarriaga.   

The root folder contains the scripts for:
1. Splitting datasets (split.py)
2. Training miniXception models (train.py)
3. Inferencing using the trained models ()

Variables can be changed in the (settings.cfg) file.

Other directory contents:
(models) Trained models 
(data)   Dataset csvs
(utils)  Utility files
(vis)    Jupyter notebook visualization files
(temp)   Backup of files being cleaned
(bin)    Dump for other files