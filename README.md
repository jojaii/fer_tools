# fer_tools

Fer tools contains scripts for 
1. dataset aggregation,
2. model training and testing, and
3. embedding generation of the different Facial Expression Recognition datasets.

The architecture currently being used for this implementation is [miniXception](https://github.com/oarriaga/face_classification) by github user [oarriaga](https://github.com/oarriaga).   

## Main files
The root folder contains the scripts for:
1. Splitting and converting datasets (split.py)
2. Training miniXception models (train.py)
3. Inferencing using the trained models (test.py)

Variables can be changed in the (settings.py) file.

## Directories

| Directory     | Contents      |
| ------------- |:-------------|
| models      | Trained models |
| data      | Dataset csvs     |
| utils | Utility files |
| vis | Jupyter notebook visualization files
| temp | Backup of files being cleaned      |
| bin | Dump for other files      |


## Datasets
Below are the supported datasets:

| Dataset     | Description      |
| ------------- |:-------------|
| FER2013      | Grayscale 64x64 images stored in csv format |
| FERcolored      | FER2013 recolored using a neural network. Colored 64x64 images|
| AffectNet | Colored webscraped images |
| *ExpW |     |
| *SFEW |     |
| *KDEF |  |

*Descriptions to be updated
