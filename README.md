<h1>AI Protocol for Retrieving Protein Dynamic Structures from Two-Dimensional Infrared Spectra</h1>



 

:bell:  **News**

Our latest dataset now contains 49,547 different proteins, all sourced from RCSB and SWISSPROT (AFDB-SWISSPROT). You can find them in the Quick Start section below, it includes the Two-Dimensional Infrared Spectroscopy (2DIR) data and PDB data for all proteins. You will need to randomly split the dataset into training and test sets yourself!
Training the model from scratch takes approximately five minutes!


We tested the model's ability to transfer to proteins with 100 to 150 residues, and the complete dataset link is: [this link](https://zenodo.org/records/14561794). 






2DIR：Predicting protein dynamic structures using two-dimensional infrared spectroscopy, with unknown structures.

![2DIR_model](img/2dir_image.png)

## Requirements
Operating System: Linux (Recommended)  
No non-standard hardware is required.

## Getting started
To get started using 2DIR, clone the repo:
```bash
git clone https://github.com/ZhuLvs/2DIR.git
cd 2DIR
```
## Install Dependencies
```bash
conda create -n 2DIR python=3.8
conda activate 2DIR
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r ./install/requirements.txt

```



## Quick Start

🚀 First, you need to download the training dataset for 2DIR. [this link](https://zenodo.org/records/14233899)  and save it under the `data` directory.
Training dataset：  `data/2DIR` `data/contact` (distance map),
Validation dataset：   `data/valA` `data/valB`    (distance map).
Additional test datasets can be added by the user!

The PDBFliess dataset link is [this link](https://zenodo.org/records/14233904). 

Then, you calculate the distance matrix (distance map) between the CA atoms of each residue in the protein PDB structure file. The calculation script can be found in the  `helper_scripts`  directory.

You may manually modify the parameters in `model/main.py`.


## Preparation before starting

You need to download the 2DIR and PDB data using the link above, then unzip and place them in the `data` folder (data/2DIR/xxx.png, data/PDB/xxx.pdb). After that, run the script below, and it will automatically complete the data preprocessing.

```bash
bash Preprocessing.sh
```

## Training
Before training begins, the protein residue distance matrix needs to be padded to ensure uniform size, which facilitates model processing, accelerates training, and so on. The padding code can be found in the `helper_scripts` directory and can be modified as needed（Running Preprocessing.sh will handle this process）.

### Known Length Protein
```bash
bash train.sh
```

### Unknown Length Protein
```bash
bash train_unknown_protein.sh
```
## Inference
After inference is complete, the predicted results need to be trimmed based on the protein length, following the format provided in `data/output.txt`.
### Known Length Protein
```bash
bash test.sh
```
### Unknown Length Protein
For proteins with unknown lengths, you need to run `model/pre_length.py` to predict the protein length, and then refer to the scripts in the `helper_scripts` directory for trimming and processing.
```bash
bash test_len.sh
```



To generate the protein backbone structure from the protein residue distance matrix, please use the gradient descent algorithm available in the `PyRosetta` protocols, providing the predicted residue distances from the model and constraints from `residue_constants.py`.

For obtaining the amino acid sequence of unknown proteins, it is recommended to use the backbone structure as input for `ProteinMPNN`. The model typically converges after around 300 epochs.
