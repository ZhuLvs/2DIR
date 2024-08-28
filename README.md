# 2DIR

2DIR: Predicting protein dynamic structures using two-dimensional infrared spectroscopy, with unknown structures.

## Requirements
Operating System: Linux (Recommended)  
No non-standard hardware is required.

## Install Dependencies
```bash
conda create -n 2DIR python=3.8
conda activate 2DIR

pip install -r ./install/requirements.txt

## Quick Start

First, you need to download the training dataset. You can download it from [this link](https://github.com/ZhuLvs/2DIR/tree/main) and save it under the `data` directory. You may manually modify the parameters in `model/main.py`.

## Training

### Known Length Protein
```bash
bash train.sh

### Known Length Protein
```bash
bash train_unknown_protein.sh
