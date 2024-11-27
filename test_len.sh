#!/bin/bash

# Activate the Python environment
source activate 2DIR

# Run pre.py and pre_length.py in the model directory
python model/pre.py
python model/pre_length.py

# Run the scripts in the helper_scripts directory
python helper_scripts/output_len.py
python helper_scripts/Matrix_trim.py
python helper_scripts/Add_header.py

# Optional: Keep the terminal open after the script completes
exec "$SHELL"
