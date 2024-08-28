#!/bin/bash

# Activate the Python environment
source activate 2DIR

# Run the model/pre.py script
python model/pre.py

# Optional: Keep the terminal open after the script completes
exec "$SHELL"
