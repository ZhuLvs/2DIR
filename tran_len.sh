#!/bin/bash

# Set the path to the main.py file
MAIN_PY_PATH="model/Length_main.py"

# Check if the main.py file exists
if [ ! -f "$MAIN_PY_PATH" ]; then
    echo "Error: $MAIN_PY_PATH not found."
    exit 1
fi

# Run the Python script
python "$MAIN_PY_PATH"
echo "Script execution completed."
