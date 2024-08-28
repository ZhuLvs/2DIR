#!/bin/bash


# Set the paths to Length_main.py and main.py files
LENGTH_MAIN_PY_PATH="model/Length_main.py"
MAIN_PY_PATH="model/main.py"

# Check if Length_main.py file exists
if [ ! -f "$LENGTH_MAIN_PY_PATH" ]; then
    echo "Error: $LENGTH_MAIN_PY_PATH not found."
    exit 1
fi

# Check if main.py file exists
if [ ! -f "$MAIN_PY_PATH" ]; then
    echo "Error: $MAIN_PY_PATH not found."
    exit 1
fi

# Run the Length_main.py script
python "$LENGTH_MAIN_PY_PATH"

# Run the main.py script
python "$MAIN_PY_PATH"

echo "Script execution completed."
