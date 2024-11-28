#!/bin/bash



if [ ! -d "data/valA" ]; then
    echo "Creating data/valA folder..."
    mkdir -p data/valA
else
    echo "data/valA already exists."
fi

if [ ! -d "data/valB" ]; then
    echo "Creating data/valB folder..."
    mkdir -p data/valB
else
    echo "data/valB already exists."
fi

echo "Running Calculate Distance Map.py..."
python3 helper_scripts/Calculate\ Distance\ Map.py

echo "Running Padding.py..."
python3 helper_scripts/Padding.py

echo "Running Dataset_split.py..."
python3 helper_scripts/Dataset_split.py

echo "Running move.py..."
python3 helper_scripts/move.py

echo "All tasks completed."

