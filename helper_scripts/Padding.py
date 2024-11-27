
#The purpose of padding is to standardize the size, making it easier for the model to process.
# The size is set by default to the length of the longest protein, but you can modify the target_size parameter as needed.


import os
import pandas as pd
import numpy as np

def process_csv(file_path, target_size=99):

    data = pd.read_csv(file_path, skiprows=1, header=None)
    current_shape = data.shape

    if current_shape != (target_size, target_size):

        padded_data = np.zeros((target_size, target_size))

        padded_data[:current_shape[0], :current_shape[1]] = data.values
        return pd.DataFrame(padded_data)
    else:
        return data

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            processed_data = process_csv(file_path)
            processed_data.to_csv(os.path.join(output_folder, filename), index=False)

input_folder = './data/contact'
output_folder = './data/contact'

process_folder(input_folder, output_folder)
