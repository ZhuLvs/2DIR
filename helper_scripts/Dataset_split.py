

#  Here is a 90/10 split between the training set and validation set; you can modify it as needed.

import os
import shutil
import random

source_folder = '../data/2DIR'
testA_folder = '../data/valA'

file_list = os.listdir(source_folder)

total_files = len(file_list)
num_files_to_move = int(total_files * 0.1)

files_to_move = random.sample(file_list, num_files_to_move)

for file_name in files_to_move:
    source_path = os.path.join(source_folder, file_name)
    dest_path = os.path.join(testA_folder, file_name)
    shutil.move(source_path, dest_path)

