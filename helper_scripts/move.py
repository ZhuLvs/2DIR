import os
import shutil

folder_a = '../data/valA'
folder_b = '../data/contact'
folder_c = '../data/valB'  #B -> C

a_file_prefixes = set()
for filename in os.listdir(folder_a):
    file_prefix = os.path.splitext(filename)[0]
    a_file_prefixes.add(file_prefix)

for filename in os.listdir(folder_b):
    file_prefix = os.path.splitext(filename)[0]
    if file_prefix in a_file_prefixes:
        source_path = os.path.join(folder_b, filename)
        target_path = os.path.join(folder_c, filename)
        shutil.move(source_path, target_path)
        print(f"Moved {filename} to {folder_c}")

print("File move complete.")
