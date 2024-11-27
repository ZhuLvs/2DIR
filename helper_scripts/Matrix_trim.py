import os
import csv

txt_file_path = './data/output.txt'
folder_path = './data/out/'

with open(txt_file_path, 'r') as txt_file:
    lines = txt_file.readlines()

for line in lines:
    file_name, length = line.strip().split(':')
    length = int(length)
    csv_file_path = os.path.join(folder_path, file_name)

    if os.path.isfile(csv_file_path):
        with open(csv_file_path, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            data = list(csvreader)

        if len(data) > length:
            data = [row[:length] for row in data[:length]]

            with open(csv_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(data)

            print(f"Trimmed matrix '{file_name}' to {length}x{length}.")
        else:
            print(f"The matrix size of '{file_name}' is smaller than the specified length {length}, no operation performed.")
    else:
        print(f"The file '{file_name}' does not exist.")

print('Processing completed!')
