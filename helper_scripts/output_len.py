import os
import csv

csv_folder = './data/outduo'
output_txt = './data/output.txt'

csv_files = [file for file in os.listdir(csv_folder) if file.endswith('.csv')]

with open(output_txt, 'w') as outfile:
    for csv_file in csv_files:
        with open(os.path.join(csv_folder, csv_file), 'r') as infile:
            csv_reader = csv.reader(infile)
            first_row = next(csv_reader)
            first_number = float(first_row[0])
            rounded_number = round(first_number)
            outfile.write(f"{csv_file}: {rounded_number}\n")
