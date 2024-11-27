import pandas as pd
import os

folder_path = './data/out'

csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, header=None)
    column_names = list(range(df.shape[1]))
    df.columns = column_names
    df.to_csv(file_path, index=False)
