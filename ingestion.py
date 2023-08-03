import pandas as pd
import os
import json
from datetime import datetime


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    dataframe = pd.DataFrame()
    f = open(f'{output_folder_path}/ingestedfiles.txt', 'w')
    f.write('Sources: ')
    for i, file in enumerate(os.listdir(input_folder_path)):
        if i != 0:
            f.write(', ')
        path = os.path.join(input_folder_path, file)
        dataframe = dataframe.append(pd.read_csv(path))
        f.write(path)

    f.write('\n')

    dataframe = dataframe.drop_duplicates()

    if dataframe.shape[0] > 0:
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)
        dataframe.to_csv(os.path.join(
            output_folder_path,
            'finaldata.csv'
        ), index=False)

        f.write('output: finaldata.csv\n')
        f.write(f'rows: {len(dataframe.index)}\n')
        f.write(f'timestamp: {str(datetime.now())}\n')


if __name__ == '__main__':
    merge_multiple_dataframe()
