import training
import scoring
import deployment
import diagnostics
import ingestion
import reporting
import os
import json
import pickle
import numpy as np
from sklearn import metrics

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)


input_folder_path = config['input_folder_path']
dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

filehandler = open(f'{prod_deployment_path}/trainedmodel.pkl', 'rb')
model = pickle.load(filehandler)

# Check and read new data
# first, read ingestedfiles.txt
ingested = open(f'{prod_deployment_path}/ingestedfiles.txt', 'r')
source_string = ingested.readline()
print(source_string)
source_files = source_string.split(':')[1].strip().split(', ')
output_string = ingested.readline()
output_file = dataset_csv_path + '/' + output_string.split(':')[1].strip()

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_files = False
current_files = [
    f'{input_folder_path}/{file}' for file in os.listdir(input_folder_path)
    if '.csv' in file
]

for file in current_files:
    if file not in source_files:
        new_files = True
        break

# Deciding whether to proceed, part 1
# if you found new data, you
# check whether the score from should proceed. otherwise, do end the process here
if new_files:
# Checking for model drift the deployed model is different from the score from the model that uses the newest ingested data
    ingestion.merge_multiple_dataframe()
    scoring.score_model()

    latest_f1_score = float(open(
        f'{prod_deployment_path}/latestscore.txt', 'r'
    ).readline())

    new_f1_score = float(open(
        f'{dataset_csv_path}/latestscore.txt', 'r'
    ).readline())


# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process
# if you found evidence for model drift, re-run the deployment.py script
    delta = 0.05
    difference = np.abs(new_f1_score - latest_f1_score)
    if (difference > delta):
        deployment.run()
        reporting.reporting()
        diagnostics.run()
