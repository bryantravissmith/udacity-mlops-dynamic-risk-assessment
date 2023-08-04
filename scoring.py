import pandas as pd
import pickle
import os
import numpy as np
from sklearn import metrics
import json


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


# Function for model scoring
def score_model():
    data = pd.read_csv(f"{test_data_path}/testdata.csv")
    y = data.pop('exited')
    x = data[['lastmonth_activity', 'lastyear_activity',
              'number_of_employees']]

    filehandler = open(f'{model_path}/trainedmodel.pkl', 'rb')
    model = pickle.load(filehandler)

    y_pred = model.predict(x)
    f1_score = np.round(metrics.f1_score(y, y_pred), 4)

    f = open(f'{dataset_csv_path}/latestscore.txt', 'w')

    f.write(f'{f1_score}')


if __name__ == '__main__':
    score_model()
