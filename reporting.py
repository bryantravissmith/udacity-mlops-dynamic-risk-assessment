import pickle
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


# Function for reporting
def reporting():
    filehandler = open(f'{prod_deployment_path}/trainedmodel.pkl', 'rb')
    model = pickle.load(filehandler)
    testdata = pd.read_csv(f'{test_data_path}/testdata.csv')
    y_pred = model.predict(testdata[['lastmonth_activity', 'lastyear_activity',
                                     'number_of_employees']])
    confusion_matrix = metrics.confusion_matrix(
        testdata['exited'],
        y_pred
    )

    plt.clf()
    plt.figure(figsize=(8, 8), tight_layout=True)
    sns.heatmap(confusion_matrix, annot=True)
    plt.savefig(f'{dataset_csv_path}/confusion_matrix_2.png')


if __name__ == '__main__':
    reporting()
